"""Reference of the MiniMax-M3.1 DSpark draft backbone on TP4 (one request's seven-row block).

Each step follows the SGLang operator that runs it on the standard draft path and keeps that
operator's BF16 rounding points:

- embedding: vocab-parallel lookup + TP sum (exact: one owner per token)
- norms: Gemma RMSNorm, x * rsqrt(mean(x^2) + eps) * (1 + w) with 1 + w in FP32; at a TP
  boundary x = bf16(TP sum) + residual (unrounded) and the residual stream keeps bf16(x)
- linear: MXFP8 activation quantization (UE8M0 per 32 values, E4M3 RNE) x MXFP8 weight -> BF16
- q/k norm + RoPE: per-head Gemma norm, NeoX rotation of dims 0..63, BF16
- attention: every block row attends the last min(kv_len, 4096) cached keys (the block is
  bidirectional), FP32 softmax, BF16 output
- FFN: clamped SwiGLU-OAI rounded to BF16, then MXFP8
- TP4 sums: BF16 partials summed in rank order in FP32, rounded to BF16

``Reference(torch.float64)`` is the exact reference. ``Reference(torch.float32, chunked=True)``
is an independent FP32 implementation with flash-style attention (BF16 probabilities per
128-key chunk); its distance to the exact reference is the accuracy bound a kernel is held to,
since every legitimate rounding difference is amplified by the MXFP8 requantization of the
next layer.
"""

import torch

LAYERS, TP, ROWS, HIDDEN, HEAD_DIM, ROTARY_DIM = 5, 4, 7, 6144, 128, 64
Q_HEADS, INTER, WINDOW, PAGE, BLOCK = 16, 3072, 4096, 128, 32
ALPHA, LIMIT = 1.702, 7.0
CHUNK = 128


def e8m0_scale(amax):
    """The UE8M0 exponent of a group amax: the smallest power of two bringing it within 448."""
    bits = amax.to(torch.float32).view(torch.int32)
    e0 = ((bits >> 23) & 0xFF) - 135
    threshold = (((e0 + 135) << 23) | 0x600000).view(torch.float32)
    e = e0 + (amax.to(torch.float32) > threshold).to(torch.int32)
    return torch.clamp(e, min=-127)


class Reference:
    def __init__(self, acc=torch.float64, chunked=False):
        self.acc, self.chunked = acc, chunked

    def bf16(self, x):
        return x.to(torch.bfloat16).to(self.acc)

    def mxfp8_quantize(self, x):
        """Rows of x (BF16-representable) -> (E4M3 codes, UE8M0 exponents)."""
        rows, k = x.shape
        groups = x.view(rows, k // BLOCK, BLOCK)
        e = e8m0_scale(groups.abs().amax(-1).to(torch.float32))
        scaled = groups.to(torch.float32) * torch.exp2(-e.to(torch.float32))[..., None]
        return scaled.to(torch.float8_e4m3fn).to(self.acc).view(rows, k), e

    def mxfp8_dequant(self, codes, exponents):
        rows, k = codes.shape
        scale = torch.exp2(exponents.to(self.acc))
        return (codes.view(rows, k // BLOCK, BLOCK) * scale[..., None]).view(rows, k)

    def linear(self, x, weight, weight_sf):
        """MXFP8 linear of BF16 activations x [T, K] and an MXFP8 weight [N, K] -> BF16 [T, N]."""
        a = self.mxfp8_dequant(*self.mxfp8_quantize(x))
        w = self.mxfp8_dequant(weight.to(self.acc), weight_sf.to(torch.int32) - 127)
        return self.bf16(a @ w.T)

    def gemma_norm(self, x, w, eps):
        variance = (x * x).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + eps) * (1.0 + w.to(self.acc))

    def tp_sum(self, partials):
        total = partials[0]
        for p in partials[1:]:
            total = total + p
        return self.bf16(total)

    @staticmethod
    def rope(x, cos, sin):
        """NeoX rotation of dims [0, 64) of [T, H, 128]; cos/sin [T, 32]."""
        x1, x2 = x[..., : ROTARY_DIM // 2], x[..., ROTARY_DIM // 2 : ROTARY_DIM]
        c, s = cos[:, None, :], sin[:, None, :]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, x[..., ROTARY_DIM:]], dim=-1)

    def window(self, cache, page_table, kv_len):
        start = max(0, kv_len - WINDOW)
        keys = torch.arange(start, kv_len, device=cache.device)
        slots = page_table[keys // PAGE].to(torch.int64) * PAGE + keys % PAGE
        return cache[slots].to(self.acc)

    def attention(self, q, k_window, v_window):
        """q [T, 16, 128], k/v [S, 128] (one KV head) -> BF16 [T, 16 * 128]."""
        scores = torch.einsum("thd,sd->ths", q, k_window) * HEAD_DIM**-0.5
        if not self.chunked:
            probs = torch.softmax(scores, dim=-1)
            return self.bf16(torch.einsum("ths,sd->thd", probs, v_window)).reshape(q.shape[0], -1)
        # Flash form: per key chunk BF16 P = exp(s - chunk max) feeds PV; chunks merge exactly.
        outs, maxes, sums = [], [], []
        for c0 in range(0, scores.shape[-1], CHUNK):
            s = scores[..., c0 : c0 + CHUNK]
            m = s.amax(-1, keepdim=True)
            p = torch.exp(s - m)
            sums.append(p.sum(-1, keepdim=True))
            outs.append(torch.einsum("ths,sd->thd", self.bf16(p), v_window[c0 : c0 + CHUNK]))
            maxes.append(m)
        m = torch.stack(maxes).amax(0)
        num = sum(o * torch.exp(mi - m) for o, mi in zip(outs, maxes))
        den = sum(l * torch.exp(mi - m) for l, mi in zip(sums, maxes))
        return self.bf16(num / den).reshape(q.shape[0], -1)

    def swiglu_oai(self, gate_up):
        gate, up = gate_up[:, :INTER], gate_up[:, INTER:]
        gate = gate.clamp(max=LIMIT)
        up = up.clamp(-LIMIT, LIMIT)
        return self.bf16(gate * torch.sigmoid(gate * ALPHA) * (up + 1.0))

    def forward(self, ranks, step, caches):
        """The TP4 draft step.

        ranks: TP x dict(layers=[dict], embed, vocab_start, final_norm, eps)
        step: dict(input_ids, positions, slot_mapping, page_table, kv_len, cos_sin)
        caches: TP x LAYERS x (k, v) BF16 [slots, 128], updated in place with the block's rows.
        Returns the final-normed hidden [7, 6144] (BF16 values), common to every rank.
        """
        ids, positions, slots = step["input_ids"], step["positions"], step["slot_mapping"]
        kv_len = int(step["kv_len"].item())
        cos = step["cos_sin"][positions][:, : ROTARY_DIM // 2].to(self.acc)
        sin = step["cos_sin"][positions][:, ROTARY_DIM // 2 :].to(self.acc)
        eps = ranks[0]["eps"]

        partials = []
        for r in ranks:
            local = ids - r["vocab_start"]
            owned = (local >= 0) & (local < r["embed"].shape[0])
            rows = r["embed"][local.clamp(0, r["embed"].shape[0] - 1)].to(self.acc)
            partials.append(rows * owned[:, None])
        residual = self.tp_sum(partials)
        hidden = self.bf16(self.gemma_norm(residual, ranks[0]["layers"][0]["input_norm"], eps))

        for layer in range(LAYERS):
            partials = []
            for rank, r in enumerate(ranks):
                p = r["layers"][layer]
                qkv = self.linear(hidden, p["qkv"], p["qkv_sf"])
                q = qkv[:, : Q_HEADS * HEAD_DIM].view(ROWS, Q_HEADS, HEAD_DIM)
                k = qkv[:, Q_HEADS * HEAD_DIM : (Q_HEADS + 1) * HEAD_DIM].view(ROWS, 1, HEAD_DIM)
                v = qkv[:, (Q_HEADS + 1) * HEAD_DIM :]
                q = self.bf16(self.rope(self.gemma_norm(q, p["q_norm"], eps), cos, sin))
                k = self.bf16(self.rope(self.gemma_norm(k, p["k_norm"], eps), cos, sin)).view(ROWS, HEAD_DIM)
                k_cache, v_cache = caches[rank][layer]
                valid = slots >= 0
                k_cache[slots[valid]] = k[valid].to(torch.bfloat16)
                v_cache[slots[valid]] = v[valid].to(torch.bfloat16)
                out = self.attention(q, self.window(k_cache, step["page_table"], kv_len),
                                     self.window(v_cache, step["page_table"], kv_len))
                partials.append(self.linear(out, p["o"], p["o_sf"]))
            summed = self.tp_sum(partials) + residual
            residual = self.bf16(summed)
            hidden = self.bf16(self.gemma_norm(summed, ranks[0]["layers"][layer]["post_norm"], eps))

            partials = []
            for r in ranks:
                p = r["layers"][layer]
                gate_up = self.linear(hidden, p["gate_up"], p["gate_up_sf"])
                partials.append(self.linear(self.swiglu_oai(gate_up), p["down"], p["down_sf"]))
            summed = self.tp_sum(partials) + residual
            residual = self.bf16(summed)
            last = layer + 1 == LAYERS
            w = ranks[0]["final_norm"] if last else ranks[0]["layers"][layer + 1]["input_norm"]
            hidden = self.bf16(self.gemma_norm(summed, w, eps))
        return hidden
