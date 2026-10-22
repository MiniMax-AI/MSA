"""Independent FP32 MoE reference and numerical acceptance checks."""

import torch

T, H, I, E, K, R, WORLD = (8, 6144, 768, 128, 4, 32, 4)


def mx_decode(x, sf):
    values = x.contiguous().view(torch.float8_e4m3fn).float()
    scales = torch.pow(2.0, sf.float() - 127).repeat_interleave(32, dim=-1)
    return values * scales


def nv_decode(x, sf):
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=x.device,
    )
    codes = torch.stack([x & 15, x >> 4], dim=-1).reshape(*x.shape[:-1], x.shape[-1] * 2).long()
    scales = sf.contiguous().view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return lut[codes] * scales


def mx_quant(x):
    z = x.reshape(*x.shape[:-1], -1, 32)
    amax = z.abs().amax(dim=-1)
    desired = amax / 448.0
    positive = desired > 0
    # UE8M0 rounding includes the subnormal/code-zero rule.
    bits = desired.contiguous().view(torch.int32)
    exponent = bits >> 23 & 255
    mantissa = bits & 8388607
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 4194304))
    code = (exponent + bump.to(torch.int32)).clamp_max(254).to(torch.uint8)
    code = torch.where(positive, code, torch.zeros_like(code))
    inv = torch.where(code > 0, torch.pow(2.0, 127.0 - code.float()), torch.zeros_like(desired))
    q = (
        (z * inv.unsqueeze(-1))
        .flatten(-2)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    return (q, code)


def nv_quant(x):
    z = x.reshape(*x.shape[:-1], -1, 16)
    amax = z.abs().amax(dim=-1)
    scales = (amax / 6).clamp_max(448)
    sf = scales.to(torch.float8_e4m3fn).view(torch.uint8)
    rt = sf.view(torch.float8_e4m3fn).float()
    inv = torch.where(rt > 0, 1.0 / rt, torch.zeros_like(rt))
    q = (z * inv.unsqueeze(-1)).flatten(-2)
    a = q.abs()
    # E2M1 quantization rounds midpoint ties to the even adjacent code.
    edges = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    code = torch.bucketize(a.contiguous(), edges, right=False).to(torch.uint8)
    for j in range(7):
        if j & 1:
            code = torch.where(a == edges[j], torch.full_like(code, j + 1), code)
    code = code | torch.where(
        (q < 0) & (code != 0), torch.full_like(code, 8), torch.zeros_like(code)
    )
    packed = code[..., ::2] | code[..., 1::2] << 4
    return (packed, sf)


def bf16_round(x):
    return x.bfloat16().float()


def activation(gate, up):
    gate = gate.clamp_max(7)
    up = up.clamp(-7, 7)
    return (up + 1) * gate * torch.sigmoid(1.702 * gate)


def shared_activation(gate, up):
    """msa_v1 Shared SwiGLU: BF16 gate/up inputs and a BF16 result."""
    return bf16_round(activation(bf16_round(gate), bf16_round(up)))


def local_partial(weights, hidden, fc2_alpha):
    """Return this rank's BF16 partial and its active expert count.

    fc2_alpha includes the routing multiplier, as applied by the runner.
    """
    device = hidden.device
    x = hidden.float()
    raw = torch.sigmoid(x @ weights.router_weight.T)
    ids = torch.argsort(
        raw + weights.router_correction_bias, dim=-1, descending=True, stable=True
    )[..., :K]
    selected = raw.gather(-1, ids)
    denom = selected.sum(-1, keepdim=True)
    route_weights = selected / torch.where(denom > 0, denom, torch.ones_like(denom))
    shared_input = mx_decode(*mx_quant(x))
    routed_input = nv_decode(*nv_quant(x))
    fc1 = mx_decode(weights.shared_fc1_weight.view(torch.uint8), weights.shared_fc1_weight_sf)
    shared_dot = shared_input @ fc1.T
    shared_mid = mx_decode(*mx_quant(shared_activation(shared_dot[:, :I], shared_dot[:, I:])))
    fc2 = mx_decode(weights.shared_fc2_weight.view(torch.uint8), weights.shared_fc2_weight_sf)
    shared = shared_mid @ fc2.T
    routed = torch.zeros_like(shared)
    flat_ids = ids.flatten()
    token_ids = torch.arange(T, device=device).repeat_interleave(K)
    for e in sorted(set(flat_ids.cpu().tolist())):
        routes = torch.nonzero(flat_ids == e, as_tuple=True)[0]
        w1 = nv_decode(weights.routed_fc1_weight[e], weights.routed_fc1_weight_sf[e])
        dot = routed_input[token_ids[routes]] @ w1.T * weights.routed_fc1_alpha[e]
        mid = nv_decode(*nv_quant(activation(dot[:, :I], dot[:, I:])))
        w2 = nv_decode(weights.routed_fc2_weight[e], weights.routed_fc2_weight_sf[e])
        # msa_v1 rounds each alpha-scaled route output and its weight to BF16.
        down = bf16_round(mid @ w2.T * fc2_alpha[e])
        weighted = down * bf16_round(route_weights.flatten()[routes]).unsqueeze(-1)
        routed.index_add_(0, token_ids[routes], weighted)
    return (bf16_round(shared) + routed).bfloat16(), int(torch.unique(ids).numel())


def ordered_output(partials, residual):
    """Rank-ordered FP32 TP4 sum, BF16, plus residual, BF16."""
    total = torch.zeros((T, H), dtype=torch.float32, device=residual.device)
    for partial in partials:
        total += partial.to(residual.device).float()
    return (total.bfloat16().float() + residual.float()).bfloat16()
