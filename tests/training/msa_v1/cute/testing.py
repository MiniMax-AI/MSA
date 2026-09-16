"""Independent FP32 references for MSA v1 CuTe release tests."""

from __future__ import annotations

import math
import signal
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

import torch

from datas.training.cases import (
    CpRankCase,
    make_torch_metadata,
    make_torch_topk,
)
from tests.training.msa_v1.cute.cases import MsaTestCase

HEAD_DIM = 128
Q_HEADS = 64
KV_HEADS = 4
INDEX_HEADS = 4
BLOCK_SIZE = 128
TOPK = 16
_FIRST_LAUNCHES: set[str] = set()
T = TypeVar("T")


def case_seed(item: MsaTestCase, salt: int) -> int:
    case = item.rank_case
    return 1_000_003 * item.item_id + 10_007 * case.case_id + 101 * case.rank + salt


def cuda_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def random_bf16(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator,
    scale: float = 0.2,
) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=torch.bfloat16, device=device)
    tensor.normal_(generator=generator)
    return tensor.mul_(scale)


@contextmanager
def _deadline(seconds: int) -> Iterator[None]:
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def on_timeout(_signum, _frame) -> None:
        raise TimeoutError(
            f"kernel launch exceeded {seconds}s and is treated as deadlock"
        )

    previous = signal.signal(signal.SIGALRM, on_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def launch_and_sync(name: str, call: Callable[[], T]) -> T:
    """Allow the first JIT, then apply the 30-second launch deadline."""

    first = name not in _FIRST_LAUNCHES
    start = time.monotonic()
    if first:
        result = call()
        torch.cuda.synchronize()
        _FIRST_LAUNCHES.add(name)
        print(f"{name} first compile/warm launch in {time.monotonic() - start:.1f}s")
        return result
    with _deadline(30):
        result = call()
        torch.cuda.synchronize()
    return result


def materialize_metadata(
    case: CpRankCase,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    metadata = make_torch_metadata(case, device=device)
    return (
        metadata["cu_seqlens_q"],
        metadata["cu_seqlens_kv"],
        metadata["fragment_indices"],
        make_torch_topk(case, device=device),
    )


def make_shared_fragment_metadata():
    """Build the focused two-fragment metadata shared by operator tests."""

    from msa_v1 import attention

    device = torch.device("cuda", torch.cuda.current_device())
    cu_q = torch.tensor((0, 32, 64), dtype=torch.int32, device=device)
    cu_k = torch.tensor((0, 128, 256), dtype=torch.int32, device=device)
    fragments = torch.tensor((0, 0), dtype=torch.int32, device=device)
    topk = torch.full(
        (INDEX_HEADS, 64, TOPK),
        -1,
        dtype=torch.int32,
        device=device,
    )
    topk[:, :32, 0] = 0
    topk[:, 32:, :2] = torch.tensor(
        (0, 1),
        dtype=torch.int32,
        device=device,
    )
    metadata = attention.prepare(
        topk,
        cu_q,
        cu_k,
        total_k=256,
        total_rows=3,
        max_seqlen_q=32,
        max_seqlen_k=256,
        fragment_indices=fragments,
    )

    work_count = int(metadata.schedule.work_count.item())
    active = metadata.schedule.scheduler_metadata[:work_count].cpu()
    owners = metadata.schedule.dkv_owner_counts.cpu()
    expected = torch.zeros_like(owners)
    cu_k_host = cu_k.cpu()
    fragments_host = fragments.cpu()
    for head, _, _, _, fragment_idx, kv_block_idx in active.tolist():
        physical_batch = int(fragments_host[fragment_idx])
        physical_block = (
            int(cu_k_host[physical_batch]) + physical_batch * BLOCK_SIZE
        ) // BLOCK_SIZE + kv_block_idx
        expected[head, physical_block] += 1
    torch.testing.assert_close(owners, expected, atol=0, rtol=0)
    expected_split = torch.nonzero(expected.reshape(-1) > 1).reshape(-1).to(torch.int32)
    split_count = int(metadata.schedule.dkv_split_count.item())
    assert split_count == expected_split.numel()
    torch.testing.assert_close(
        metadata.schedule.dkv_split_indices[:split_count].cpu().sort().values,
        expected_split.sort().values,
        atol=0,
        rtol=0,
    )

    kl_work_count = int(metadata.kl_schedule.work_count.item())
    kl_active = metadata.kl_schedule.scheduler_metadata[:kl_work_count].cpu()
    kl_owner_expected = torch.zeros_like(metadata.kl_schedule.dki_owner_counts.cpu())
    for record in kl_active.tolist():
        physical_block, k_offset, macro_begin, macro_count = record
        assert k_offset == physical_block * BLOCK_SIZE
        assert macro_begin >= 0
        assert 0 < macro_count <= 16
        kl_owner_expected[physical_block] += 1
    torch.testing.assert_close(
        metadata.kl_schedule.dki_owner_counts.cpu(),
        kl_owner_expected,
        atol=0,
        rtol=0,
    )
    return metadata


def shared_fragment_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the focused shared-fragment attention reference in FP32."""

    outputs = []
    lses = []
    scale = HEAD_DIM**-0.5
    for q_begin, q_end, k_end in ((0, 32, 128), (32, 64, 256)):
        q_rows = q[q_begin:q_end].float()
        k_heads = (
            k[:k_end]
            .float()
            .repeat_interleave(
                Q_HEADS // KV_HEADS,
                dim=1,
            )
        )
        v_heads = (
            v[:k_end]
            .float()
            .repeat_interleave(
                Q_HEADS // KV_HEADS,
                dim=1,
            )
        )
        scores = torch.einsum("qhd,khd->qhk", q_rows * scale, k_heads)
        visible = torch.arange(32, device=q.device) + k_end - 31
        scores.masked_fill_(
            torch.arange(k_end, device=q.device)[None, None, :]
            >= visible[:, None, None],
            -torch.inf,
        )
        probability = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("qhk,khd->qhd", probability, v_heads))
        lses.append(torch.logsumexp(scores, dim=-1))
    return torch.cat(outputs), torch.cat(lses)


def kl_shared_fragment_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    qi: torch.Tensor,
    ki: torch.Tensor,
    metadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the focused shared-fragment KL reference in FP32."""

    total_q = q.shape[0]
    scale = HEAD_DIM**-0.5
    teacher_lse = torch.empty(
        total_q,
        Q_HEADS,
        dtype=torch.float32,
        device=q.device,
    )
    indexer_lse = torch.empty(
        INDEX_HEADS,
        total_q,
        dtype=torch.float32,
        device=q.device,
    )
    dqi = torch.zeros_like(qi, dtype=torch.float32)
    dki = torch.zeros_like(ki, dtype=torch.float32)
    topk = metadata.topk_indices

    for q_begin, q_end, k_end in ((0, 32, 128), (32, 64, 256)):
        q_len = q_end - q_begin
        key_ids = torch.arange(k_end, device=q.device)
        causal_limit = k_end - q_len + torch.arange(q_len, device=q.device)
        causal_mask = key_ids[None, :] <= causal_limit[:, None]
        block_ids = key_ids // BLOCK_SIZE
        sparse_mask = torch.zeros(
            INDEX_HEADS,
            q_len,
            k_end,
            dtype=torch.bool,
            device=q.device,
        )
        for head in range(INDEX_HEADS):
            for q_local in range(q_len):
                selected = topk[head, q_begin + q_local]
                selected = selected[selected >= 0]
                sparse_mask[head, q_local] = torch.isin(block_ids, selected)
        sparse_mask &= causal_mask[None]

        k_teacher = (
            k[:k_end]
            .float()
            .repeat_interleave(
                Q_HEADS // KV_HEADS,
                dim=1,
            )
        )
        teacher_scores = (
            torch.einsum(
                "qhd,khd->hqk",
                q[q_begin:q_end].float(),
                k_teacher,
            )
            * scale
        )
        teacher_mask = sparse_mask.repeat_interleave(
            Q_HEADS // INDEX_HEADS,
            dim=0,
        )
        teacher_scores.masked_fill_(~teacher_mask, -torch.inf)
        teacher_lse_chunk = torch.logsumexp(teacher_scores, dim=-1)
        teacher_lse[q_begin:q_end] = teacher_lse_chunk.T

        student_scores = (
            torch.einsum(
                "qhd,kd->hqk",
                qi[q_begin:q_end].float(),
                ki[:k_end, 0].float(),
            )
            * scale
        )
        student_scores.masked_fill_(~sparse_mask, -torch.inf)
        student_lse_chunk = torch.logsumexp(student_scores, dim=-1)
        indexer_lse[:, q_begin:q_end] = student_lse_chunk

        teacher_probability = torch.exp(
            teacher_scores - teacher_lse_chunk[..., None]
        ).masked_fill(~teacher_mask, 0.0)
        student_probability = torch.exp(
            student_scores - student_lse_chunk[..., None]
        ).masked_fill(~sparse_mask, 0.0)
        teacher_mean = teacher_probability.view(
            INDEX_HEADS,
            Q_HEADS // INDEX_HEADS,
            q_len,
            k_end,
        ).mean(dim=1)
        ds = (student_probability - teacher_mean).masked_fill(
            ~sparse_mask,
            0.0,
        )
        ds = ds.to(torch.bfloat16).float()
        grad_scale = scale / INDEX_HEADS / total_q
        dqi[q_begin:q_end] = (
            torch.einsum(
                "hqk,kd->qhd",
                ds,
                ki[:k_end, 0].float(),
            )
            * grad_scale
        )
        dki[:k_end, 0] += (
            torch.einsum(
                "hqk,qhd->kd",
                ds,
                qi[q_begin:q_end].float(),
            )
            * grad_scale
        )

    return (
        teacher_lse,
        indexer_lse,
        dqi.to(torch.bfloat16),
        dki.to(torch.bfloat16),
    )


def _iter_attention_chunks(
    case: CpRankCase,
    topk: torch.Tensor,
    *,
    chunk_size: int = 128,
):
    for fragment_id, fragment in enumerate(case.fragments):
        fragment_begin = case.cu_seqlens_q[fragment_id]
        fragment_end = case.cu_seqlens_q[fragment_id + 1]
        k_base = case.cu_seqlens_kv[case.fragment_indices[fragment_id]]
        for q_begin in range(fragment_begin, fragment_end, chunk_size):
            q_end = min(q_begin + chunk_size, fragment_end)
            ids = topk[:, q_begin:q_end]
            valid_ids = ids[ids >= 0]
            first_block = int(valid_ids.min().item())
            last_block = int(valid_ids.max().item())
            key_begin = first_block * BLOCK_SIZE
            key_end = min(
                (last_block + 1) * BLOCK_SIZE,
                fragment.local_begin + q_end - fragment_begin,
            )
            local_visible = torch.arange(
                fragment.local_begin + q_begin - fragment_begin + 1,
                fragment.local_begin + q_end - fragment_begin + 1,
                dtype=torch.int64,
                device=topk.device,
            )
            keys = torch.arange(key_begin, key_end, device=topk.device)
            key_blocks = torch.div(keys, BLOCK_SIZE, rounding_mode="floor")
            selected = (ids[..., None] == key_blocks[None, None, None, :]).any(dim=2)
            mask = selected & (keys[None, None, :] < local_visible[None, :, None])
            yield q_begin, q_end, k_base + key_begin, k_base + key_end, mask


def attention_full_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk: torch.Tensor,
    case: CpRankCase,
    dout: torch.Tensor | None = None,
):
    """Compute MSA v1 forward and optional backward in FP32 chunks."""

    out = torch.empty_like(q, dtype=torch.float32)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    dq = torch.zeros_like(q, dtype=torch.float32) if dout is not None else None
    dk = torch.zeros_like(k, dtype=torch.float32) if dout is not None else None
    dv = torch.zeros_like(v, dtype=torch.float32) if dout is not None else None
    scale = HEAD_DIM**-0.5
    for q_begin, q_end, k_begin, k_end, mask4 in _iter_attention_chunks(case, topk):
        q_rows = q[q_begin:q_end].float()
        k_rows = k[k_begin:k_end].float()
        v_rows = v[k_begin:k_end].float()
        k_heads = k_rows.repeat_interleave(Q_HEADS // KV_HEADS, dim=1)
        v_heads = v_rows.repeat_interleave(Q_HEADS // KV_HEADS, dim=1)
        mask = mask4.repeat_interleave(Q_HEADS // INDEX_HEADS, dim=0).permute(1, 0, 2)
        scores = torch.einsum("qhd,khd->qhk", q_rows * scale, k_heads)
        scores.masked_fill_(~mask, -torch.inf)
        probability = torch.softmax(scores, dim=-1)
        out[q_begin:q_end] = torch.einsum("qhk,khd->qhd", probability, v_heads)
        lse[q_begin:q_end] = torch.logsumexp(scores, dim=-1)
        if dout is None:
            continue
        assert dq is not None and dk is not None and dv is not None
        do_rows = dout[q_begin:q_end].float()
        dp = torch.einsum("qhd,khd->qhk", do_rows, v_heads)
        ds = probability * (dp - (dp * probability).sum(dim=-1, keepdim=True))
        dq[q_begin:q_end] += torch.einsum("qhk,khd->qhd", ds, k_heads) * scale
        dk_heads = torch.einsum("qhk,qhd->khd", ds, q_rows) * scale
        dv_heads = torch.einsum("qhk,qhd->khd", probability, do_rows)
        dk[k_begin:k_end] += dk_heads.view(
            k_end - k_begin, KV_HEADS, Q_HEADS // KV_HEADS, HEAD_DIM
        ).sum(dim=2)
        dv[k_begin:k_end] += dv_heads.view(
            k_end - k_begin, KV_HEADS, Q_HEADS // KV_HEADS, HEAD_DIM
        ).sum(dim=2)
    if dout is None:
        return out, lse
    return out, lse, dq, dk, dv


def kl_full_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    qi: torch.Tensor,
    ki: torch.Tensor,
    topk: torch.Tensor,
    case: CpRankCase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the MSA v1 KL gradients in bounded FP32 chunks."""

    teacher_lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    indexer_lse = torch.empty(
        (INDEX_HEADS, q.shape[0]), dtype=torch.float32, device=q.device
    )
    dqi = torch.zeros_like(qi, dtype=torch.float32)
    dki = torch.zeros_like(ki, dtype=torch.float32)
    scale = HEAD_DIM**-0.5
    grad_scale = scale / INDEX_HEADS / q.shape[0]

    for q_begin, q_end, k_begin, k_end, mask4 in _iter_attention_chunks(case, topk):
        q_rows = q[q_begin:q_end].float()
        k_rows = k[k_begin:k_end].float()
        qi_rows = qi[q_begin:q_end].float()
        ki_rows = ki[k_begin:k_end, 0].float()
        teacher_mask = mask4.repeat_interleave(Q_HEADS // INDEX_HEADS, dim=0).permute(
            1, 0, 2
        )
        student_mask = mask4.permute(1, 0, 2)

        teacher_scores = torch.einsum(
            "qhd,khd->qhk",
            q_rows * scale,
            k_rows.repeat_interleave(Q_HEADS // KV_HEADS, dim=1),
        )
        teacher_scores.masked_fill_(~teacher_mask, -torch.inf)
        student_scores = torch.einsum("qhd,kd->qhk", qi_rows * scale, ki_rows)
        student_scores.masked_fill_(~student_mask, -torch.inf)

        teacher_lse[q_begin:q_end] = torch.logsumexp(teacher_scores, dim=-1)
        indexer_lse[:, q_begin:q_end] = torch.logsumexp(
            student_scores, dim=-1
        ).transpose(0, 1)
        teacher_probability = torch.softmax(teacher_scores, dim=-1)
        student_probability = torch.softmax(student_scores, dim=-1)
        teacher_mean = teacher_probability.view(
            q_end - q_begin,
            INDEX_HEADS,
            Q_HEADS // INDEX_HEADS,
            k_end - k_begin,
        ).mean(dim=2)
        ds = (student_probability - teacher_mean).to(torch.bfloat16).float()
        dqi[q_begin:q_end] = torch.einsum("qhk,kd->qhd", ds, ki_rows) * grad_scale
        dki[k_begin:k_end, 0] += torch.einsum("qhk,qhd->kd", ds, qi_rows) * grad_scale

    return (
        teacher_lse,
        indexer_lse.contiguous(),
        dqi.to(torch.bfloat16),
        dki.to(torch.bfloat16),
    )


def indexer_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    case: CpRankCase,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact Top15 plus local and selected LSE in bounded chunks."""

    ids_out = torch.full(
        (INDEX_HEADS, case.total_q, TOPK),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    lse_out = torch.empty(
        (INDEX_HEADS, case.total_q),
        dtype=torch.float32,
        device=q.device,
    )
    scale = HEAD_DIM**-0.5
    for fragment_id, fragment in enumerate(case.fragments):
        q_fragment_begin = case.cu_seqlens_q[fragment_id]
        q_fragment_end = case.cu_seqlens_q[fragment_id + 1]
        k_base = case.cu_seqlens_kv[case.fragment_indices[fragment_id]]
        effective_k = case.cu_seqlens_kv[fragment_id + 1] - k_base
        max_rows = max(1, 8_000_000 // (INDEX_HEADS * effective_k))
        for q_begin in range(q_fragment_begin, q_fragment_end, max_rows):
            q_end = min(q_begin + max_rows, q_fragment_end)
            local_begin = fragment.local_begin + q_begin - q_fragment_begin
            visible = torch.arange(
                local_begin + 1,
                local_begin + q_end - q_begin + 1,
                device=q.device,
            )
            scores = torch.einsum(
                "qhd,kd->hqk",
                q[q_begin:q_end].float() * scale,
                k[k_base : k_base + effective_k, 0].float(),
            )
            keys = torch.arange(effective_k, device=q.device)
            scores.masked_fill_(
                keys[None, None, :] >= visible[None, :, None], -torch.inf
            )
            blocks = (effective_k + BLOCK_SIZE - 1) // BLOCK_SIZE
            padded = torch.nn.functional.pad(
                scores,
                (0, blocks * BLOCK_SIZE - effective_k),
                value=-torch.inf,
            )
            block_scores = padded.view(
                INDEX_HEADS, q_end - q_begin, blocks, BLOCK_SIZE
            ).amax(-1)
            local = torch.div(visible - 1, BLOCK_SIZE, rounding_mode="floor")
            nonlocal_scores = block_scores.scatter(
                2,
                local.view(1, -1, 1).expand(INDEX_HEADS, -1, -1),
                -torch.inf,
            )
            slots = min(TOPK - 1, max(0, blocks - 1))
            selected = torch.full(
                (INDEX_HEADS, q_end - q_begin, TOPK),
                -1,
                dtype=torch.int32,
                device=q.device,
            )
            valid_count = torch.zeros(
                (INDEX_HEADS, q_end - q_begin), dtype=torch.int64, device=q.device
            )
            if slots:
                values, candidates = torch.topk(nonlocal_scores, slots, dim=-1)
                valid = torch.isfinite(values)
                selected[..., :slots] = torch.where(
                    valid, candidates.to(torch.int32), -1
                )
                valid_count = valid.sum(dim=-1)
            selected.scatter_(
                2,
                valid_count.unsqueeze(-1),
                local.view(1, -1, 1).expand(INDEX_HEADS, -1, -1).to(torch.int32),
            )
            ids_out[:, q_begin:q_end] = selected
            selected_blocks = (
                selected[..., None]
                == torch.arange(blocks, device=q.device)[None, None, None, :]
            ).any(dim=2)
            selected_tokens = selected_blocks.repeat_interleave(BLOCK_SIZE, dim=-1)[
                ..., :effective_k
            ]
            lse_out[:, q_begin:q_end] = torch.logsumexp(
                scores.masked_fill(~selected_tokens, -torch.inf), dim=-1
            )
    return ids_out, lse_out


def assert_indexer_full_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    case: CpRankCase,
    ids: torch.Tensor,
    *,
    use_fp16_score: bool = False,
) -> torch.Tensor:
    """Validate every selected block and return LSE for the actual selection."""

    lse_out = torch.empty(
        (INDEX_HEADS, case.total_q), dtype=torch.float32, device=q.device
    )
    scale = HEAD_DIM**-0.5
    slots = torch.arange(TOPK, device=q.device)
    for fragment_id, fragment in enumerate(case.fragments):
        q_fragment_begin = case.cu_seqlens_q[fragment_id]
        q_fragment_end = case.cu_seqlens_q[fragment_id + 1]
        k_base = case.cu_seqlens_kv[case.fragment_indices[fragment_id]]
        effective_k = case.cu_seqlens_kv[fragment_id + 1] - k_base
        max_rows = max(1, 8_000_000 // (INDEX_HEADS * effective_k))
        for q_begin in range(q_fragment_begin, q_fragment_end, max_rows):
            q_end = min(q_begin + max_rows, q_fragment_end)
            local_begin = fragment.local_begin + q_begin - q_fragment_begin
            visible = torch.arange(
                local_begin + 1,
                local_begin + q_end - q_begin + 1,
                device=q.device,
            )
            scores = torch.einsum(
                "qhd,kd->hqk",
                q[q_begin:q_end].float() * scale,
                k[k_base : k_base + effective_k, 0].float(),
            )
            keys = torch.arange(effective_k, device=q.device)
            scores.masked_fill_(
                keys[None, None, :] >= visible[None, :, None], -torch.inf
            )
            blocks = (effective_k + BLOCK_SIZE - 1) // BLOCK_SIZE
            padded = torch.nn.functional.pad(
                scores,
                (0, blocks * BLOCK_SIZE - effective_k),
                value=-torch.inf,
            )
            block_scores = padded.view(
                INDEX_HEADS, q_end - q_begin, blocks, BLOCK_SIZE
            ).amax(-1)
            stored_block_scores = (
                block_scores.to(torch.float16).float()
                if use_fp16_score
                else block_scores
            )
            selected = ids[:, q_begin:q_end].to(torch.int64)
            visible_blocks = torch.div(
                visible + BLOCK_SIZE - 1, BLOCK_SIZE, rounding_mode="floor"
            )
            expected_count = visible_blocks.clamp(max=TOPK)
            valid = slots[None, None, :] < expected_count[None, :, None]
            assert bool(((selected >= 0) | ~valid).all().item())
            assert bool(((selected == -1) | valid).all().item())
            assert bool(
                ((selected < visible_blocks[None, :, None]) | ~valid).all().item()
            )
            ordered = torch.sort(
                torch.where(valid, selected, visible_blocks[None, :, None]), dim=-1
            ).values
            assert not bool(
                ((ordered[..., 1:] == ordered[..., :-1]) & valid[..., 1:]).any().item()
            )
            local = visible_blocks - 1
            actual_local = selected.gather(
                2,
                (expected_count - 1).view(1, -1, 1).expand(INDEX_HEADS, -1, -1),
            ).squeeze(-1)
            torch.testing.assert_close(
                actual_local,
                local.view(1, -1).expand(INDEX_HEADS, -1),
                atol=0,
                rtol=0,
            )
            large = visible_blocks > TOPK
            if bool(large.any().item()):
                eligible = stored_block_scores.clone()
                eligible.scatter_(
                    2,
                    local.view(1, -1, 1).expand(INDEX_HEADS, -1, -1),
                    -torch.inf,
                )
                threshold = torch.topk(eligible, TOPK - 1, dim=-1).values[..., -1]
                safe = selected.clamp(min=0)
                selected_scores = torch.gather(
                    stored_block_scores, 2, safe
                ).masked_fill(
                    ~valid | (selected == local.view(1, -1, 1)), torch.inf
                )
                selected_min = selected_scores.amin(dim=-1)
                assert bool(
                    ((selected_min >= threshold - 5e-3) | ~large[None]).all().item()
                )
            safe = selected.clamp(min=0)
            block_tokens = padded.view(
                INDEX_HEADS, q_end - q_begin, blocks, BLOCK_SIZE
            )
            block_sums = torch.exp(
                block_tokens - block_scores.unsqueeze(-1)
            ).masked_fill(~torch.isfinite(block_tokens), 0.0).sum(dim=-1)
            selected_scores = torch.gather(stored_block_scores, 2, safe)
            selected_sums = torch.gather(block_sums, 2, safe)
            selected_terms = selected_scores + torch.log(selected_sums)
            lse_out[:, q_begin:q_end] = torch.logsumexp(
                selected_terms.masked_fill(~valid, -torch.inf), dim=-1
            )
    return lse_out


def make_causal_topk(
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Build sorted causal block selections with the local block present."""

    total_q = sum(q_lens)
    topk = torch.full(
        (INDEX_HEADS, total_q, TOPK),
        -1,
        dtype=torch.int32,
        device=device,
    )
    q_offset = 0
    for q_len, k_len in zip(q_lens, k_lens):
        suffix_offset = k_len - q_len
        for q_idx in range(q_len):
            local_block = (suffix_offset + q_idx) // BLOCK_SIZE
            selected = list(range(min(local_block, TOPK - 1))) + [local_block]
            values = torch.tensor(selected, dtype=torch.int32, device=device)
            topk[:, q_offset + q_idx, : len(selected)] = values
        q_offset += q_len
    return topk


def make_packed_attention_metadata(
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
):
    """Build causal TopK and prepared metadata for focused packed tests."""

    from msa_v1 import attention

    topk = make_causal_topk(q_lens, k_lens, device="cuda")
    cu_q = torch.tensor(
        (0, *torch.tensor(q_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    cu_k = torch.tensor(
        (0, *torch.tensor(k_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    metadata = attention.prepare(
        topk,
        cu_q,
        cu_k,
        total_k=sum(k_lens),
        total_rows=total_rows(k_lens),
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
    )
    return topk, metadata


def make_single_query_varlen_metadata(batch: int):
    """Build deterministic one-query documents with distinct K lengths."""

    from msa_v1 import attention

    k_lens = (torch.arange(batch, dtype=torch.int32, device="cuda") * 37).remainder(
        BLOCK_SIZE
    ) + 1
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    cu_k = torch.nn.functional.pad(
        k_lens.cumsum(0, dtype=torch.int32),
        (1, 0),
    )
    topk = torch.full(
        (INDEX_HEADS, batch, TOPK),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    topk[:, :, 0] = 0
    metadata = attention.prepare(
        topk,
        cu_q,
        cu_k,
        total_k=int(k_lens.sum()),
        total_rows=batch,
        max_seqlen_q=1,
        max_seqlen_k=BLOCK_SIZE,
    )
    return k_lens, cu_k, metadata


def pad_packed_kv(
    tensor: torch.Tensor,
    k_lens: torch.Tensor,
    cu_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a padded view of packed short KV sequences."""

    local_k = torch.arange(BLOCK_SIZE, device=tensor.device)
    valid = local_k.unsqueeze(0) < k_lens.unsqueeze(1)
    packed_idx = cu_k[:-1].unsqueeze(1) + local_k.unsqueeze(0)
    safe_idx = torch.where(valid, packed_idx, torch.zeros_like(packed_idx))
    return tensor[safe_idx], valid


def total_rows(k_lens: tuple[int, ...]) -> int:
    return sum((length + BLOCK_SIZE - 1) // BLOCK_SIZE for length in k_lens)


def csr_reference(
    q2k: torch.Tensor,
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the level-major K-to-Q CSR reference on the host."""

    q2k_cpu = q2k.cpu()
    rows = []
    max_blocks = max((length + BLOCK_SIZE - 1) // BLOCK_SIZE for length in k_lens)
    for level in range(max_blocks):
        for batch_idx, k_len in enumerate(k_lens):
            if level < (k_len + BLOCK_SIZE - 1) // BLOCK_SIZE:
                rows.append((batch_idx, level))

    row_ptr = torch.zeros((INDEX_HEADS, len(rows) + 1), dtype=torch.int32)
    q_indices = torch.full(
        (INDEX_HEADS, q2k.shape[1] * TOPK),
        -1,
        dtype=torch.int32,
    )
    q_offsets = [0]
    for length in q_lens:
        q_offsets.append(q_offsets[-1] + length)
    for head in range(INDEX_HEADS):
        cursor = 0
        for row_idx, (batch_idx, block_idx) in enumerate(rows):
            values = []
            for q_local in range(q_lens[batch_idx]):
                q_global = q_offsets[batch_idx] + q_local
                if (q2k_cpu[head, q_global] == block_idx).any():
                    values.append(q_local)
            if values:
                q_indices[head, cursor : cursor + len(values)] = torch.tensor(
                    values,
                    dtype=torch.int32,
                )
            cursor += len(values)
            row_ptr[head, row_idx + 1] = cursor
    return row_ptr.to(q2k.device), q_indices.to(q2k.device)


def packed_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k: torch.Tensor,
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
    *,
    probability_qat: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute FP32 causal sparse attention for packed sequence lengths."""

    outputs = []
    lses = []
    q_offset = 0
    k_offset = 0
    scale = HEAD_DIM**-0.5
    for q_len, k_len in zip(q_lens, k_lens):
        q_doc = q[q_offset : q_offset + q_len].float().unsqueeze(0)
        k_doc = k[k_offset : k_offset + k_len].float().unsqueeze(0)
        v_doc = v[k_offset : k_offset + k_len].float().unsqueeze(0)
        q2k_doc = q2k[:, q_offset : q_offset + q_len]
        num_blocks = (k_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        block_mask = torch.zeros(
            (1, INDEX_HEADS, q_len, num_blocks),
            dtype=torch.bool,
            device=q.device,
        )
        valid = q2k_doc >= 0
        safe = torch.where(valid, q2k_doc, torch.zeros_like(q2k_doc)).long()
        head_idx = (
            torch.arange(INDEX_HEADS, device=q.device)
            .view(INDEX_HEADS, 1, 1)
            .expand_as(safe)
        )
        q_idx = torch.arange(q_len, device=q.device).view(1, q_len, 1).expand_as(safe)
        block_mask[0, head_idx[valid], q_idx[valid], safe[valid]] = True
        token_mask = block_mask.repeat_interleave(BLOCK_SIZE, dim=-1)[..., :k_len]
        token_mask = token_mask.repeat_interleave(Q_HEADS // INDEX_HEADS, dim=1)

        q_pos = torch.arange(q_len, device=q.device)
        k_pos = torch.arange(k_len, device=q.device)
        causal_limit = q_pos + (k_len - q_len)
        token_mask &= k_pos.view(1, 1, 1, -1) <= causal_limit.view(1, 1, -1, 1)

        k_heads = k_doc.repeat_interleave(Q_HEADS // KV_HEADS, dim=2)
        v_heads = v_doc.repeat_interleave(Q_HEADS // KV_HEADS, dim=2)
        scores = torch.einsum("bshd,bthd->bhst", q_doc * scale, k_heads)
        scores = scores.masked_fill(~token_mask, -torch.inf)
        lse = torch.logsumexp(scores, dim=-1).transpose(1, 2).contiguous()

        if probability_qat:
            padded_tokens = num_blocks * BLOCK_SIZE
            if padded_tokens != k_len:
                pad = padded_tokens - k_len
                scores_block = torch.nn.functional.pad(
                    scores,
                    (0, pad),
                    value=-torch.inf,
                )
                mask_block = torch.nn.functional.pad(
                    token_mask,
                    (0, pad),
                    value=False,
                )
            else:
                scores_block = scores
                mask_block = token_mask
            scores_block = scores_block.view(
                1,
                Q_HEADS,
                q_len,
                num_blocks,
                BLOCK_SIZE,
            )
            mask_block = mask_block.view(
                1,
                Q_HEADS,
                q_len,
                num_blocks,
                BLOCK_SIZE,
            )
            block_has_value = mask_block.any(dim=-1)
            block_max = scores_block.max(dim=-1).values
            safe_max = torch.where(
                block_has_value,
                block_max,
                torch.zeros_like(block_max),
            )
            p_unnorm = torch.exp(scores_block - safe_max.unsqueeze(-1))
            p_unnorm = torch.where(
                mask_block,
                p_unnorm,
                torch.zeros_like(p_unnorm),
            )
            probability_scale = 448.0
            p_rtn = (
                (p_unnorm * probability_scale).to(torch.float8_e4m3fn).float()
                / probability_scale
            )
            block_scale = torch.where(
                block_has_value,
                torch.exp(safe_max - lse.transpose(1, 2).unsqueeze(-1)),
                torch.zeros_like(safe_max),
            )
            p_smooth = p_unnorm * block_scale.unsqueeze(-1)
            p_quant = p_rtn * block_scale.unsqueeze(-1)
            probability = p_smooth + (p_quant - p_smooth).detach()
            probability = probability.view(
                1,
                Q_HEADS,
                q_len,
                padded_tokens,
            )[..., :k_len]
        else:
            probability = torch.softmax(scores, dim=-1)
            probability = torch.where(
                token_mask,
                probability,
                torch.zeros_like(probability),
            )
        out = torch.einsum("bhst,bthd->bshd", probability, v_heads)
        outputs.append(out.squeeze(0))
        lses.append(lse.squeeze(0))
        q_offset += q_len
        k_offset += k_len
    return torch.cat(outputs, dim=0), torch.cat(lses, dim=0)


def causal_indexer_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    lse_temperature: float = 1.0,
    use_fp16_score: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Top15-by-block-max plus mandatory-local indexer results."""

    q_len = q.shape[0]
    k_len = k.shape[0]
    q_offset = k_len - q_len
    scores = torch.einsum(
        "qhd,kd->hqk",
        q.float(),
        k[:, 0].float(),
    ) / math.sqrt(HEAD_DIM)
    expected_ids = torch.full(
        (INDEX_HEADS, q_len, TOPK),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    expected_lse = torch.empty(
        (INDEX_HEADS, q_len),
        dtype=torch.float32,
        device=q.device,
    )
    for q_idx in range(q_len):
        visible_end = q_offset + q_idx + 1
        local_block = (visible_end - 1) // BLOCK_SIZE
        for head in range(INDEX_HEADS):
            candidates = []
            for block in range((visible_end + BLOCK_SIZE - 1) // BLOCK_SIZE):
                begin = block * BLOCK_SIZE
                end = min(begin + BLOCK_SIZE, visible_end)
                block_values = scores[head, q_idx, begin:end] / lse_temperature
                block_max = block_values.max()
                block_sum = torch.exp(block_values - block_max).sum()
                stored_max = (
                    block_max.to(torch.float16).float()
                    if use_fp16_score
                    else block_max
                )
                candidates.append((stored_max, block, block_sum))
            nonlocal_candidates = [
                item for item in candidates if item[1] != local_block
            ]
            nonlocal_candidates.sort(key=lambda item: (-float(item[0]), item[1]))
            selected_candidates = nonlocal_candidates[: TOPK - 1] + [
                item for item in candidates if item[1] == local_block
            ]
            selected = [block for _, block, _ in selected_candidates]
            expected_ids[head, q_idx, : len(selected)] = torch.tensor(
                selected,
                dtype=torch.int32,
                device=q.device,
            )
            selected_terms = torch.stack(
                [score + torch.log(block_sum) for score, _, block_sum in selected_candidates]
            )
            expected_lse[head, q_idx] = torch.logsumexp(selected_terms, dim=0)
    return expected_ids, expected_lse


def _causal_indexer_reference_vectorized(
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the indexer reference without host-side score inspection."""

    q_len = q.shape[0]
    k_len = k.shape[0]
    q_offset = k_len - q_len
    scores = torch.einsum(
        "qhd,kd->hqk",
        q.float(),
        k[:, 0].float(),
    ) / math.sqrt(HEAD_DIM)
    visible_end = q_offset + torch.arange(q_len, device=q.device) + 1
    token_idx = torch.arange(k_len, device=q.device)
    causal = token_idx.unsqueeze(0) < visible_end.unsqueeze(1)
    num_blocks = (k_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded_scores = torch.nn.functional.pad(
        scores.masked_fill(~causal.unsqueeze(0), -torch.inf),
        (0, num_blocks * BLOCK_SIZE - k_len),
        value=-torch.inf,
    )
    block_scores = padded_scores.view(
        INDEX_HEADS,
        q_len,
        num_blocks,
        BLOCK_SIZE,
    ).amax(dim=-1)
    local_block = (visible_end - 1) // BLOCK_SIZE
    nonlocal_scores = block_scores.scatter(
        2,
        local_block.view(1, q_len, 1).expand(INDEX_HEADS, -1, -1),
        -torch.inf,
    )
    expected_ids = torch.full(
        (INDEX_HEADS, q_len, TOPK),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    selected_blocks = (
        torch.nn.functional.one_hot(local_block, num_classes=num_blocks)
        .to(torch.bool)
        .unsqueeze(0)
        .expand(INDEX_HEADS, -1, -1)
    )
    top_slots = min(TOPK - 1, num_blocks - 1)
    if top_slots > 0:
        candidate_scores, candidate_ids = torch.topk(
            nonlocal_scores,
            k=top_slots,
            dim=-1,
            sorted=True,
        )
        candidate_valid = torch.isfinite(candidate_scores)
        expected_ids[:, :, :top_slots] = torch.where(
            candidate_valid,
            candidate_ids.to(torch.int32),
            -1,
        )
        local_slot = candidate_valid.sum(dim=-1, dtype=torch.int64)
        expected_ids.scatter_(
            2,
            local_slot.unsqueeze(-1),
            local_block.view(1, q_len, 1).expand(INDEX_HEADS, -1, -1).to(torch.int32),
        )
        candidate_blocks = torch.nn.functional.one_hot(
            candidate_ids,
            num_classes=num_blocks,
        ).to(torch.bool)
        selected_blocks = selected_blocks | (
            candidate_blocks & candidate_valid.unsqueeze(-1)
        ).any(dim=2)
    selected_tokens = selected_blocks.repeat_interleave(
        BLOCK_SIZE,
        dim=-1,
    )[..., :k_len]
    selected_tokens &= causal.unsqueeze(0)
    expected_lse = torch.logsumexp(
        scores.masked_fill(~selected_tokens, -torch.inf),
        dim=-1,
    )
    return expected_ids, expected_lse


causal_indexer_reference_compiled = torch.compile(
    _causal_indexer_reference_vectorized,
    mode="max-autotune",
)


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            left.float().flatten(), right.float().flatten(), dim=0
        ).item()
    )
