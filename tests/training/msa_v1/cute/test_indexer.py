"""Real-case correctness suite for the packed-varlen MSA v1 indexer."""

from __future__ import annotations

import warnings

import pytest
import torch

from msa_v1 import attention, indexer
from tests.training.msa_v1.cute.cases import MsaTestCase
from tests.training.msa_v1.cute.testing import (
    HEAD_DIM,
    INDEX_HEADS,
    assert_indexer_full_reference,
    case_seed,
    causal_indexer_reference,
    causal_indexer_reference_compiled,
    cuda_generator,
    launch_and_sync,
    materialize_metadata,
    random_bf16,
)

pytestmark = [pytest.mark.gpu, pytest.mark.acceptance]


@torch.inference_mode()
def test_indexer_real_cases(msa_case: MsaTestCase, cuda_device: torch.device) -> None:
    case = msa_case.rank_case
    generator = cuda_generator(cuda_device, case_seed(msa_case, 47))
    cu_q, cu_kv, fragments, _ = materialize_metadata(case, cuda_device)
    if msa_case.structured:
        q = torch.zeros(
            case.total_q,
            INDEX_HEADS,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=cuda_device,
        )
        k = torch.zeros(
            case.total_kv, 1, HEAD_DIM, dtype=torch.bfloat16, device=cuda_device
        )
    else:
        q = random_bf16(
            (case.total_q, INDEX_HEADS, HEAD_DIM),
            device=cuda_device,
            generator=generator,
        )
        k = random_bf16(
            (case.total_kv, 1, HEAD_DIM),
            device=cuda_device,
            generator=generator,
        )
    schedule = launch_and_sync(
        "indexer-prepare",
        lambda: indexer.prepare_indexer_schedule(
            cu_q,
            cu_kv,
            total_q=case.total_q,
            fragment_indices=fragments,
        ),
    )
    ids, lse = launch_and_sync(
        "indexer-forward-deterministic-0",
        lambda: indexer.forward(
            q,
            k,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=case.max_seqlen_q,
            max_seqlen_kv=case.max_seqlen_kv,
            fragment_indices=fragments,
            schedule=schedule,
            deterministic=True,
        ),
    )
    ids = ids.clone()
    lse = lse.clone()
    ids_repeat, lse_repeat = launch_and_sync(
        "indexer-forward-deterministic-1",
        lambda: indexer.forward(
            q,
            k,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=case.max_seqlen_q,
            max_seqlen_kv=case.max_seqlen_kv,
            fragment_indices=fragments,
            schedule=schedule,
            deterministic=True,
        ),
    )
    assert torch.equal(ids_repeat, ids)
    assert torch.equal(lse_repeat.view(torch.int32), lse.view(torch.int32))
    lse_ref = assert_indexer_full_reference(q, k, case, ids)
    torch.testing.assert_close(lse, lse_ref, atol=2e-3, rtol=2e-3)
    valid_count = (ids >= 0).sum(dim=-1)
    local = torch.div(
        torch.cat(
            [
                torch.arange(
                    fragment.local_begin,
                    fragment.local_end,
                    device=cuda_device,
                )
                for fragment in case.fragments
            ]
        ),
        128,
        rounding_mode="floor",
    )
    actual_local = ids.gather(
        2, (valid_count - 1).clamp(min=0).unsqueeze(-1).to(torch.int64)
    ).squeeze(-1)
    torch.testing.assert_close(
        actual_local,
        local.view(1, -1).expand(INDEX_HEADS, -1).to(torch.int32),
        atol=0,
        rtol=0,
    )


def _make_cu_seqlens(lengths: tuple[int, ...]) -> torch.Tensor:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, dtype=torch.int32, device="cuda")


def _packed_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
    fragment_indices: tuple[int, ...] | None = None,
    *,
    lse_temperature: float = 1.0,
    use_fp16_score: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_offsets = [0]
    k_offsets = [0]
    for q_len in q_lens:
        q_offsets.append(q_offsets[-1] + q_len)
    for k_len in k_lens:
        k_offsets.append(k_offsets[-1] + k_len)

    ids = []
    lse = []
    for batch_idx in range(len(q_lens)):
        k_start_idx = (
            batch_idx if fragment_indices is None else fragment_indices[batch_idx]
        )
        fragment_ids, fragment_lse = causal_indexer_reference(
            q[q_offsets[batch_idx] : q_offsets[batch_idx + 1]],
            k[k_offsets[k_start_idx] : k_offsets[batch_idx + 1]],
            lse_temperature=lse_temperature,
            use_fp16_score=use_fp16_score,
        )
        ids.append(fragment_ids)
        lse.append(fragment_lse)
    return torch.cat(ids, dim=1), torch.cat(lse, dim=1)


def _assert_indexer_outputs(
    ids: torch.Tensor,
    selected_lse: torch.Tensor,
    ids_ref: torch.Tensor,
    lse_ref: torch.Tensor,
    q_lens: tuple[int, ...],
    effective_k_lens: tuple[int, ...],
) -> None:
    torch.cuda.synchronize()
    total_q = sum(q_lens)
    assert ids.shape == (INDEX_HEADS, total_q, 16)
    assert ids.dtype == torch.int32 and ids.is_cuda
    assert selected_lse.shape == (INDEX_HEADS, total_q)
    assert selected_lse.dtype == torch.float32 and selected_lse.is_cuda
    assert torch.all((ids >= 0) | (ids == -1))
    q_offset = 0
    for q_len, k_len in zip(q_lens, effective_k_lens):
        for q_local in range(q_len):
            q_global = q_offset + q_local
            local_block = (k_len - q_len + q_local) // 128
            for head in range(INDEX_HEADS):
                valid = ids[head, q_global][ids[head, q_global] >= 0]
                valid_ref = ids_ref[head, q_global][ids_ref[head, q_global] >= 0]
                assert set(valid.cpu().tolist()) == set(valid_ref.cpu().tolist())
                assert len(valid) == len(set(valid.cpu().tolist()))
                assert int(valid[-1]) == local_block
        q_offset += q_len
    torch.testing.assert_close(selected_lse, lse_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("use_fp16_score", (False, True))
def test_single_fragment_indexer_and_attention_integration(
    use_fp16_score: bool,
) -> None:
    torch.manual_seed(23)
    q_lens = (65,)
    k_lens = (2304,)
    q_len = sum(q_lens)
    k_len = sum(k_lens)
    cu_q = _make_cu_seqlens(q_lens)
    cu_k = _make_cu_seqlens(k_lens)
    q_index = torch.randn(
        q_len,
        INDEX_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_index = torch.randn(
        k_len,
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    ids, selected_lse = indexer.forward(
        q_index,
        k_index,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(k_lens),
        use_fp16_score=use_fp16_score,
    )
    ids_ref, lse_ref = _packed_reference(
        q_index,
        k_index,
        q_lens,
        k_lens,
        use_fp16_score=use_fp16_score,
    )
    if not use_fp16_score:
        compiled_ids_ref, compiled_lse_ref = causal_indexer_reference_compiled(
            q_index,
            k_index,
        )
        if not torch.equal(
            torch.sort(compiled_ids_ref, dim=-1).values,
            torch.sort(ids_ref, dim=-1).values,
        ):
            warnings.warn("compiled indexer reference ids differ from eager reference")
        if not torch.allclose(compiled_lse_ref, lse_ref, atol=2e-3, rtol=2e-3):
            warnings.warn("compiled indexer reference LSE differs from eager reference")
    _assert_indexer_outputs(
        ids,
        selected_lse,
        ids_ref,
        lse_ref,
        q_lens,
        k_lens,
    )

    metadata = attention.prepare(
        ids,
        cu_q,
        cu_k,
        total_k=k_len,
        total_rows=(k_len + 127) // 128,
        max_seqlen_q=q_len,
        max_seqlen_k=k_len,
    )
    q = torch.randn(
        q_len,
        64,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    k = torch.randn(
        k_len,
        4,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    v = torch.randn_like(k, requires_grad=True)
    out = attention.forward(q, k, v, metadata)
    grads = torch.autograd.grad(out, (q, k, v), torch.randn_like(out))
    assert all(torch.isfinite(tensor).all() for tensor in (out, *grads))


@pytest.mark.parametrize("use_fp16_score", (False, True))
def test_packed_varlen_indexer_reusable_schedule_and_workspace(
    use_fp16_score: bool,
) -> None:
    torch.manual_seed(29)
    q_lens = (33, 65, 70) + tuple(1 + index % 3 for index in range(30))
    k_lens = (257, 320, 513) + tuple(
        q_len + 1 + index % 4 for index, q_len in enumerate(q_lens[3:])
    )
    total_q = sum(q_lens)
    cu_q = _make_cu_seqlens(q_lens)
    cu_k = _make_cu_seqlens(k_lens)
    q = torch.randn(
        total_q,
        INDEX_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.randn(
        sum(k_lens),
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    schedule = indexer.prepare_indexer_schedule(cu_q, cu_k, total_q=total_q)
    workspace = indexer.allocate_indexer_workspace(
        total_q=total_q,
        batch=len(q_lens),
        max_seqlen_kv=max(k_lens),
        device=q.device,
        use_fp16_score=use_fp16_score,
    )
    ids, selected_lse = indexer.forward(
        q,
        k,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(k_lens),
        schedule=schedule,
        workspace=workspace,
        use_fp16_score=use_fp16_score,
    )
    ids_ref, lse_ref = _packed_reference(
        q,
        k,
        q_lens,
        k_lens,
        use_fp16_score=use_fp16_score,
    )
    _assert_indexer_outputs(
        ids,
        selected_lse,
        ids_ref,
        lse_ref,
        q_lens,
        k_lens,
    )
    assert workspace.schedule is schedule
    assert workspace.score.dtype == (
        torch.float16 if use_fp16_score else torch.float32
    )
    assert workspace.block_sum.dtype == torch.float32


@pytest.mark.parametrize("deterministic", (False, True))
def test_fp16_score_ties_use_block_id_order(deterministic: bool) -> None:
    q_lens = (1,)
    k_lens = (2304,)
    cu_q = _make_cu_seqlens(q_lens)
    cu_k = _make_cu_seqlens(k_lens)
    q = torch.zeros(1, INDEX_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(2304, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")

    ids, selected_lse = indexer.forward(
        q,
        k,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_k,
        max_seqlen_q=1,
        max_seqlen_kv=2304,
        deterministic=deterministic,
        use_fp16_score=True,
    )
    ids_ref, lse_ref = _packed_reference(
        q,
        k,
        q_lens,
        k_lens,
        use_fp16_score=True,
    )
    _assert_indexer_outputs(ids, selected_lse, ids_ref, lse_ref, q_lens, k_lens)
    expected = torch.tensor(
        [*range(15), 17], dtype=torch.int32, device="cuda"
    )
    if deterministic:
        torch.testing.assert_close(ids[:, 0], expected.expand(INDEX_HEADS, -1))


def test_packed_varlen_indexer_fragment_indices() -> None:
    torch.manual_seed(31)
    q_lens = (17, 41, 73)
    k_lens = (149, 181, 259)
    fragment_indices_host = (0, 0, 1)
    effective_k_lens = (149, 330, 440)
    total_q = sum(q_lens)
    cu_q = _make_cu_seqlens(q_lens)
    cu_k = _make_cu_seqlens(k_lens)
    fragment_indices = torch.tensor(
        fragment_indices_host,
        dtype=torch.int32,
        device="cuda",
    )
    q = torch.randn(
        total_q,
        INDEX_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.randn(
        sum(k_lens),
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    schedule = indexer.prepare_indexer_schedule(
        cu_q,
        cu_k,
        total_q=total_q,
        fragment_indices=fragment_indices,
    )
    ids, selected_lse = indexer.forward(
        q,
        k,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(effective_k_lens),
        fragment_indices=fragment_indices,
        schedule=schedule,
    )
    ids_ref, lse_ref = _packed_reference(
        q,
        k,
        q_lens,
        k_lens,
        fragment_indices_host,
    )
    _assert_indexer_outputs(
        ids,
        selected_lse,
        ids_ref,
        lse_ref,
        q_lens,
        effective_k_lens,
    )


@pytest.mark.parametrize("lse_temperature", (0.5, 2.0))
@pytest.mark.parametrize(
    "q_lens,k_lens,fragment_indices_host",
    (
        ((33,), (512,), None),
        ((17, 19), (1152, 1152), (0, 0)),
    ),
)
def test_packed_varlen_indexer_lse_temperature(
    lse_temperature: float,
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
    fragment_indices_host: tuple[int, ...] | None,
) -> None:
    torch.manual_seed(37)
    total_q = sum(q_lens)
    cu_q = _make_cu_seqlens(q_lens)
    cu_k = _make_cu_seqlens(k_lens)
    fragment_indices = (
        None
        if fragment_indices_host is None
        else torch.tensor(
            fragment_indices_host,
            dtype=torch.int32,
            device="cuda",
        )
    )
    effective_k_lens = tuple(
        sum(k_lens[start : batch_idx + 1])
        for batch_idx, start in enumerate(
            range(len(k_lens))
            if fragment_indices_host is None
            else fragment_indices_host
        )
    )
    q = torch.randn(
        total_q,
        INDEX_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.randn(
        sum(k_lens),
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    kwargs = {
        "cu_seqlens_q": cu_q,
        "cu_seqlens_kv": cu_k,
        "max_seqlen_q": max(q_lens),
        "max_seqlen_kv": max(effective_k_lens),
        "fragment_indices": fragment_indices,
    }
    ids_default, lse_default = indexer.forward(q, k, **kwargs)
    ids, selected_lse = indexer.forward(
        q,
        k,
        **kwargs,
        lse_temperature=lse_temperature,
    )
    ids_ref, lse_ref = _packed_reference(
        q,
        k,
        q_lens,
        k_lens,
        fragment_indices_host,
        lse_temperature=lse_temperature,
    )
    assert torch.equal(ids_default, ids)
    assert not torch.equal(lse_default, selected_lse)
    _assert_indexer_outputs(
        ids,
        selected_lse,
        ids_ref,
        lse_ref,
        q_lens,
        effective_k_lens,
    )


def test_packed_varlen_indexer_unit_and_invalid_temperature() -> None:
    q_lens = (17,)
    k_lens = (257,)
    cu_q = _make_cu_seqlens(q_lens)
    cu_k = _make_cu_seqlens(k_lens)
    q = torch.randn(
        17,
        INDEX_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.randn(
        257,
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    kwargs = {
        "cu_seqlens_q": cu_q,
        "cu_seqlens_kv": cu_k,
        "max_seqlen_q": 17,
        "max_seqlen_kv": 257,
    }
    ids_default, lse_default = indexer.forward(q, k, **kwargs)
    ids_one, lse_one = indexer.forward(
        q,
        k,
        **kwargs,
        lse_temperature=1.0,
    )
    assert torch.equal(ids_default, ids_one)
    assert torch.equal(lse_default, lse_one)
    for value in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="lse_temperature"):
            indexer.forward(q, k, **kwargs, lse_temperature=value)
    for value in (True, torch.tensor(1.0, device="cuda")):
        with pytest.raises(TypeError, match="lse_temperature"):
            indexer.forward(q, k, **kwargs, lse_temperature=value)
    for value in (1, torch.tensor(True, device="cuda")):
        with pytest.raises(TypeError, match="use_fp16_score"):
            indexer.forward(q, k, **kwargs, use_fp16_score=value)
    with pytest.raises(TypeError, match="use_fp16_score"):
        indexer.allocate_indexer_workspace(
            total_q=17,
            batch=1,
            max_seqlen_kv=257,
            device=q.device,
            use_fp16_score=1,
        )
    fp32_workspace = indexer.allocate_indexer_workspace(
        total_q=17,
        batch=1,
        max_seqlen_kv=257,
        device=q.device,
    )
    with pytest.raises(ValueError, match="torch.float16"):
        indexer.forward(
            q,
            k,
            **kwargs,
            workspace=fp32_workspace,
            use_fp16_score=True,
        )
