"""MSA v1 attention prepare and scheduler tests."""

import dataclasses
import gc
import inspect
import math

import pytest
import torch
from msa_v1 import attention, kl
from msa_v1.attention.prepare_k2q_csr import SparseK2qCsrBuilderSm100
from msa_v1.attention.prepare_scheduler import (
    KL_VALID_ROWS_MASK,
    KL_WRITER_RANK_MASK,
    KL_WRITER_RANK_SHIFT,
    SPARSE_SCHEDULE_MODEL,
    SparseAttentionSchedule,
)

from tests.training.msa_v1.cute.testing import (
    HEAD_DIM,
    INDEX_HEADS,
    KV_HEADS,
    Q_HEADS,
    cosine_similarity,
    csr_reference,
    make_causal_topk,
    make_packed_attention_metadata,
    make_single_query_varlen_metadata,
    packed_attention_reference,
    pad_packed_kv,
    total_rows,
)


def test_csr_builder_requires_exact_total_rows() -> None:
    parameter = inspect.signature(SparseK2qCsrBuilderSm100.__call__).parameters[
        "total_rows"
    ]

    assert parameter.default is inspect.Parameter.empty


def test_csr_builder_is_stateless() -> None:
    builder = SparseK2qCsrBuilderSm100()
    assert vars(builder) == {}


def test_schedule_api_exposes_only_production_kl_ownership() -> None:
    builder_parameters = inspect.signature(SparseK2qCsrBuilderSm100.__call__).parameters
    schedule_fields = {
        field.name for field in dataclasses.fields(SparseAttentionSchedule)
    }

    assert "return_schedule" not in builder_parameters
    assert {
        "physical_row_ptr",
        "physical_q_indices",
        "physical_valid_rows",
        "dki_owner_counts",
        "dki_split_indices",
        "dki_split_count",
    } <= (schedule_fields)
    assert (
        not {
            "dki_single_scheduler_metadata",
            "dki_single_work_count",
            "dki_reference_counts",
            "dki_logical_row_counts",
        }
        & schedule_fields
    )


def test_attention_schedule_1024_capacity_contracts() -> None:
    for case_idx in range(1024):
        work_count = case_idx % 257
        split_work_count = min(work_count, (case_idx * 13) % 97)
        capacity = work_count + 1 + (case_idx * 17) % 263
        scheduler_metadata = torch.arange(capacity * 6, dtype=torch.int32).reshape(
            capacity, 6
        )
        split_indices = torch.arange(capacity, dtype=torch.int32)
        schedule = SparseAttentionSchedule(
            enabled=True,
            scheduler_metadata=scheduler_metadata,
            work_count=torch.tensor((work_count,), dtype=torch.int32),
            dkv_split_indices=split_indices,
            dkv_split_count=torch.tensor((split_work_count,), dtype=torch.int32),
            dki_split_indices=split_indices,
            dki_split_count=torch.tensor((split_work_count,), dtype=torch.int32),
            target_q_per_cta=1728,
        )

        assert schedule.work_capacity == capacity
        assert schedule.dkv_split_capacity == capacity
        assert schedule.dki_split_capacity == capacity
        assert int(schedule.work_count[0]) == work_count
        assert int(schedule.dkv_split_count[0]) == split_work_count
        assert int(schedule.dki_split_count[0]) == split_work_count

        total_q = case_idx
        padded_blocks = 1 + (case_idx * 19) % 257
        refs = total_q * 16 * INDEX_HEADS
        physical_counts = [0] * (INDEX_HEADS * padded_blocks)
        sink_refs = refs * 95 // 100
        physical_counts[case_idx % len(physical_counts)] = sink_refs
        tail_refs = refs - sink_refs
        tail_base, tail_extra = divmod(tail_refs, len(physical_counts))
        for count_idx in range(len(physical_counts)):
            physical_counts[count_idx] += tail_base + (count_idx < tail_extra)
        actual_work = 0
        for physical_block in range(padded_blocks):
            macros = sum(
                math.ceil(physical_counts[head * padded_blocks + physical_block] / 16)
                for head in range(INDEX_HEADS)
            )
            actual_work += math.ceil(macros / 16)
        capacity_bound = SPARSE_SCHEDULE_MODEL.physical_kl_schedule_capacity(
            total_q=total_q,
            topk=16,
            head_kv=INDEX_HEADS,
            padded_kv_blocks=padded_blocks,
            q_per_macro=16,
            macros_per_work=16,
        )
        assert capacity_bound >= actual_work


@pytest.mark.gpu
def test_prepare_keeps_capacity_for_empty_attention_work() -> None:
    topk = torch.full((4, 1, 16), -1, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.tensor((0, 1), dtype=torch.int32, device="cuda")

    metadata = attention.prepare(
        topk,
        cu_seqlens,
        cu_seqlens,
        total_k=1,
        total_rows=1,
        max_seqlen_q=1,
        max_seqlen_k=1,
    )

    assert int(metadata.schedule.work_count.item()) == 0
    assert metadata.schedule.work_capacity >= 1
    assert int(metadata.schedule.dkv_split_count.item()) == 0
    assert torch.count_nonzero(metadata.schedule.dkv_owner_counts) == 0
    assert int(metadata.kl_schedule.work_count.item()) == 0
    assert int(metadata.kl_schedule.dki_split_count.item()) == 0
    assert torch.count_nonzero(metadata.kl_schedule.dki_owner_counts) == 0


@pytest.mark.gpu
def test_prepare_supports_single_batch_single_kv_block() -> None:
    cases = ((1, 1), (2, 127), (7, 128), (33, 64), (129, 128), (257, 96))
    for q_len, k_len in cases:
        q_lens = (q_len,)
        k_lens = (k_len,)
        topk = make_causal_topk(q_lens, k_lens, device="cuda")
        cu_q = torch.tensor((0, q_len), dtype=torch.int32, device="cuda")
        cu_k = torch.tensor((0, k_len), dtype=torch.int32, device="cuda")
        metadata = attention.prepare(
            topk,
            cu_q,
            cu_k,
            total_k=k_len,
            total_rows=1,
            max_seqlen_q=q_len,
            max_seqlen_k=k_len,
        )
        row_ptr_ref, q_indices_ref = csr_reference(topk, q_lens, k_lens)
        torch.testing.assert_close(metadata.k2q_row_ptr, row_ptr_ref, atol=0, rtol=0)
        torch.testing.assert_close(
            metadata.k2q_q_indices, q_indices_ref, atol=0, rtol=0
        )
        assert metadata.total_rows == 1
        assert 0 < int(metadata.schedule.work_count.item())
        assert 0 < int(metadata.kl_schedule.work_count.item())
        owner_count = int(metadata.kl_schedule.dki_owner_counts[0].item())
        assert owner_count > 0
        assert int(metadata.kl_schedule.dki_split_count.item()) == int(owner_count > 1)

    q_len, k_len = cases[0]
    topk = make_causal_topk((q_len,), (k_len,), device="cuda")
    cu_q = torch.tensor((0, q_len), dtype=torch.int32, device="cuda")
    cu_k = torch.tensor((0, k_len), dtype=torch.int32, device="cuda")
    metadata_without_kl = attention.prepare(
        topk,
        cu_q,
        cu_k,
        total_k=k_len,
        total_rows=1,
        max_seqlen_q=q_len,
        max_seqlen_k=k_len,
        prepare_kl_schedule=False,
    )
    assert not metadata_without_kl.kl_schedule.enabled
    assert int(metadata_without_kl.kl_schedule.work_count.item()) == 0


@pytest.mark.gpu
def test_prepare_builds_independent_metadata_for_each_topk_result() -> None:
    q_lens = (17, 19)
    k_lens = (4096, 4352)
    topk_a = make_causal_topk(q_lens, k_lens, device="cuda")
    topk_b = topk_a.clone()
    topk_b[:, :, :15] = torch.arange(1, 16, dtype=torch.int32, device="cuda")
    cu_q = torch.tensor((0, 17, 36), dtype=torch.int32, device="cuda")
    cu_k = torch.tensor((0, 4096, 8448), dtype=torch.int32, device="cuda")

    def prepare(topk: torch.Tensor):
        return attention.prepare(
            topk,
            cu_q,
            cu_k,
            total_k=sum(k_lens),
            total_rows=total_rows(k_lens),
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(k_lens),
        )

    metadata_a = prepare(topk_a)
    row_ptr_a = metadata_a.k2q_row_ptr.clone()
    q_indices_a = metadata_a.k2q_q_indices.clone()
    metadata_b = prepare(topk_b)

    for topk, metadata in ((topk_a, metadata_a), (topk_b, metadata_b)):
        row_ptr_ref, q_indices_ref = csr_reference(topk, q_lens, k_lens)
        torch.testing.assert_close(metadata.k2q_row_ptr, row_ptr_ref, atol=0, rtol=0)
        torch.testing.assert_close(
            metadata.k2q_q_indices, q_indices_ref, atol=0, rtol=0
        )

    torch.testing.assert_close(metadata_a.k2q_row_ptr, row_ptr_a, atol=0, rtol=0)
    torch.testing.assert_close(metadata_a.k2q_q_indices, q_indices_a, atol=0, rtol=0)
    assert not torch.equal(metadata_a.k2q_row_ptr, metadata_b.k2q_row_ptr)
    assert metadata_a.k2q_row_ptr.data_ptr() != metadata_b.k2q_row_ptr.data_ptr()
    assert (
        metadata_a.schedule.scheduler_metadata.data_ptr()
        != metadata_b.schedule.scheduler_metadata.data_ptr()
    )


@pytest.mark.gpu
def test_prepare_matches_csr_reference_and_builds_complete_schedule() -> None:
    q_lens = (33, 65)
    k_lens = (256, 384)
    topk = make_causal_topk(q_lens, k_lens, device="cuda")
    cu_q = torch.tensor((0, 33, 98), dtype=torch.int32, device="cuda")
    cu_k = torch.tensor((0, 256, 640), dtype=torch.int32, device="cuda")
    metadata = attention.prepare(
        topk,
        cu_q,
        cu_k,
        total_k=sum(k_lens),
        total_rows=total_rows(k_lens),
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
    )
    row_ptr_ref, q_indices_ref = csr_reference(topk, q_lens, k_lens)
    torch.testing.assert_close(metadata.k2q_row_ptr, row_ptr_ref, atol=0, rtol=0)
    torch.testing.assert_close(metadata.k2q_q_indices, q_indices_ref, atol=0, rtol=0)
    metadata_repeated = attention.prepare(
        topk,
        cu_q,
        cu_k,
        total_k=sum(k_lens),
        total_rows=total_rows(k_lens),
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
    )
    torch.testing.assert_close(
        metadata_repeated.k2q_row_ptr,
        metadata.k2q_row_ptr,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        metadata_repeated.k2q_q_indices,
        metadata.k2q_q_indices,
        atol=0,
        rtol=0,
    )
    assert metadata_repeated.k2q_row_ptr.data_ptr() != metadata.k2q_row_ptr.data_ptr()
    assert (
        metadata_repeated.k2q_q_indices.data_ptr() != metadata.k2q_q_indices.data_ptr()
    )
    schedule = metadata.schedule
    assert schedule.scheduler_metadata.dtype == torch.int32
    assert schedule.scheduler_metadata.shape[1] == 6
    assert schedule.work_count.dtype == torch.int32
    assert schedule.work_count.shape == (1,)
    assert schedule.qsplit_indices.shape == metadata.k2q_q_indices.shape
    assert schedule.split_counts.shape == (sum(q_lens), 4)
    work_count = int(schedule.work_count.cpu().item())
    assert 0 < work_count <= schedule.work_capacity
    active = schedule.scheduler_metadata[:work_count]
    assert (active[:, 2] >= 0).all()
    assert (active[:, 3] > 0).all()
    assert (active[:, 4] >= 0).all() and (active[:, 4] < len(q_lens)).all()
    owner_counts = schedule.dkv_owner_counts.cpu()
    expected_owner_counts = torch.zeros_like(owner_counts)
    for head, _, _, _, batch_idx, kv_block_idx in active.cpu().tolist():
        physical_block = (int(cu_k[batch_idx]) + batch_idx * 128) // 128 + kv_block_idx
        expected_owner_counts[head, physical_block] += 1
    torch.testing.assert_close(owner_counts, expected_owner_counts, atol=0, rtol=0)
    expected_split_indices = (
        torch.nonzero(
            expected_owner_counts.reshape(-1) > 1,
            as_tuple=False,
        )
        .reshape(-1)
        .to(torch.int32)
    )
    split_count = int(schedule.dkv_split_count.item())
    actual_split_indices = schedule.dkv_split_indices[:split_count].cpu()
    assert split_count == expected_split_indices.numel()
    torch.testing.assert_close(
        actual_split_indices.sort().values,
        expected_split_indices.sort().values,
        atol=0,
        rtol=0,
    )

    kl_schedule = metadata.kl_schedule
    assert kl_schedule.target_q_per_cta == 256
    assert kl_schedule.scheduler_metadata.dtype == torch.int32
    assert kl_schedule.scheduler_metadata.shape[1] == 4
    assert kl_schedule.work_count.dtype == torch.int32
    assert kl_schedule.work_count.shape == (1,)
    kl_work_count = int(kl_schedule.work_count.cpu().item())
    assert 0 < kl_work_count <= kl_schedule.work_capacity
    kl_active = kl_schedule.scheduler_metadata[:kl_work_count].cpu()
    physical_row_ptr = kl_schedule.physical_row_ptr.cpu()
    physical_q_indices = kl_schedule.physical_q_indices.cpu()
    physical_valid_rows = kl_schedule.physical_valid_rows.cpu()
    padded_blocks = kl_schedule.dki_owner_counts.numel()
    assert tuple(physical_row_ptr.shape) == (4, padded_blocks + 1)
    assert torch.all(physical_row_ptr[:, 1:] >= physical_row_ptr[:, :-1])

    expected_entries: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    topk_host = topk.cpu()
    q_offset = 0
    k_offset = 0
    for batch_idx, (q_len, k_len) in enumerate(zip(q_lens, k_lens)):
        physical_base = (k_offset + batch_idx * 128) // 128
        for q_local in range(q_len):
            q_logical = k_len - q_len + q_local
            for head in range(4):
                blocks = topk_host[head, q_offset + q_local].tolist()
                valid_blocks = [
                    block for block in blocks if 0 <= block < math.ceil(k_len / 128)
                ]
                for kv_block in blocks:
                    if kv_block < 0:
                        continue
                    valid_rows = min(128, k_len - kv_block * 128)
                    valid_rows = min(
                        valid_rows,
                        max(0, q_logical - kv_block * 128 + 1),
                    )
                    expected_entries.setdefault(
                        (head, physical_base + kv_block), []
                    ).append(
                        (
                            q_offset + q_local,
                            valid_rows,
                            sum(block < kv_block for block in valid_blocks),
                        )
                    )
        q_offset += q_len
        k_offset += k_len

    expected_macros = torch.zeros(padded_blocks, dtype=torch.int32)
    for head in range(4):
        for physical_block in range(padded_blocks):
            begin = int(physical_row_ptr[head, physical_block])
            end = int(physical_row_ptr[head, physical_block + 1])
            packed = physical_valid_rows[head, begin:end]
            actual_entries = sorted(
                zip(
                    physical_q_indices[head, begin:end].tolist(),
                    (packed & KL_VALID_ROWS_MASK).tolist(),
                    ((packed >> KL_WRITER_RANK_SHIFT) & KL_WRITER_RANK_MASK).tolist(),
                )
            )
            expected = sorted(expected_entries.get((head, physical_block), []))
            assert actual_entries == expected
            expected_macros[physical_block] += math.ceil(len(expected) / 16)

    chunks_by_block: dict[int, list[tuple[int, int]]] = {}
    expected_dki_owners = torch.zeros_like(kl_schedule.dki_owner_counts.cpu())
    for physical_block, k_offset, macro_begin, macro_count in kl_active.tolist():
        assert 0 <= physical_block < padded_blocks
        assert k_offset >= 0
        assert macro_begin >= 0
        assert 0 < macro_count <= 16
        expected_dki_owners[physical_block] += 1
        chunks_by_block.setdefault(physical_block, []).append(
            (macro_begin, macro_count)
        )
    for physical_block in range(padded_blocks):
        expected_begin = 0
        for macro_begin, macro_count in sorted(chunks_by_block.get(physical_block, [])):
            assert macro_begin == expected_begin
            expected_begin += macro_count
        assert expected_begin == int(expected_macros[physical_block])

    torch.testing.assert_close(
        kl_schedule.dki_owner_counts.cpu(), expected_dki_owners, atol=0, rtol=0
    )
    expected_split = torch.nonzero(expected_dki_owners > 1).reshape(-1).to(torch.int32)
    split_count = int(kl_schedule.dki_split_count.item())
    assert split_count == expected_split.numel()
    torch.testing.assert_close(
        kl_schedule.dki_split_indices[:split_count].cpu().sort().values,
        expected_split.sort().values,
        atol=0,
        rtol=0,
    )

    metadata_without_kl = attention.prepare(
        topk,
        cu_q,
        cu_k,
        total_k=sum(k_lens),
        total_rows=total_rows(k_lens),
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        prepare_kl_schedule=False,
    )
    assert not metadata_without_kl.kl_schedule.enabled
    assert int(metadata_without_kl.kl_schedule.work_count.cpu().item()) == 0
    torch.testing.assert_close(
        metadata_without_kl.k2q_row_ptr,
        metadata.k2q_row_ptr,
        atol=0,
        rtol=0,
    )
    with pytest.raises(ValueError, match="shape \\[4, total_q, 16\\]"):
        attention.prepare(
            topk[:, :, :8].contiguous(),
            cu_q,
            cu_k,
            total_k=sum(k_lens),
            total_rows=total_rows(k_lens),
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(k_lens),
        )


@pytest.mark.gpu
def test_prepare_metadata_reuse_across_forward_and_backward() -> None:
    torch.manual_seed(7)
    q_lens = (33, 65)
    k_lens = (256, 384)
    topk, metadata = make_packed_attention_metadata(q_lens, k_lens)
    pointers = (
        metadata.k2q_row_ptr.data_ptr(),
        metadata.k2q_q_indices.data_ptr(),
        metadata.schedule.scheduler_metadata.data_ptr(),
        metadata.schedule.work_count.data_ptr(),
    )
    q = torch.randn(
        sum(q_lens),
        Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    k = torch.randn(
        sum(k_lens),
        KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    v = torch.randn_like(k, requires_grad=True)
    out, lse = attention.forward(q, k, v, metadata, return_softmax_lse=True)
    out_ref, lse_ref = packed_attention_reference(
        q,
        k,
        v,
        topk,
        q_lens,
        k_lens,
    )
    torch.testing.assert_close(out.float(), out_ref, atol=3e-2, rtol=3e-2)
    finite = lse_ref.isfinite()
    torch.testing.assert_close(
        lse[finite],
        lse_ref[finite],
        atol=1e-3,
        rtol=1e-3,
    )

    dout = torch.randn_like(out)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        out_ref,
        (q, k, v),
        dout.float(),
    )
    assert cosine_similarity(dq, dq_ref) > 0.99
    assert cosine_similarity(dk, dk_ref) > 0.99
    assert cosine_similarity(dv, dv_ref) > 0.99
    assert all(torch.isfinite(tensor).all() for tensor in (out, lse, dq, dk, dv))

    explicit = attention.backward(
        q.detach(),
        k.detach(),
        v.detach(),
        dout,
        out.detach(),
        lse,
        metadata,
    )
    for actual, expected in zip((dq, dk, dv), explicit):
        assert cosine_similarity(actual, expected) > 0.999
    assert pointers == (
        metadata.k2q_row_ptr.data_ptr(),
        metadata.k2q_q_indices.data_ptr(),
        metadata.schedule.scheduler_metadata.data_ptr(),
        metadata.schedule.work_count.data_ptr(),
    )


@pytest.mark.gpu
def test_prepare_device_schedule_handles_1024_varlen_documents() -> None:
    torch.manual_seed(5)
    batch = 1024
    k_lens, cu_k, metadata = make_single_query_varlen_metadata(batch)
    work_count = int(metadata.schedule.work_count.item())
    assert work_count > torch.cuda.get_device_properties(0).multi_processor_count

    q = (
        (torch.randn(batch, Q_HEADS, HEAD_DIM, device="cuda") * 0.2)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    k = (
        (torch.randn(int(cu_k[-1]), KV_HEADS, HEAD_DIM, device="cuda") * 0.2)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    v = (
        (torch.randn_like(k, dtype=torch.float32) * 0.2)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    k_pad, valid = pad_packed_kv(k, k_lens, cu_k)
    v_pad, _ = pad_packed_kv(v, k_lens, cu_k)
    q_grouped = q.float().reshape(
        batch,
        INDEX_HEADS,
        Q_HEADS // INDEX_HEADS,
        HEAD_DIM,
    )
    scores = torch.einsum(
        "bgid,bkgd->bgik",
        q_grouped,
        k_pad.float(),
    ) * (1.0 / math.sqrt(HEAD_DIM))
    scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
    probability = torch.softmax(scores, dim=-1)
    out_ref = torch.einsum(
        "bgik,bkgd->bgid",
        probability,
        v_pad.float(),
    ).reshape(batch, Q_HEADS, HEAD_DIM)
    lse_ref = torch.logsumexp(scores, dim=-1).reshape(batch, Q_HEADS)

    out, lse = attention.forward(
        q,
        k,
        v,
        metadata,
        return_softmax_lse=True,
    )
    torch.testing.assert_close(out.float(), out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
    dout = torch.randn_like(out)
    grads = attention.backward(q, k, v, dout, out, lse, metadata)
    grads_ref = torch.autograd.grad(out_ref, (q, k, v), dout.float())
    for actual, reference in zip(grads, grads_ref):
        assert cosine_similarity(actual, reference) > 0.99
        assert torch.isfinite(actual).all()

    qi = (torch.randn(batch, INDEX_HEADS, HEAD_DIM, device="cuda") * 0.2).to(
        torch.bfloat16
    )
    ki = (torch.randn(int(cu_k[-1]), 1, HEAD_DIM, device="cuda") * 0.2).to(
        torch.bfloat16
    )
    ki_pad, _ = pad_packed_kv(ki, k_lens, cu_k)
    student_scores = torch.einsum(
        "bgd,bkd->bgk",
        qi.float(),
        ki_pad[:, :, 0].float(),
    ) * (1.0 / math.sqrt(HEAD_DIM))
    student_scores = student_scores.masked_fill(
        ~valid[:, None, :],
        -torch.inf,
    )
    teacher_lse = torch.logsumexp(scores, dim=-1).reshape(batch, Q_HEADS)
    indexer_lse = torch.logsumexp(student_scores, dim=-1).transpose(0, 1).contiguous()
    teacher_probability = torch.softmax(scores, dim=-1)
    student_probability = torch.softmax(student_scores, dim=-1)
    ds = (
        (student_probability - teacher_probability.mean(dim=2))
        .to(torch.bfloat16)
        .float()
    )
    grad_scale = 1.0 / math.sqrt(HEAD_DIM) / INDEX_HEADS / batch
    dqi_ref = (
        torch.einsum(
            "bgk,bkd->bgd",
            ds,
            ki_pad[:, :, 0].float(),
        )
        * grad_scale
    )
    dki_pad_ref = torch.einsum("bgk,bgd->bkd", ds, qi.float()) * grad_scale
    dki_ref = dki_pad_ref[valid]

    dqi, dki = kl.backward(
        q,
        k,
        teacher_lse,
        qi,
        ki,
        indexer_lse,
        metadata,
    )
    torch.testing.assert_close(
        dqi,
        dqi_ref.to(torch.bfloat16),
        atol=2e-3,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        dki[:, 0],
        dki_ref.to(torch.bfloat16),
        atol=2e-3,
        rtol=1e-2,
    )


@pytest.mark.gpu
def test_prepare_cuda_graph_rebuilds_metadata_from_current_topk() -> None:
    q_lens = (1,)
    k_lens = (4096,)
    topk_a = make_causal_topk(q_lens, k_lens, device="cuda")
    topk_b = topk_a.clone()
    topk_b[:, :, :15] = torch.arange(1, 16, dtype=torch.int32, device="cuda")
    topk_input = topk_a.clone()
    cu_q = torch.tensor((0, 1), dtype=torch.int32, device="cuda")
    cu_k = torch.tensor((0, 4096), dtype=torch.int32, device="cuda")

    def prepare():
        return attention.prepare(
            topk_input,
            cu_q,
            cu_k,
            total_k=4096,
            total_rows=total_rows(k_lens),
            max_seqlen_q=1,
            max_seqlen_k=4096,
        )

    prepare()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_metadata = prepare()

    graph.replay()
    torch.cuda.synchronize()
    row_ptr_a = captured_metadata.k2q_row_ptr.clone()
    q_indices_a = captured_metadata.k2q_q_indices.clone()
    row_ptr_ref, q_indices_ref = csr_reference(topk_a, q_lens, k_lens)
    torch.testing.assert_close(row_ptr_a, row_ptr_ref, atol=0, rtol=0)
    torch.testing.assert_close(q_indices_a, q_indices_ref, atol=0, rtol=0)

    topk_input.copy_(topk_b)
    graph.replay()
    torch.cuda.synchronize()
    row_ptr_ref, q_indices_ref = csr_reference(topk_b, q_lens, k_lens)
    torch.testing.assert_close(
        captured_metadata.k2q_row_ptr, row_ptr_ref, atol=0, rtol=0
    )
    torch.testing.assert_close(
        captured_metadata.k2q_q_indices,
        q_indices_ref,
        atol=0,
        rtol=0,
    )
    assert not torch.equal(captured_metadata.k2q_row_ptr, row_ptr_a)


@pytest.mark.gpu
def test_prepare_cuda_graph_capture_and_replay() -> None:
    torch.manual_seed(6)
    batch = 64
    k_lens, cu_k, metadata = make_single_query_varlen_metadata(batch)
    assert int(metadata.schedule.work_count.item()) > 0
    q = torch.randn(
        batch,
        Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.randn(
        int(cu_k[-1]),
        KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v = torch.randn_like(k)
    dout = torch.randn_like(q)
    qi = torch.randn(
        batch,
        INDEX_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    ki = torch.randn(
        int(cu_k[-1]),
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    ki_pad, valid = pad_packed_kv(ki, k_lens, cu_k)
    student_scores = torch.einsum(
        "bgd,bkd->bgk",
        qi.float(),
        ki_pad[:, :, 0].float(),
    ) * (1.0 / math.sqrt(HEAD_DIM))
    student_scores.masked_fill_(~valid[:, None, :], -torch.inf)
    indexer_lse = torch.logsumexp(student_scores, dim=-1).transpose(0, 1).contiguous()

    warm_out, warm_lse = attention.forward(
        q,
        k,
        v,
        metadata,
        return_softmax_lse=True,
        deterministic=True,
    )
    attention.backward(
        q,
        k,
        v,
        dout,
        warm_out,
        warm_lse,
        metadata,
        deterministic=True,
    )
    kl.backward(
        q,
        k,
        warm_lse,
        qi,
        ki,
        indexer_lse,
        metadata,
        deterministic=True,
    )
    torch.cuda.synchronize()
    qi = qi.clone()

    total_k = int(cu_k[-1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_metadata = attention.prepare(
            metadata.topk_indices,
            metadata.cu_seqlens_q,
            metadata.cu_seqlens_k,
            total_k=total_k,
            total_rows=batch,
            max_seqlen_q=1,
            max_seqlen_k=128,
        )
        out, lse = attention.forward(
            q,
            k,
            v,
            captured_metadata,
            return_softmax_lse=True,
            deterministic=True,
        )
        grads = attention.backward(
            q,
            k,
            v,
            dout,
            out,
            lse,
            captured_metadata,
            deterministic=True,
        )
        kl_grads = kl.backward(
            q,
            k,
            lse,
            qi,
            ki,
            indexer_lse,
            captured_metadata,
            deterministic=True,
        )

    graph.replay()
    first = tuple(
        tensor.detach().cpu().clone() for tensor in (out, lse, *grads, *kl_grads)
    )
    graph.replay()
    second = tuple(tensor.detach().cpu() for tensor in (out, lse, *grads, *kl_grads))
    for lhs, rhs in zip(first, second):
        torch.testing.assert_close(lhs, rhs, atol=2e-2, rtol=2e-2)
        assert torch.isfinite(rhs).all()
    for lhs, rhs in zip(first[2:], second[2:]):
        assert torch.equal(lhs.view(torch.int16), rhs.view(torch.int16))

    del captured_metadata, out, lse, grads, kl_grads, graph
    gc.collect()
    torch.cuda.synchronize()
    eager_value = (
        torch.arange(
            256,
            dtype=torch.float32,
            device="cuda",
        )
        .cos()
        .sum()
    )
    torch.cuda.synchronize()
    assert bool(torch.isfinite(eager_value).item())
