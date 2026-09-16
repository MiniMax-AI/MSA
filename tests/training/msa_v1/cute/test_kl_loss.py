"""Real-case correctness suite for MSA v1 sparse KL backward."""

from __future__ import annotations

import pytest
import torch

from msa_v1 import attention, kl
from tests.training.msa_v1.cute.cases import MsaTestCase
from tests.training.msa_v1.cute.testing import (
    HEAD_DIM,
    INDEX_HEADS,
    KV_HEADS,
    Q_HEADS,
    case_seed,
    cosine_similarity,
    cuda_generator,
    kl_full_reference,
    kl_shared_fragment_reference,
    launch_and_sync,
    make_shared_fragment_metadata,
    materialize_metadata,
    random_bf16,
)

pytestmark = [pytest.mark.gpu, pytest.mark.acceptance]

_MIN_COSINE_SIMILARITY = 0.99


def _assert_gradient_matches_reference(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    assert cosine_similarity(actual, reference) > _MIN_COSINE_SIMILARITY


@torch.inference_mode()
def test_kl_loss_real_cases(
    msa_case: MsaTestCase,
    cuda_device: torch.device,
) -> None:
    case = msa_case.rank_case
    generator = cuda_generator(cuda_device, case_seed(msa_case, 43))
    cu_q, cu_kv, fragments, topk = materialize_metadata(case, cuda_device)
    shape_q = (case.total_q, Q_HEADS, HEAD_DIM)
    shape_k = (case.total_kv, KV_HEADS, HEAD_DIM)
    shape_qi = (case.total_q, INDEX_HEADS, HEAD_DIM)
    shape_ki = (case.total_kv, 1, HEAD_DIM)
    if msa_case.structured:
        q = torch.zeros(shape_q, dtype=torch.bfloat16, device=cuda_device)
        k = torch.zeros(shape_k, dtype=torch.bfloat16, device=cuda_device)
        qi = torch.zeros(shape_qi, dtype=torch.bfloat16, device=cuda_device)
        ki = torch.zeros(shape_ki, dtype=torch.bfloat16, device=cuda_device)
    else:
        q = random_bf16(shape_q, device=cuda_device, generator=generator)
        k = random_bf16(shape_k, device=cuda_device, generator=generator)
        qi = random_bf16(shape_qi, device=cuda_device, generator=generator)
        ki = random_bf16(shape_ki, device=cuda_device, generator=generator)

    teacher_lse, indexer_lse, dqi_ref, dki_ref = kl_full_reference(
        q, k, qi, ki, topk, case
    )
    metadata = launch_and_sync(
        "attention-prepare",
        lambda: attention.prepare(
            topk,
            cu_q,
            cu_kv,
            total_k=case.total_kv,
            total_rows=case.total_kv_rows,
            max_seqlen_q=case.max_seqlen_q,
            max_seqlen_k=case.max_seqlen_kv,
            fragment_indices=fragments,
        ),
    )
    dqi, dki = launch_and_sync(
        "kl-backward-deterministic-0",
        lambda: kl.backward(
            q,
            k,
            teacher_lse,
            qi,
            ki,
            indexer_lse,
            metadata,
            deterministic=True,
        ),
    )
    grads = (dqi.clone(), dki.clone())
    grads_repeat = launch_and_sync(
        "kl-backward-deterministic-1",
        lambda: kl.backward(
            q,
            k,
            teacher_lse,
            qi,
            ki,
            indexer_lse,
            metadata,
            deterministic=True,
        ),
    )
    for actual, repeated in zip(grads, grads_repeat):
        assert torch.equal(actual.view(torch.int16), repeated.view(torch.int16))
    dqi, dki = grads

    assert dqi.dtype == torch.bfloat16
    assert dki.dtype == torch.bfloat16
    assert bool(torch.isfinite(dqi).all().item())
    assert bool(torch.isfinite(dki).all().item())
    if msa_case.structured:
        torch.testing.assert_close(dqi, dqi_ref, atol=0, rtol=0)
        torch.testing.assert_close(dki, dki_ref, atol=0, rtol=0)
    else:
        _assert_gradient_matches_reference(dqi, dqi_ref)
        _assert_gradient_matches_reference(dki, dki_ref)


def test_kl_shared_fragment_backward_matches_fp32_reference() -> None:
    torch.manual_seed(109)
    metadata = make_shared_fragment_metadata()
    q = (torch.randn(64, Q_HEADS, HEAD_DIM, device="cuda") * 0.2).to(torch.bfloat16)
    k = (torch.randn(256, KV_HEADS, HEAD_DIM, device="cuda") * 0.2).to(torch.bfloat16)
    qi = (torch.randn(64, INDEX_HEADS, HEAD_DIM, device="cuda") * 0.2).to(
        torch.bfloat16
    )
    ki = (torch.randn(256, 1, HEAD_DIM, device="cuda") * 0.2).to(torch.bfloat16)
    teacher_lse, indexer_lse, dqi_ref, dki_ref = kl_shared_fragment_reference(
        q, k, qi, ki, metadata
    )

    dqi, dki = kl.backward(
        q,
        k,
        teacher_lse,
        qi,
        ki,
        indexer_lse,
        metadata,
        deterministic=True,
    )
    dqi_repeat, dki_repeat = kl.backward(
        q,
        k,
        teacher_lse,
        qi,
        ki,
        indexer_lse,
        metadata,
        deterministic=True,
    )

    assert torch.equal(dqi.view(torch.int16), dqi_repeat.view(torch.int16))
    assert torch.equal(dki.view(torch.int16), dki_repeat.view(torch.int16))
    torch.testing.assert_close(dqi, dqi_ref, atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(dki, dki_ref, atol=2e-3, rtol=1e-2)


def test_kl_zero_owner_blocks_write_zero_gradients() -> None:
    torch.manual_seed(113)
    q_len, k_len = 7, 128
    topk = torch.full(
        (INDEX_HEADS, q_len, 16),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
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
    assert int(metadata.kl_schedule.work_count.item()) == 0
    assert int(metadata.kl_schedule.dki_split_count.item()) == 0

    q = torch.randn(
        q_len,
        Q_HEADS,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        k_len,
        KV_HEADS,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    qi = torch.randn(
        q_len,
        INDEX_HEADS,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    ki = torch.randn(
        k_len,
        1,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    teacher_lse = torch.zeros(
        q_len,
        Q_HEADS,
        dtype=torch.float32,
        device="cuda",
    )
    indexer_lse = torch.zeros(
        INDEX_HEADS,
        q_len,
        dtype=torch.float32,
        device="cuda",
    )

    dqi, dki = kl.backward(
        q,
        k,
        teacher_lse,
        qi,
        ki,
        indexer_lse,
        metadata,
    )

    assert torch.count_nonzero(dqi) == 0
    assert torch.count_nonzero(dki) == 0


def test_kl_single_owner_padded_store_matches_reference() -> None:
    torch.manual_seed(127)
    q_len, k_len = 1, 128
    topk = torch.full(
        (INDEX_HEADS, q_len, 16),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    topk[0, 0, 0] = 0
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
    assert int(metadata.kl_schedule.work_count.item()) == 1
    assert int(metadata.kl_schedule.dki_owner_counts[0].item()) == 1
    assert int(metadata.kl_schedule.dki_split_count.item()) == 0

    q = (torch.randn(q_len, Q_HEADS, HEAD_DIM, device="cuda") * 0.2).to(torch.bfloat16)
    k = (torch.randn(k_len, KV_HEADS, HEAD_DIM, device="cuda") * 0.2).to(torch.bfloat16)
    qi = (torch.randn(q_len, INDEX_HEADS, HEAD_DIM, device="cuda") * 0.2).to(
        torch.bfloat16
    )
    ki = (torch.randn(k_len, 1, HEAD_DIM, device="cuda") * 0.2).to(torch.bfloat16)
    scale = HEAD_DIM**-0.5

    teacher_scores = (
        torch.einsum(
            "hd,kd->hk",
            q[0, : Q_HEADS // INDEX_HEADS].float(),
            k[:, 0].float(),
        )
        * scale
    )
    student_scores = (
        torch.einsum(
            "d,kd->k",
            qi[0, 0].float(),
            ki[:, 0].float(),
        )
        * scale
    )
    teacher_lse = torch.zeros(
        q_len,
        Q_HEADS,
        dtype=torch.float32,
        device="cuda",
    )
    teacher_lse[0, : Q_HEADS // INDEX_HEADS] = torch.logsumexp(
        teacher_scores,
        dim=-1,
    )
    indexer_lse = torch.zeros(
        INDEX_HEADS,
        q_len,
        dtype=torch.float32,
        device="cuda",
    )
    indexer_lse[0, 0] = torch.logsumexp(student_scores, dim=-1)
    teacher_mean = torch.softmax(teacher_scores, dim=-1).mean(dim=0)
    ds = (
        (torch.softmax(student_scores, dim=-1) - teacher_mean)
        .to(torch.bfloat16)
        .float()
    )
    grad_scale = scale / INDEX_HEADS
    dqi_ref = torch.zeros_like(qi, dtype=torch.float32)
    dqi_ref[0, 0] = (
        torch.einsum(
            "k,kd->d",
            ds,
            ki[:, 0].float(),
        )
        * grad_scale
    )
    dki_ref = torch.einsum("k,d->kd", ds, qi[0, 0].float()) * grad_scale

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


def test_kl_adjacent_partial_single_owner_blocks_do_not_overlap() -> None:
    torch.manual_seed(129)
    q_lens = (1, 1)
    k_lens = (39, 35)
    cu_q = torch.tensor((0, 1, 2), dtype=torch.int32, device="cuda")
    cu_k = torch.tensor((0, 39, 74), dtype=torch.int32, device="cuda")
    topk = torch.full(
        (INDEX_HEADS, sum(q_lens), 16),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    topk[0, :, 0] = 0
    metadata = attention.prepare(
        topk,
        cu_q,
        cu_k,
        total_k=sum(k_lens),
        total_rows=2,
        max_seqlen_q=1,
        max_seqlen_k=max(k_lens),
    )
    assert int(metadata.kl_schedule.work_count.item()) == 2
    assert int(metadata.kl_schedule.dki_split_count.item()) == 0

    q = (torch.randn(2, Q_HEADS, HEAD_DIM, device="cuda") * 0.2).to(
        torch.bfloat16
    )
    k = (torch.randn(74, KV_HEADS, HEAD_DIM, device="cuda") * 0.2).to(
        torch.bfloat16
    )
    qi = (torch.randn(2, INDEX_HEADS, HEAD_DIM, device="cuda") * 0.2).to(
        torch.bfloat16
    )
    ki = (torch.randn(74, 1, HEAD_DIM, device="cuda") * 0.2).to(
        torch.bfloat16
    )
    teacher_lse = torch.zeros(2, Q_HEADS, dtype=torch.float32, device="cuda")
    indexer_lse = torch.zeros(INDEX_HEADS, 2, dtype=torch.float32, device="cuda")
    dqi_ref = torch.zeros_like(qi, dtype=torch.float32)
    dki_ref = torch.zeros_like(ki, dtype=torch.float32)
    scale = HEAD_DIM**-0.5
    grad_scale = scale / INDEX_HEADS / sum(q_lens)

    for q_idx, (k_begin, k_end) in enumerate(((0, 39), (39, 74))):
        teacher_scores = torch.einsum(
            "hd,kd->hk",
            q[q_idx, : Q_HEADS // INDEX_HEADS].float(),
            k[k_begin:k_end, 0].float(),
        ) * scale
        student_scores = torch.einsum(
            "d,kd->k",
            qi[q_idx, 0].float(),
            ki[k_begin:k_end, 0].float(),
        ) * scale
        teacher_lse[q_idx, : Q_HEADS // INDEX_HEADS] = torch.logsumexp(
            teacher_scores, dim=-1
        )
        indexer_lse[0, q_idx] = torch.logsumexp(student_scores, dim=-1)
        ds = (
            torch.softmax(student_scores, dim=-1)
            - torch.softmax(teacher_scores, dim=-1).mean(dim=0)
        ).to(torch.bfloat16).float()
        dqi_ref[q_idx, 0] = (
            torch.einsum("k,kd->d", ds, ki[k_begin:k_end, 0].float())
            * grad_scale
        )
        dki_ref[k_begin:k_end, 0] = (
            torch.einsum("k,d->kd", ds, qi[q_idx, 0].float()) * grad_scale
        )

    dqi_ref = dqi_ref.to(torch.bfloat16)
    dki_ref = dki_ref.to(torch.bfloat16)
    for _ in range(20):
        dqi, dki = kl.backward(
            q, k, teacher_lse, qi, ki, indexer_lse, metadata
        )
        torch.cuda.synchronize()
        _assert_gradient_matches_reference(dqi, dqi_ref)
        _assert_gradient_matches_reference(dki[:39], dki_ref[:39])
        _assert_gradient_matches_reference(dki[39:], dki_ref[39:])


def test_kl_split_owner_reduction_matches_reference() -> None:
    torch.manual_seed(131)
    q_len, k_len = 65, 128
    topk = torch.full(
        (INDEX_HEADS, q_len, 16),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    topk[:, :, 0] = 0
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
    assert int(metadata.kl_schedule.work_count.item()) == 2
    assert int(metadata.kl_schedule.dki_owner_counts[0].item()) == 2
    assert int(metadata.kl_schedule.dki_split_count.item()) == 1

    q = (torch.randn(q_len, Q_HEADS, HEAD_DIM, device="cuda") * 0.2).to(
        torch.bfloat16
    )
    k = (torch.randn(k_len, KV_HEADS, HEAD_DIM, device="cuda") * 0.2).to(
        torch.bfloat16
    )
    qi = (torch.randn(q_len, INDEX_HEADS, HEAD_DIM, device="cuda") * 0.2).to(
        torch.bfloat16
    )
    ki = (torch.randn(k_len, 1, HEAD_DIM, device="cuda") * 0.2).to(
        torch.bfloat16
    )
    scale = HEAD_DIM**-0.5
    q_grouped = q.float().reshape(q_len, INDEX_HEADS, 16, HEAD_DIM)
    teacher_scores = torch.einsum(
        "qhtd,khd->qhtk", q_grouped, k.float()
    ) * scale
    student_scores = torch.einsum(
        "qhd,kd->qhk", qi.float(), ki[:, 0].float()
    ) * scale
    key_idx = torch.arange(k_len, device="cuda")
    visible_rows = torch.arange(64, 129, device="cuda")
    visible = key_idx[None, :] < visible_rows[:, None]
    teacher_scores = teacher_scores.masked_fill(
        ~visible[:, None, None, :], -torch.inf
    )
    student_scores = student_scores.masked_fill(
        ~visible[:, None, :], -torch.inf
    )
    teacher_lse = torch.logsumexp(teacher_scores, dim=-1).reshape(q_len, Q_HEADS)
    indexer_lse = torch.logsumexp(student_scores, dim=-1).transpose(0, 1).contiguous()
    ds = (
        torch.softmax(student_scores, dim=-1)
        - torch.softmax(teacher_scores, dim=-1).mean(dim=2)
    ).to(torch.bfloat16).float()
    grad_scale = scale / INDEX_HEADS / q_len
    dqi_ref = torch.einsum("qhk,kd->qhd", ds, ki[:, 0].float()) * grad_scale
    dki_ref = torch.einsum("qhk,qhd->kd", ds, qi.float()) * grad_scale

    dqi, dki = kl.backward(
        q,
        k,
        teacher_lse,
        qi,
        ki,
        indexer_lse,
        metadata,
        deterministic=True,
    )
    dqi_repeat, dki_repeat = kl.backward(
        q,
        k,
        teacher_lse,
        qi,
        ki,
        indexer_lse,
        metadata,
        deterministic=True,
    )

    assert torch.equal(dqi.view(torch.int16), dqi_repeat.view(torch.int16))
    assert torch.equal(dki.view(torch.int16), dki_repeat.view(torch.int16))
    torch.testing.assert_close(dqi, dqi_ref.to(torch.bfloat16), atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(
        dki[:, 0], dki_ref.to(torch.bfloat16), atol=2e-3, rtol=1e-2
    )


def test_kl_clc_work_count_matches_resident_ctas() -> None:
    batch = torch.cuda.get_device_properties(0).multi_processor_count
    cu_seqlens = torch.arange(
        batch + 1,
        dtype=torch.int32,
        device="cuda",
    )
    topk = torch.full(
        (INDEX_HEADS, batch, 16),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    topk[0, :, 0] = 0
    metadata = attention.prepare(
        topk,
        cu_seqlens,
        cu_seqlens,
        total_k=batch,
        total_rows=batch,
        max_seqlen_q=1,
        max_seqlen_k=1,
    )
    assert int(metadata.kl_schedule.work_count.item()) == batch

    q = torch.zeros(
        batch,
        Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.zeros(
        batch,
        KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    qi = torch.zeros(
        batch,
        INDEX_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    ki = torch.zeros(
        batch,
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    teacher_lse = torch.zeros(
        batch,
        Q_HEADS,
        dtype=torch.float32,
        device="cuda",
    )
    indexer_lse = torch.zeros(
        INDEX_HEADS,
        batch,
        dtype=torch.float32,
        device="cuda",
    )

    dqi, dki = kl.backward(
        q,
        k,
        teacher_lse,
        qi,
        ki,
        indexer_lse,
        metadata,
    )

    assert torch.count_nonzero(dqi) == 0
    assert torch.count_nonzero(dki) == 0
