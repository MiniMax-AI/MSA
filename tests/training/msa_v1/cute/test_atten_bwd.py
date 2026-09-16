"""Real-case correctness suite for MSA v1 sparse attention backward."""

from __future__ import annotations

import pytest
import torch

from msa_v1 import attention
from msa_v1.attention.bwd.atten_bwd import SparseAttentionBackwardSm100
from tests.training.msa_v1.cute.cases import MsaTestCase
from tests.training.msa_v1.cute.testing import (
    HEAD_DIM,
    INDEX_HEADS,
    KV_HEADS,
    Q_HEADS,
    attention_full_reference,
    case_seed,
    cosine_similarity,
    cuda_generator,
    launch_and_sync,
    make_packed_attention_metadata,
    make_shared_fragment_metadata,
    materialize_metadata,
    packed_attention_reference,
    random_bf16,
    shared_fragment_attention_reference,
)

pytestmark = [pytest.mark.gpu, pytest.mark.acceptance]


@torch.inference_mode()
def test_atten_bwd_real_cases(msa_case: MsaTestCase, cuda_device: torch.device) -> None:
    case = msa_case.rank_case
    generator = cuda_generator(cuda_device, case_seed(msa_case, 29))
    cu_q, cu_kv, fragments, topk = materialize_metadata(case, cuda_device)
    shape_q = (case.total_q, Q_HEADS, HEAD_DIM)
    shape_kv = (case.total_kv, KV_HEADS, HEAD_DIM)
    if msa_case.structured:
        q = torch.zeros(shape_q, dtype=torch.bfloat16, device=cuda_device)
        k = torch.zeros(shape_kv, dtype=torch.bfloat16, device=cuda_device)
        v = torch.zeros_like(k)
        dout = torch.zeros_like(q)
    else:
        q = random_bf16(shape_q, device=cuda_device, generator=generator)
        k = random_bf16(shape_kv, device=cuda_device, generator=generator)
        v = random_bf16(shape_kv, device=cuda_device, generator=generator)
        dout = random_bf16(shape_q, device=cuda_device, generator=generator, scale=0.1)
    out_ref, lse_ref, dq_ref, dk_ref, dv_ref = attention_full_reference(
        q, k, v, topk, case, dout
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
    out, lse = launch_and_sync(
        "attention-fwd",
        lambda: attention.forward(q, k, v, metadata, return_softmax_lse=True),
    )
    dq, dk, dv = launch_and_sync(
        "attention-bwd-deterministic-0",
        lambda: attention.backward(
            q,
            k,
            v,
            dout,
            out,
            lse,
            metadata,
            deterministic=True,
        ),
    )
    grads = tuple(tensor.clone() for tensor in (dq, dk, dv))
    grads_repeat = launch_and_sync(
        "attention-bwd-deterministic-1",
        lambda: attention.backward(
            q,
            k,
            v,
            dout,
            out,
            lse,
            metadata,
            deterministic=True,
        ),
    )
    for actual, repeated in zip(grads, grads_repeat):
        assert torch.equal(actual.view(torch.int16), repeated.view(torch.int16))
    dq, dk, dv = grads
    torch.testing.assert_close(out.float(), out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
    for actual, reference in ((dq, dq_ref), (dk, dk_ref), (dv, dv_ref)):
        assert bool(torch.isfinite(actual).all().item())
        if not msa_case.structured:
            assert cosine_similarity(actual, reference) > 0.99


def test_bf16_shared_fragment_forward_and_backward() -> None:
    torch.manual_seed(101)
    metadata = make_shared_fragment_metadata()
    q = torch.randn(
        64,
        Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    k = torch.randn(
        256,
        KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    v = torch.randn_like(k, requires_grad=True)
    out, lse = attention.forward(
        q,
        k,
        v,
        metadata,
        return_softmax_lse=True,
        deterministic=True,
    )
    out_ref, lse_ref = shared_fragment_attention_reference(q, k, v)
    torch.testing.assert_close(out.float(), out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout)
    grads_ref = torch.autograd.grad(out_ref, (q, k, v), dout.float())
    explicit = attention.backward(
        q.detach(),
        k.detach(),
        v.detach(),
        dout,
        out.detach(),
        lse,
        metadata,
        deterministic=True,
    )
    for actual, reference, explicit_actual in zip(grads, grads_ref, explicit):
        assert cosine_similarity(actual, reference) > 0.99
        assert cosine_similarity(actual, explicit_actual) > 0.999


def test_probability_qat_shared_fragment_backward_is_finite() -> None:
    torch.manual_seed(103)
    metadata = make_shared_fragment_metadata()
    q_fp8 = torch.randn(
        64,
        Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    k_fp8 = torch.randn(
        256,
        KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    v_fp8 = torch.randn(
        256,
        KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    q = q_fp8.to(torch.bfloat16)
    k = k_fp8.to(torch.bfloat16)
    v = v_fp8.to(torch.bfloat16)
    out, lse = attention.forward(
        q_fp8,
        k_fp8,
        v_fp8,
        metadata,
        return_softmax_lse=True,
    )
    dout = torch.randn_like(out)
    with pytest.raises(ValueError, match="q_fp8 and k_fp8 are required"):
        attention.backward(
            q,
            k,
            v,
            dout,
            out,
            lse,
            metadata,
            sparse_attn_p_mode="fp8",
            deterministic=True,
        )
    grads = attention.backward(
        q,
        k,
        v,
        dout,
        out,
        lse,
        metadata,
        sparse_attn_p_mode="fp8",
        q_fp8=q_fp8,
        k_fp8=k_fp8,
        deterministic=True,
    )
    grads_repeat = attention.backward(
        q,
        k,
        v,
        dout,
        out,
        lse,
        metadata,
        sparse_attn_p_mode="fp8",
        q_fp8=q_fp8,
        k_fp8=k_fp8,
        deterministic=True,
    )
    for actual, repeated in zip(grads, grads_repeat):
        assert torch.equal(actual.view(torch.int16), repeated.view(torch.int16))
    assert all(bool(tensor.isfinite().all().item()) for tensor in (out, lse, *grads))


def test_native_fp8_shared_fragment_backward_is_rejected() -> None:
    torch.manual_seed(107)
    metadata = make_shared_fragment_metadata()
    q = torch.randn(
        64,
        Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    k = torch.randn(
        256,
        KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    v = torch.randn(
        256,
        KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    out = torch.zeros(
        (64, Q_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    lse = torch.zeros(
        (64, Q_HEADS),
        dtype=torch.float32,
        device="cuda",
    )
    with pytest.raises(NotImplementedError, match="only BF16 Q/K/V"):
        attention.backward(
            q,
            k,
            v,
            torch.randn_like(out),
            out,
            lse,
            metadata,
        )


@pytest.mark.parametrize("qhead_per_kvhead", (1, 2, 4, 8))
def test_backward_kernel_requires_gqa16(qhead_per_kvhead: int) -> None:
    with pytest.raises(NotImplementedError, match="qhead_per_kvhead=16"):
        SparseAttentionBackwardSm100(
            head_dim=HEAD_DIM,
            qhead_per_kvhead=qhead_per_kvhead,
        )


def test_bf16_backward_single_and_zero_owner_dkv_blocks() -> None:
    torch.manual_seed(9)
    q_lens = (1,)
    k_lens = (4096,)
    topk, metadata = make_packed_attention_metadata(q_lens, k_lens)
    owners = metadata.schedule.dkv_owner_counts
    assert int(metadata.schedule.dkv_split_count.item()) == 0
    assert torch.count_nonzero(owners == 1) == INDEX_HEADS * 16
    assert torch.count_nonzero(owners == 0) == owners.numel() - INDEX_HEADS * 16

    q = torch.randn(
        1,
        Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    k = torch.randn(
        4096,
        KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    v = torch.randn_like(k, requires_grad=True)
    out, _lse = attention.forward(q, k, v, metadata, return_softmax_lse=True)
    out_ref, _ = packed_attention_reference(
        q,
        k,
        v,
        topk,
        q_lens,
        k_lens,
    )
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout)
    grads_ref = torch.autograd.grad(out_ref, (q, k, v), dout.float())
    for grad, grad_ref in zip(grads, grads_ref):
        assert cosine_similarity(grad, grad_ref) > 0.99
        assert torch.isfinite(grad).all()

    selected = set(torch.unique(topk[topk >= 0]).tolist())
    for block_idx in sorted(set(range(32)) - selected):
        begin = block_idx * 128
        end = begin + 128
        assert torch.count_nonzero(grads[1][begin:end]) == 0
        assert torch.count_nonzero(grads[2][begin:end]) == 0


def test_bf16_backward_unaligned_varlen_dkv_tail() -> None:
    torch.manual_seed(10)
    q_lens = (33, 65)
    k_lens = (257, 385)
    topk, metadata = make_packed_attention_metadata(q_lens, k_lens)
    owners = metadata.schedule.dkv_owner_counts
    assert torch.all(owners[:, torch.tensor((2, 6), device="cuda")] == 1)

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
    out_ref, _ = packed_attention_reference(
        q,
        k,
        v,
        topk,
        q_lens,
        k_lens,
    )
    dout = torch.randn_like(out)
    grads = attention.backward(
        q, k, v, dout, out, lse, metadata, deterministic=True
    )
    grads_ref = torch.autograd.grad(out_ref, (q, k, v), dout.float())
    for grad, grad_ref in zip(grads, grads_ref):
        assert cosine_similarity(grad, grad_ref) > 0.99
        assert torch.isfinite(grad).all()


def test_bf16_backward_skips_empty_last_document_dkv_tiles() -> None:
    torch.manual_seed(12)
    q_lens = (33, 1)
    k_lens = (385, 1)
    topk, metadata = make_packed_attention_metadata(q_lens, k_lens)
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
    out_ref, _ = packed_attention_reference(
        q,
        k,
        v,
        topk,
        q_lens,
        k_lens,
    )
    dout = torch.randn_like(out)
    grads = attention.backward(
        q, k, v, dout, out, lse, metadata, deterministic=True
    )
    grads_ref = torch.autograd.grad(out_ref, (q, k, v), dout.float())
    for grad, grad_ref in zip(grads, grads_ref):
        assert cosine_similarity(grad, grad_ref) > 0.99
        assert torch.isfinite(grad).all()


@pytest.mark.parametrize("bf16_qat", (False, True))
@pytest.mark.parametrize("temperature", (0.5, 2.0))
def test_probability_qat_matches_logical_probability_ste_reference(
    bf16_qat: bool, temperature: float
) -> None:
    torch.manual_seed(11)
    q_lens = (65,)
    k_lens = (384,)
    topk, metadata = make_packed_attention_metadata(q_lens, k_lens)
    q_fp8 = torch.randn(
        65,
        Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    k_fp8 = torch.randn(
        384,
        KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    v_fp8 = torch.randn(
        384,
        KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    q = q_fp8.to(torch.bfloat16).requires_grad_(True)
    k = k_fp8.to(torch.bfloat16).requires_grad_(True)
    v = v_fp8.to(torch.bfloat16).requires_grad_(True)
    out, lse, temperature_lse = attention.forward(
        q.detach() if bf16_qat else q_fp8,
        k.detach() if bf16_qat else k_fp8,
        v.detach() if bf16_qat else v_fp8,
        metadata,
        return_softmax_lse=True,
        return_temperature_lse=True,
        lse_temperature_scale=temperature,
        sparse_attn_p_mode="fp8" if bf16_qat else "",
    )
    out_ref, lse_ref = packed_attention_reference(
        q,
        k,
        v,
        topk,
        q_lens,
        k_lens,
        probability_qat=True,
    )
    torch.testing.assert_close(out.float(), out_ref, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
    with torch.no_grad():
        _, temperature_lse_ref = packed_attention_reference(
            q.float() / temperature, k, v, topk, q_lens, k_lens
        )
    assert torch.isfinite(temperature_lse).all()
    torch.testing.assert_close(
        temperature_lse, temperature_lse_ref, atol=1e-3, rtol=1e-3
    )
    dout = torch.randn_like(out)
    grads = attention.backward(
        q.detach(),
        k.detach(),
        v.detach(),
        dout,
        out,
        lse,
        metadata,
        sparse_attn_p_mode="fp8",
        q_fp8=q_fp8,
        k_fp8=k_fp8,
        deterministic=True,
    )
    grads_ref = torch.autograd.grad(out_ref, (q, k, v), dout.float())
    for grad, grad_ref in zip(grads, grads_ref):
        assert cosine_similarity(grad, grad_ref) > 0.99
        assert torch.isfinite(grad).all()


def test_native_fp8_backward_is_rejected() -> None:
    torch.manual_seed(19)
    q_lens = (64,)
    k_lens = (256,)
    _, metadata = make_packed_attention_metadata(q_lens, k_lens)
    q = torch.randn(
        64,
        Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    k = torch.randn(
        256,
        KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    v = torch.randn(
        256,
        KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    out = torch.zeros(
        (64, Q_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    lse = torch.zeros(
        (64, Q_HEADS),
        dtype=torch.float32,
        device="cuda",
    )
    with pytest.raises(NotImplementedError, match="only BF16 Q/K/V"):
        attention.backward(
            q,
            k,
            v,
            torch.randn_like(out),
            out,
            lse,
            metadata,
        )
