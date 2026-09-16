"""Real-case correctness suite for MSA v1 sparse attention forward."""

from __future__ import annotations

import pytest
import torch

from msa_v1 import attention
from tests.training.msa_v1.cute.cases import MsaTestCase
from tests.training.msa_v1.cute.testing import (
    HEAD_DIM,
    KV_HEADS,
    Q_HEADS,
    attention_full_reference,
    case_seed,
    cuda_generator,
    launch_and_sync,
    materialize_metadata,
    random_bf16,
)

pytestmark = [pytest.mark.gpu, pytest.mark.acceptance]


@torch.inference_mode()
def test_atten_fwd_real_cases(msa_case: MsaTestCase, cuda_device: torch.device) -> None:
    case = msa_case.rank_case
    generator = cuda_generator(cuda_device, case_seed(msa_case, 11))
    cu_q, cu_kv, fragments, topk = materialize_metadata(case, cuda_device)
    if msa_case.structured:
        q = torch.zeros(
            case.total_q, Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=cuda_device
        )
        k = torch.zeros(
            case.total_kv, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=cuda_device
        )
        v = torch.zeros_like(k)
    else:
        q = random_bf16(
            (case.total_q, Q_HEADS, HEAD_DIM),
            device=cuda_device,
            generator=generator,
        )
        k = random_bf16(
            (case.total_kv, KV_HEADS, HEAD_DIM),
            device=cuda_device,
            generator=generator,
        )
        v = random_bf16(
            (case.total_kv, KV_HEADS, HEAD_DIM),
            device=cuda_device,
            generator=generator,
        )
    out_ref, lse_ref = attention_full_reference(q, k, v, topk, case)
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
    assert bool(torch.isfinite(out).all().item())
    assert bool(torch.isfinite(lse).all().item())
    torch.testing.assert_close(out.float(), out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
