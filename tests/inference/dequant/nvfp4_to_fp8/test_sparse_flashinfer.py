"""Optional sparse-dequant integration with external FlashInfer Q8K8 decode."""

from __future__ import annotations

import pytest
import torch

from inference.dequant import SparsePagedNvfp4ToFp8Wrapper
from inference.dequant.nvfp4_to_fp8 import jit
from inference.msa_v1.attention.decode._flashinfer import load_backend
from tests.inference.msa_v1.attention.decode.q8kv4.real_cases import (
    make_decode_attention_inputs,
)
from tests.inference.msa_v1.attention.decode.q8kv4.reference import (
    decode_attention_reference,
)
from tests.inference.msa_v1.attention.decode.q8kv8.runtime import compile_module
from tests.inference.msa_v1.attention.decode.runtime import run_cuda

pytestmark = pytest.mark.gpu


def _require_backend() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
        pytest.skip("sparse dequant integration requires SM100 or SM103")
    try:
        compile_module()
    except RuntimeError as error:
        pytest.skip(str(error))
    return device


@pytest.mark.parametrize("gqa_ratio", (8, 16), ids=("gqa8", "gqa16"))
@pytest.mark.parametrize("num_kv_heads,q_len_per_req", ((1, 1), (2, 4), (4, 8)))
def test_compact_hnd_cache_runs_through_flashinfer(
    gqa_ratio, num_kv_heads, q_len_per_req
) -> None:
    device = _require_backend()
    inputs = make_decode_attention_inputs(
        torch.tensor([385, 257], dtype=torch.int32),
        seed=1701,
        device=device,
        q_len_per_req=q_len_per_req,
        page_layout="permuted",
        num_q_heads=gqa_ratio * num_kv_heads,
        num_kv_heads=num_kv_heads,
    )
    wrapper = SparsePagedNvfp4ToFp8Wrapper()
    wrapper.plan(
        inputs.topk_indices,
        inputs.page_table,
        inputs.seq_lens,
        q_len_per_req=inputs.q_len_per_req,
    )
    jit.load_extension(device)
    dequantized = run_cuda(
        "sparse dequant warmup",
        lambda: wrapper.run(
            (inputs.packed_k, inputs.packed_v),
            kv_cache_sf=(inputs.k_scale, inputs.v_scale),
        ),
    )
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    counter_bytes = (
        (max(inputs.q.shape[0] * inputs.q.shape[1], sm_count) + 7) // 8 * 8
    ) * 4
    counter = torch.zeros(counter_bytes, dtype=torch.uint8, device=device)
    output = torch.empty_like(inputs.q, dtype=torch.bfloat16)

    def attend(cache):
        return load_backend().run(
            inputs.q,
            cache.paged_kv_cache,
            workspace,
            cache.block_tables,
            cache.seq_lens,
            cache.max_seq_len,
            bmm1_scale=128**-0.5,
            bmm2_scale=1.0,
            out=output,
            out_dtype=torch.bfloat16,
            kv_layout="HND",
            backend="trtllm-gen",
            q_len_per_req=1,
            enable_pdl=True,
            multi_ctas_kv_counter_buffer=counter,
            enable_block_sparse_attention=True,
        )

    actual = run_cuda("sparse FlashInfer warmup", lambda: attend(dequantized))
    expected = decode_attention_reference(inputs)
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_fp32.flatten(), expected_fp32.flatten(), dim=0
    )
    assert float(cosine) >= 0.999
    torch.testing.assert_close(actual_fp32, expected_fp32, atol=0.05, rtol=0.05)

    # Capture the complete chain, including sparse dequant on every invocation.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        current = wrapper.run(
            (inputs.packed_k, inputs.packed_v),
            kv_cache_sf=(inputs.k_scale, inputs.v_scale),
        )
        attend(current)
    for _ in range(3):
        output.fill_(-777)
        run_cuda("sparse-dequant + FlashInfer graph", graph.replay)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.float(), expected_fp32, atol=0.05, rtol=0.05)
