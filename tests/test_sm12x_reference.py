# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import pytest
import torch


def _need_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _need_sm12x_cuda() -> None:
    _need_cuda()
    major, minor = torch.cuda.get_device_capability()
    if major != 12 or minor not in (0, 1):
        pytest.skip("SM120/SM121 CUDA device is required")


def test_sm12x_dense_reference_matches_torch() -> None:
    _need_cuda()
    from fmha_sm12x import fmha_sm12x, fmha_sm12x_plan

    device = torch.device("cuda")
    q = torch.randn((3, 2, 16), device=device, dtype=torch.bfloat16)
    k = torch.randn((5, 1, 16), device=device, dtype=torch.bfloat16)
    v = torch.randn((5, 1, 16), device=device, dtype=torch.bfloat16)
    qo = torch.tensor([1, 2], dtype=torch.int32)
    kv = torch.tensor([2, 3], dtype=torch.int32)
    plan = fmha_sm12x_plan(qo, kv, 2, 1, causal=True)

    out, _ = fmha_sm12x(q, k, v, plan, sm_scale=1.0 / math.sqrt(16))

    refs = []
    q_start = 0
    k_start = 0
    for q_len, k_len in [(1, 2), (2, 3)]:
        per_batch = []
        for local_q in range(q_len):
            visible = k_len - q_len + local_q + 1
            per_head = []
            for head in range(2):
                logits = torch.matmul(
                    k[k_start : k_start + visible, 0].float(),
                    q[q_start + local_q, head].float(),
                ) / math.sqrt(16)
                per_head.append(torch.matmul(torch.softmax(logits, dim=0), v[k_start : k_start + visible, 0].float()))
            per_batch.append(torch.stack(per_head, dim=0))
        refs.append(torch.stack(per_batch, dim=0))
        q_start += q_len
        k_start += k_len
    expected = torch.cat(refs, dim=0).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)


def test_sm12x_sparse_reference_uses_selected_blocks() -> None:
    _need_cuda()
    from fmha_sm12x import fmha_sm12x, fmha_sm12x_plan

    device = torch.device("cuda")
    page = 2
    q = torch.randn((1, 1, 8), device=device, dtype=torch.bfloat16)
    k = torch.randn((4, 1, 8), device=device, dtype=torch.bfloat16)
    v = torch.randn((4, 1, 8), device=device, dtype=torch.bfloat16)
    blocks = torch.tensor([[[1]]], device=device, dtype=torch.int32)
    plan = fmha_sm12x_plan(torch.tensor([1], dtype=torch.int32), torch.tensor([4], dtype=torch.int32), 1, 1, qo_offset=3, page_size=page, kv_block_num=1)

    out, _ = fmha_sm12x(q, k, v, plan, kv_block_indexes=blocks, sm_scale=1.0 / math.sqrt(8))

    logits = torch.matmul(k[2:4, 0].float(), q[0, 0].float()) / math.sqrt(8)
    expected = torch.matmul(torch.softmax(logits, dim=0), v[2:4, 0].float()).to(torch.bfloat16)
    torch.testing.assert_close(out[0, 0], expected, atol=2e-2, rtol=2e-2)


def test_sm12x_reference_rejects_plan_total_q_mismatch() -> None:
    _need_cuda()
    from fmha_sm12x import fmha_sm12x, fmha_sm12x_plan

    q = torch.randn((2, 1, 16), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 1, 16), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    plan = fmha_sm12x_plan(
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        1,
        1,
        causal=False,
    )

    with pytest.raises(ValueError, match="sum\\(qo_segment_lens\\)"):
        fmha_sm12x(q, k, v, plan)


def test_sm12x_build_k2q_rejects_cross_batch_block_index() -> None:
    _need_cuda()
    from fmha_sm12x import build_k2q_csr

    q2k = torch.tensor([[[1], [0]]], device="cuda", dtype=torch.int32)
    cu_q = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="batch 0"):
        build_k2q_csr(q2k, cu_q, cu_k, 2)


def test_sm12x_sparse_rejects_invalid_csr_query_index() -> None:
    _need_cuda()
    from fmha_sm12x import sparse_atten_func

    q = torch.randn((1, 1, 16), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((2, 1, 16), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    row_ptr = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    q_indices = torch.tensor([[-1]], device="cuda", dtype=torch.int32)
    cu_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 2], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="q index"):
        sparse_atten_func(
            q,
            k,
            v,
            row_ptr,
            q_indices,
            1,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=1,
            max_seqlen_k=2,
            blk_kv=2,
            causal=False,
        )


def test_sm12x_sparse_atten_temperature_lse_parity() -> None:
    # Parity with SM100 sparse_atten_func: return_temperature_lse yields a
    # 3-tuple (out, lse, temperature_lse); with lse_temperature_scale=1.0 the
    # temperature LSE equals the plain LSE, and a >1 scale shrinks it.
    _need_cuda()
    from fmha_sm12x import sparse_atten_func

    torch.manual_seed(0)
    q = torch.randn((2, 1, 16), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((2, 1, 16), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    row_ptr = torch.tensor([[0, 2]], device="cuda", dtype=torch.int32)
    q_indices = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    cu_q = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    common = dict(
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=2, max_seqlen_k=2,
        blk_kv=2, causal=False, softmax_scale=0.5, return_softmax_lse=True,
    )

    out, lse = sparse_atten_func(q, k, v, row_ptr, q_indices, 1, **common)
    out3, lse3, temp_lse = sparse_atten_func(
        q, k, v, row_ptr, q_indices, 1, return_temperature_lse=True,
        lse_temperature_scale=1.0, **common,
    )
    torch.testing.assert_close(out3, out, atol=0, rtol=0)
    torch.testing.assert_close(lse3, lse, atol=0, rtol=0)
    torch.testing.assert_close(temp_lse, lse, atol=1e-5, rtol=1e-5)

    _, _, temp_lse2 = sparse_atten_func(
        q, k, v, row_ptr, q_indices, 1, return_temperature_lse=True,
        lse_temperature_scale=4.0, **common,
    )
    assert temp_lse2.shape == lse.shape
    assert bool(torch.isfinite(temp_lse2).all().item())

    with pytest.raises(ValueError, match="return_temperature_lse"):
        sparse_atten_func(
            q, k, v, row_ptr, q_indices, 1, return_temperature_lse=True,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=2, max_seqlen_k=2,
            blk_kv=2, return_softmax_lse=False,
        )


def test_sm12x_topk_export_runs() -> None:
    _need_cuda()
    from fmha_sm12x import sparse_topk_select

    scores = torch.full((1, 8, 1), -1000.0, device="cuda", dtype=torch.float32)
    scores[0, 3, 0] = 1.0
    out = sparse_topk_select(scores.contiguous(), 16, num_valid_pages=8)
    assert out.shape == (1, 1, 16)
    selected = set(out[0, 0].to("cpu", non_blocking=False).tolist())
    assert 3 in selected


def _packed_fp4(shape: tuple[int, ...], nibble: int) -> torch.Tensor:
    return torch.full(shape, int(nibble) | (int(nibble) << 4), dtype=torch.uint8, device="cuda")


def test_sm12x_fp4_indexer_block_scores_runs() -> None:
    _need_cuda()
    from fmha_sm12x import fp4_indexer_block_scores

    q = _packed_fp4((1, 1, 64), 2)
    k = _packed_fp4((1, 1, 128, 64), 2)
    q_scale = torch.ones((1, 1, 8), device="cuda").to(torch.float8_e4m3fn)
    k_scale = torch.ones((1, 1, 128, 8), device="cuda").to(torch.float8_e4m3fn)
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 128], dtype=torch.int32, device="cuda")
    page_offsets = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    kv_indices = torch.tensor([0], dtype=torch.int32, device="cuda")

    scores = fp4_indexer_block_scores(
        q, k, q_scale, k_scale, cu_q, cu_k, page_offsets,
        max_seqlen_q=1, max_seqlen_k=128, kv_indices=kv_indices,
        fp4_format="nvfp4", scale_layout="public",
    )

    assert scores.shape == (1, 1, 1)
    torch.testing.assert_close(scores[0, 0, 0], torch.tensor(128.0, device="cuda"))


def test_sm12x_nvfp4_helper_exports_dequantize() -> None:
    _need_cuda()
    from fmha_sm12x import (
        Nvfp4QuantizedTensor,
        dequantize_nvfp4_128x4_to_bf16,
        nvfp4_global_scale_from_amax,
        swizzle_nvfp4_scale_to_128x4,
    )

    data = _packed_fp4((1, 1, 64), 2)
    scale = swizzle_nvfp4_scale_to_128x4(
        torch.ones((1, 8), device="cuda").to(torch.float8_e4m3fn),
        rows=1,
        cols=8,
    )
    global_scale = torch.ones((1,), device="cuda", dtype=torch.float32)
    quantized = Nvfp4QuantizedTensor(
        data=data,
        scale_128x4=scale,
        global_scale=global_scale,
        logical_scale_shape=(1, 8),
        original_shape=(1, 1, 128),
    )

    out = dequantize_nvfp4_128x4_to_bf16(quantized)

    torch.testing.assert_close(out, torch.ones_like(out), atol=0, rtol=0)
    expected_scale = torch.ones((1,), device="cuda", dtype=torch.float32)
    torch.testing.assert_close(nvfp4_global_scale_from_amax(torch.full_like(expected_scale, 2688.0)), expected_scale)


def test_sm12x_build_k2q_csr_uses_sm100_block_major_row_order() -> None:
    from fmha_sm12x import build_k2q_csr

    q2k = torch.tensor([[[1], [-1], [0], [0]]], dtype=torch.int32)
    cu_q = torch.tensor([0, 2, 4], dtype=torch.int32)
    cu_k = torch.tensor([0, 2, 4], dtype=torch.int32)

    row_ptr, q_indices = build_k2q_csr(q2k, cu_q, cu_k, 1)

    torch.testing.assert_close(row_ptr, torch.tensor([[0, 0, 2, 3, 3]], dtype=torch.int32))
    torch.testing.assert_close(q_indices[:, :3], torch.tensor([[0, 1, 0]], dtype=torch.int32))


def test_sm12x_optimized_k2q_builder_matches_reference() -> None:
    _need_sm12x_cuda()
    from fmha_sm12x import SparseK2qCsrBuilderSm12x, build_k2q_csr

    q2k = torch.tensor(
        [[[0, -1, -1, -1], [0, -1, -1, -1], [0, 1, -1, -1], [1, -1, -1, -1]]],
        device="cuda",
        dtype=torch.int32,
    )
    cu_q = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 128, 384], device="cuda", dtype=torch.int32)

    ref_row_ptr, ref_q_indices = build_k2q_csr(q2k, cu_q, cu_k, 128)
    row_ptr, q_indices = SparseK2qCsrBuilderSm12x()(q2k, cu_q, cu_k, total_k=384, blk_kv=128)
    torch.cuda.synchronize()

    torch.testing.assert_close(row_ptr, ref_row_ptr)
    nnz = int(row_ptr[0, -1].item())
    ref_nnz = int(ref_row_ptr[0, -1].item())
    torch.testing.assert_close(q_indices[:, :nnz], ref_q_indices[:, :ref_nnz])
    torch.testing.assert_close(q_indices[:, nnz:], torch.full_like(q_indices[:, nnz:], -1))


def test_sm12x_optimized_k2q_builder_handles_per_batch_partial_rows() -> None:
    _need_sm12x_cuda()
    from fmha_sm12x import SparseK2qCsrBuilderSm12x, build_k2q_csr

    q2k = torch.tensor(
        [[[0, -1, -1, -1], [0, -1, -1, -1]]],
        device="cuda",
        dtype=torch.int32,
    )
    cu_q = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 1, 128], device="cuda", dtype=torch.int32)

    ref_row_ptr, ref_q_indices = build_k2q_csr(q2k, cu_q, cu_k, 128)
    row_ptr, q_indices = SparseK2qCsrBuilderSm12x()(q2k, cu_q, cu_k, total_k=128, blk_kv=128)
    torch.cuda.synchronize()

    torch.testing.assert_close(row_ptr, ref_row_ptr)
    nnz = int(row_ptr[0, -1].item())
    torch.testing.assert_close(q_indices[:, :nnz], ref_q_indices[:, :nnz])


def test_sm12x_optimized_k2q_builder_rejects_bad_cu_seqlens() -> None:
    _need_sm12x_cuda()
    from fmha_sm12x import SparseK2qCsrBuilderSm12x

    q2k = torch.zeros((1, 2, 4), device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
    bad_cu_q = torch.tensor([0, 1, 1], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="cu_seqlens_q"):
        SparseK2qCsrBuilderSm12x()(q2k, bad_cu_q, cu_k, total_k=128, blk_kv=128)


def test_sm12x_optimized_k2q_builder_returns_schedule() -> None:
    _need_sm12x_cuda()
    from fmha_sm12x import SparseK2qCsrBuilderSm12x

    q2k = torch.tensor(
        [[[0, -1, -1, -1], [0, -1, -1, -1], [0, 1, -1, -1], [1, -1, -1, -1]]],
        device="cuda",
        dtype=torch.int32,
    )
    cu_q = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 128, 384], device="cuda", dtype=torch.int32)

    row_ptr, q_indices, schedule = SparseK2qCsrBuilderSm12x()(
        q2k,
        cu_q,
        cu_k,
        total_k=384,
        blk_kv=128,
        max_seqlen_q=2,
        max_seqlen_k=256,
        return_schedule=True,
    )
    torch.cuda.synchronize()

    assert schedule.enabled
    assert schedule.scheduler_metadata is not None
    assert schedule.scheduler_metadata.shape[1] == 6
    assert schedule.work_count is not None
    assert int(schedule.work_count.item()) >= 0
    assert schedule.qsplit_indices is not None
    assert schedule.qsplit_indices.shape == q_indices.shape
    assert schedule.split_counts is not None
    assert schedule.split_counts.shape == (4, 1)
    assert int(row_ptr[0, -1].item()) == 5


def test_sm12x_sparse_nvfp4_prefill_runs() -> None:
    _need_cuda()
    from fmha_sm12x import sparse_atten_nvfp4_kv_func

    q = torch.ones((1, 1, 128), dtype=torch.bfloat16, device="cuda")
    k = _packed_fp4((128, 1, 64), 2)
    v = _packed_fp4((128, 1, 64), 4)
    scale = torch.ones((128, 8), device="cuda").to(torch.float8_e4m3fn)
    row_ptr = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    q_indices = torch.tensor([[0]], dtype=torch.int32, device="cuda")
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 128], dtype=torch.int32, device="cuda")

    out, lse = sparse_atten_nvfp4_kv_func(
        q, k, v, scale, scale, None, None, row_ptr, q_indices, 1,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=1, max_seqlen_k=128,
        causal=False, return_softmax_lse=True,
    )

    expected_lse = torch.tensor(math.sqrt(128.0) + math.log(128.0), device="cuda", dtype=torch.float32)
    torch.testing.assert_close(out, torch.full_like(out, 2.0), atol=0, rtol=0)
    torch.testing.assert_close(lse, expected_lse.reshape(1, 1), atol=1e-5, rtol=1e-5)

    # Parity: the NVFP4 variant forwards temperature-LSE outputs (3-tuple).
    out3, lse3, temp_lse = sparse_atten_nvfp4_kv_func(
        q, k, v, scale, scale, None, None, row_ptr, q_indices, 1,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=1, max_seqlen_k=128,
        causal=False, return_softmax_lse=True, return_temperature_lse=True,
        lse_temperature_scale=1.0,
    )
    torch.testing.assert_close(out3, out, atol=0, rtol=0)
    torch.testing.assert_close(temp_lse, lse3, atol=1e-5, rtol=1e-5)


def test_sm12x_build_k2q_csr_reference_rejects_return_schedule() -> None:
    # The pure-Torch reference cannot emit the fused schedule; it must fail
    # clearly rather than return a 2-tuple a 3-tuple caller would mis-unpack.
    _need_cuda()
    from fmha_sm12x import build_k2q_csr

    q2k = torch.tensor([[[0, -1, -1, -1]]], device="cuda", dtype=torch.int32)
    cu_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="return_schedule"):
        build_k2q_csr(q2k, cu_q, cu_k, 128, return_schedule=True)


def test_sm12x_decode_schedule_helper_runs() -> None:
    _need_sm12x_cuda()
    from fmha_sm12x.cute.src.sm12x.decode_schedule import prepare_decode_schedule

    seqused_k = torch.tensor([128, 256], device="cuda", dtype=torch.int32)
    schedule = prepare_decode_schedule(
        seqused_k=seqused_k,
        page_size=128,
        seqlen_q=8,
        num_qo_heads=16,
        num_kv_heads=1,
        head_dim=128,
        max_seqlen_k=256,
        disable_split_kv=True,
    )
    torch.cuda.synchronize()

    assert schedule.request_indices.is_cuda
    assert schedule.request_indices.shape[0] == schedule.padded_work_count
    assert schedule.block_valid_mask.shape[0] == schedule.padded_work_count
    assert schedule.merge_indptr.ndim == 1
    assert schedule.o_indptr.ndim == 1
    assert int(schedule.merge_indptr[0].item()) == 0
    assert int(schedule.o_indptr[0].item()) == 0
    assert schedule.split_counts.shape == (2,)
    assert schedule.work_count > 0


def test_sm12x_decode_schedule_cuda_graph_padding_capacity() -> None:
    # Regression: when CUDA-graph capture pads work_count up to the captured
    # grid, padded_work_count can exceed the page-based allocation bound
    # (batch * num_q_tiles * max_pages_global).  The output index arrays must
    # be allocated to at least the graph pad or the wrapper's
    # narrow(0, 0, padded_work_count) reads past the allocation.  A small
    # max_seqlen_k with a large max_grid_size override forces that case.
    _need_sm12x_cuda()
    from fmha_sm12x.cute.src.sm12x.decode_schedule import prepare_decode_schedule

    seqused_k = torch.tensor([1152], device="cuda", dtype=torch.int32)
    grid_override = 4096  # >> batch * num_q_tiles * max_pages_global (= 9)
    schedule = prepare_decode_schedule(
        seqused_k=seqused_k,
        page_size=128,
        seqlen_q=8,
        num_qo_heads=16,
        num_kv_heads=1,
        head_dim=128,
        max_seqlen_k=1152,
        enable_cuda_graph=True,
        max_grid_size=grid_override,
    )
    torch.cuda.synchronize()

    assert schedule.split_kv
    # CUDA-graph pad = max_grid_size / num_kv_heads, far above the 9-tile
    # page bound; the narrow below would have raised before the fix.
    assert schedule.padded_work_count >= grid_override
    for arr in (
        schedule.request_indices,
        schedule.qo_tile_indices,
        schedule.kv_tile_indices,
        schedule.block_valid_mask,
    ):
        assert arr.shape[0] == schedule.padded_work_count
    # Entries past the real work_count must be zeroed padding (valid mask 0).
    assert int(schedule.block_valid_mask[schedule.padded_work_count - 1].item()) == 0


def test_sm12x_decode_schedule_rejects_too_small_max_seqlen_k() -> None:
    _need_sm12x_cuda()
    from fmha_sm12x.cute.src.sm12x.decode_schedule import prepare_decode_schedule

    seqused_k = torch.tensor([256], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="max_seqlen_k"):
        prepare_decode_schedule(
            seqused_k=seqused_k,
            page_size=128,
            seqlen_q=8,
            num_qo_heads=16,
            num_kv_heads=1,
            head_dim=128,
            max_seqlen_k=128,
        )


def test_sm12x_decode_schedule_raw_entrypoint_guards_hang() -> None:
    # Blocker: the raw launch wrapper must reject hang-inducing seqused_k
    # (seqused_k[b] < seqlen_q) on its own, so a direct caller that bypasses
    # prepare_decode_schedule cannot spin the kernel on an all-masked row.
    _need_sm12x_cuda()
    from fmha_sm12x.cute.src.sm12x.fwd_decode.build_decode_schedule import (
        build_decode_schedule,
    )

    seqused_k = torch.tensor([4], device="cuda", dtype=torch.int32)  # < seqlen_q
    with pytest.raises(ValueError, match="seqused_k"):
        build_decode_schedule(
            seqused_k,
            page_size=128,
            seqlen_q=8,
            num_qo_heads=16,
            num_kv_heads=1,
            head_dim=128,
            max_seqlen_k=128,
        )


def test_sm12x_decode_schedule_raw_entrypoint_guards_pad_overflow() -> None:
    # Blocker: the raw wrapper must reject seqused_k longer than max_seqlen_k,
    # since the work-tile arrays are sized from max_seqlen_k; otherwise the
    # kernel scatter / narrow(0, 0, padded_work_count) run out of bounds.
    _need_sm12x_cuda()
    from fmha_sm12x.cute.src.sm12x.fwd_decode.build_decode_schedule import (
        build_decode_schedule,
    )

    seqused_k = torch.tensor([256], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="max_seqlen_k"):
        build_decode_schedule(
            seqused_k,
            page_size=128,
            seqlen_q=8,
            num_qo_heads=16,
            num_kv_heads=1,
            head_dim=128,
            max_seqlen_k=128,
        )


def test_sm12x_decode_schedule_rejects_short_seqused_k_via_wrapper() -> None:
    # The high-level wrapper still surfaces the same guard (now enforced at the
    # raw boundary it funnels through).
    _need_sm12x_cuda()
    from fmha_sm12x.cute.src.sm12x.decode_schedule import prepare_decode_schedule

    seqused_k = torch.tensor([4], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="seqused_k"):
        prepare_decode_schedule(
            seqused_k=seqused_k,
            page_size=128,
            seqlen_q=8,
            num_qo_heads=16,
            num_kv_heads=1,
            head_dim=128,
            max_seqlen_k=128,
        )


def test_sm12x_sparse_decode_wrapper_runs() -> None:
    _need_cuda()
    from fmha_sm12x import SparseDecodePagedAttentionWrapper, sparse_decode_atten_func

    q = torch.ones((1, 1, 128), dtype=torch.bfloat16, device="cuda")
    k = torch.zeros((1, 1, 128, 128), dtype=torch.bfloat16, device="cuda")
    v = torch.zeros_like(k)
    v[0, 0, 0].fill_(2.0)
    v[0, 0, 1].fill_(4.0)
    page_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
    seqused = torch.tensor([2], dtype=torch.int32, device="cuda")

    direct, lse = sparse_decode_atten_func(
        q, k, v, page_table=page_table, seqused_k=seqused,
        seqlen_q=1, max_seqlen_k=2, blk_kv=128, causal=True,
        return_softmax_lse=True,
    )
    wrapper = SparseDecodePagedAttentionWrapper().plan(
        page_table=page_table, seqused_k=seqused, seqlen_q=1, max_seqlen_k=2,
        num_qo_heads=1, num_kv_heads=1, head_dim=128,
    )
    wrapped = wrapper.run(q, k, v)

    torch.testing.assert_close(direct, torch.full_like(direct, 3.0), atol=0, rtol=0)
    torch.testing.assert_close(lse, torch.full_like(lse, math.log(2.0)), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(wrapped, direct, atol=0, rtol=0)
