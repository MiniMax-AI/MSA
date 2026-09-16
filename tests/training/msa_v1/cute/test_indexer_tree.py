"""MSA v1 tree indexer correctness tests."""

import logging
import math
import time
import warnings

import pytest
import torch

from msa_v1 import attention, indexer_tree
from tests.training.msa_v1.cute.testing import (
    causal_indexer_reference,
    causal_indexer_reference_compiled,
)

logger = logging.getLogger(__name__)


def _suffix_q_positions(q_len: int, k_len: int) -> torch.Tensor:
    return torch.arange(q_len, dtype=torch.int32, device="cuda") + k_len - q_len


def _causal_func(q_len: int, k_len: int) -> torch.Tensor:
    func = torch.full((1, 1, 1, q_len + 256), k_len, dtype=torch.int32, device="cuda")
    func[0, 0, 0, :q_len] = _suffix_q_positions(q_len, k_len) + 1
    return func


def _interval_func(
    q_positions: torch.Tensor,
    k_len: int,
    block_bases: torch.Tensor,
) -> torch.Tensor:
    q_len = q_positions.numel()
    func = torch.full(
        (1, 1, 3, q_len + 256),
        k_len,
        dtype=torch.int32,
        device="cuda",
    )
    func[0, 0, 0, :q_len] = 0
    func[0, 0, 1, :q_len] = block_bases * 128
    func[0, 0, 2, :q_len] = q_positions + 1
    return func


def _forward_with_timing(*args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.time()
    result = indexer_tree.forward(*args, **kwargs)
    torch.cuda.synchronize()
    logger.info("Ran Tree indexer in %.3fms", (time.time() - t0) * 1e3)
    return result


@pytest.mark.gpu
def test_tree_indexer_defaults_to_suffix_local_block_positions() -> None:
    q_len = 33
    k_len = 512

    plan = indexer_tree.compile_plan(_causal_func(q_len, k_len), q_len, k_len)

    torch.testing.assert_close(
        plan.local_block_positions,
        _suffix_q_positions(q_len, k_len).unsqueeze(0),
        rtol=0,
        atol=0,
    )


@pytest.mark.gpu
def test_tree_indexer_uses_explicit_non_suffix_local_block() -> None:
    q_len = 1
    k_len = 2304
    q_position = 2047
    func = torch.full((1, 1, 1, q_len + 256), k_len, dtype=torch.int32, device="cuda")
    func[0, 0, 0, 0] = q_position + 1
    plan = indexer_tree.compile_plan(
        func,
        q_len,
        k_len,
        torch.tensor((q_position,), dtype=torch.int32, device="cuda"),
    )
    q = torch.zeros(q_len, 4, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(k_len, 1, 128, dtype=torch.bfloat16, device="cuda")
    q[:, :, 0] = 1
    k[:, 0, 0] = (torch.arange(k_len, dtype=torch.int32, device="cuda") // 128 + 1).to(
        torch.bfloat16
    )

    ids, selected_lse = indexer_tree.forward(q, k, plan)

    expected_ids = torch.arange(16, dtype=torch.int32, device="cuda").view(1, 1, 16)
    torch.testing.assert_close(
        torch.sort(ids, dim=-1).values,
        expected_ids.expand(4, -1, -1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        selected_lse,
        torch.full_like(
            selected_lse,
            torch.logsumexp(
                torch.arange(1, 17, dtype=torch.float32, device="cuda")
                / math.sqrt(128),
                dim=0,
            )
            + math.log(128),
        ),
        rtol=2e-3,
        atol=2e-3,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("use_fp16_score", (False, True))
def test_tree_indexer_padding_matches_eager_reference(
    use_fp16_score: bool,
) -> None:
    torch.manual_seed(29)
    q_len = 33
    k_len = 512
    func = torch.full((1, 1, 1, q_len + 256), k_len, dtype=torch.int32, device="cuda")
    func[0, 0, 0, :q_len] = torch.arange(q_len, dtype=torch.int32, device="cuda") + (
        k_len - q_len + 1
    )
    plan = indexer_tree.compile_plan(func, q_len, k_len)
    q_index = torch.randn(q_len, 4, 128, dtype=torch.bfloat16, device="cuda")
    k_index = torch.randn(k_len, 1, 128, dtype=torch.bfloat16, device="cuda")

    score_workspace = torch.empty(
        (plan.num_plan_tiles, 2, 128),
        dtype=torch.float16 if use_fp16_score else torch.float32,
        device="cuda",
    )
    block_sum_workspace = torch.empty_like(score_workspace, dtype=torch.float32)
    ids, selected_lse = indexer_tree.forward(
        q_index,
        k_index,
        plan,
        score_workspace=score_workspace,
        block_sum_workspace=block_sum_workspace,
        use_fp16_score=use_fp16_score,
    )
    ids_ref, lse_ref = causal_indexer_reference(
        q_index,
        k_index,
        use_fp16_score=use_fp16_score,
    )
    torch.cuda.synchronize()

    assert torch.any(ids == -1)
    torch.testing.assert_close(
        torch.sort(ids, dim=-1).values,
        torch.sort(ids_ref, dim=-1).values,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(selected_lse, lse_ref, atol=2e-3, rtol=2e-3)
    for head in range(4):
        for q_idx in range(q_len):
            valid = ids[head, q_idx][ids[head, q_idx] >= 0]
            local_block = (k_len - q_len + q_idx) // 128
            assert int(valid[-1]) == local_block


@pytest.mark.gpu
@pytest.mark.parametrize("q_len,k_len", ((33, 512), (65, 2304)))
@pytest.mark.parametrize("use_fp16_score", (False, True))
def test_tree_indexer_deterministic_is_bitwise_stable(
    q_len: int,
    k_len: int,
    use_fp16_score: bool,
) -> None:
    torch.manual_seed(31)
    func = torch.full(
        (1, 1, 1, q_len + 256),
        k_len,
        dtype=torch.int32,
        device="cuda",
    )
    func[0, 0, 0, :q_len] = torch.arange(
        q_len,
        dtype=torch.int32,
        device="cuda",
    ) + (k_len - q_len + 1)
    plan = indexer_tree.compile_plan(func, q_len, k_len)
    q = torch.randn(q_len, 4, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(k_len, 1, 128, dtype=torch.bfloat16, device="cuda")

    ids, lse = indexer_tree.forward(
        q,
        k,
        plan,
        deterministic=True,
        use_fp16_score=use_fp16_score,
    )
    ids = ids.clone()
    lse = lse.clone()
    ids_repeat, lse_repeat = indexer_tree.forward(
        q,
        k,
        plan,
        deterministic=True,
        use_fp16_score=use_fp16_score,
    )

    assert torch.equal(ids_repeat, ids)
    assert torch.equal(lse_repeat.view(torch.int32), lse.view(torch.int32))
    for head in range(4):
        for q_idx in range(q_len):
            valid = ids[head, q_idx][ids[head, q_idx] >= 0]
            local_block = (k_len - q_len + q_idx) // 128
            assert int(valid[-1]) == local_block


@pytest.mark.gpu
def test_tree_fp16_score_ties_select_lowest_nonlocal_ids() -> None:
    q_len = 1
    k_len = 2304
    func = torch.full(
        (1, 1, 1, q_len + 256), k_len, dtype=torch.int32, device="cuda"
    )
    plan = indexer_tree.compile_plan(func, q_len, k_len)
    q = torch.zeros(q_len, 4, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(k_len, 1, 128, dtype=torch.bfloat16, device="cuda")

    ids, selected_lse = indexer_tree.forward(
        q,
        k,
        plan,
        use_fp16_score=True,
    )
    ids_ref, lse_ref = causal_indexer_reference(
        q,
        k,
        use_fp16_score=True,
    )
    for head in range(4):
        assert set(ids[head, 0].cpu().tolist()) == set(ids_ref[head, 0].cpu().tolist())
        assert int(ids[head, 0, -1]) == 17
    torch.testing.assert_close(selected_lse, lse_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.gpu
def test_tree_indexer_rejects_non_boolean_deterministic() -> None:
    q_len = 17
    k_len = 257
    func = torch.full(
        (1, 1, 1, q_len + 256),
        k_len,
        dtype=torch.int32,
        device="cuda",
    )
    func[0, 0, 0, :q_len] = torch.arange(
        q_len,
        dtype=torch.int32,
        device="cuda",
    ) + (k_len - q_len + 1)
    plan = indexer_tree.compile_plan(func, q_len, k_len)
    q = torch.randn(q_len, 4, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(k_len, 1, 128, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(TypeError, match="deterministic"):
        indexer_tree.forward(q, k, plan, deterministic=1)
    with pytest.raises(TypeError, match="use_fp16_score"):
        indexer_tree.forward(q, k, plan, use_fp16_score=1)
    fp32_score = torch.empty(
        (plan.num_plan_tiles, 2, 128), dtype=torch.float32, device="cuda"
    )
    with pytest.raises(TypeError, match="torch.float16"):
        indexer_tree.forward(
            q,
            k,
            plan,
            score_workspace=fp32_score,
            use_fp16_score=True,
        )

    invalid_block_bases = (
        (torch.zeros(q_len, dtype=torch.int64, device="cuda"), TypeError, "int32"),
        (torch.zeros(q_len, dtype=torch.int32), ValueError, "CUDA device"),
        (
            torch.zeros(q_len + 1, dtype=torch.int32, device="cuda"),
            ValueError,
            "shape",
        ),
        (
            torch.zeros(q_len * 2, dtype=torch.int32, device="cuda")[::2],
            ValueError,
            "contiguous",
        ),
    )
    for block_bases, error_type, match in invalid_block_bases:
        with pytest.raises(error_type, match=match):
            indexer_tree.forward(q, k, plan, block_bases=block_bases)


@pytest.mark.gpu
@pytest.mark.parametrize("use_fp16_score", (False, True))
def test_tree_indexer_rebases_direct_output_in_preallocated_storage(
    use_fp16_score: bool,
) -> None:
    from msa_v1.indexer_tree import m3_indexer as tree_indexer_module

    q_len = 1
    k_len = 2304
    q_positions = torch.tensor((2047,), dtype=torch.int32, device="cuda")
    block_bases = torch.full((q_len,), 4, dtype=torch.int32, device="cuda")
    plan = indexer_tree.compile_plan(
        _interval_func(q_positions, k_len, block_bases),
        q_len,
        k_len,
        q_positions,
    )
    assert plan.max_plan_tiles <= 16
    q = torch.randn(q_len, 4, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(k_len, 1, 128, dtype=torch.bfloat16, device="cuda")

    absolute_ids, absolute_lse = _forward_with_timing(
        q,
        k,
        plan,
        deterministic=True,
        use_fp16_score=use_fp16_score,
    )
    output = torch.empty((4, q_len, 16), dtype=torch.int32, device="cuda")
    selected_lse = torch.empty((4, q_len), dtype=torch.float32, device="cuda")
    rebased_ids, rebased_lse = _forward_with_timing(
        q,
        k,
        plan,
        block_bases=block_bases,
        topk_indices=output,
        selected_lse=selected_lse,
        deterministic=True,
        use_fp16_score=use_fp16_score,
    )

    assert rebased_ids.data_ptr() == output.data_ptr()
    assert rebased_lse.data_ptr() == selected_lse.data_ptr()
    expected_ids = torch.where(
        absolute_ids >= 0,
        absolute_ids - block_bases.view(1, q_len, 1),
        absolute_ids,
    )
    assert torch.equal(rebased_ids, expected_ids)
    assert torch.equal(rebased_lse.view(torch.int32), absolute_lse.view(torch.int32))
    assert torch.any(rebased_ids == -1)
    assert torch.all((rebased_ids >= 0) | (rebased_ids == -1))
    for head in range(4):
        valid = rebased_ids[head, 0][rebased_ids[head, 0] >= 0]
        assert int(valid[-1]) == q_positions.item() // 128 - block_bases.item()

    compile_keys = set(tree_indexer_module._TOPK_COMPILE_CACHE)
    alternate_bases = block_bases - 1
    alternate_ids, _ = _forward_with_timing(
        q,
        k,
        plan,
        block_bases=alternate_bases,
        topk_indices=output,
        selected_lse=selected_lse,
        deterministic=True,
        use_fp16_score=use_fp16_score,
    )
    assert set(tree_indexer_module._TOPK_COMPILE_CACHE) == compile_keys
    alternate_expected = torch.where(
        absolute_ids >= 0,
        absolute_ids - alternate_bases.view(1, q_len, 1),
        absolute_ids,
    )
    assert torch.equal(alternate_ids, alternate_expected)
    matching_keys = {
        key[-1]
        for key in compile_keys
        if key[0] == "m3_arbitrary_indexer_topk_sm100"
        and key[2:6] == (True, True, False, use_fp16_score)
    }
    assert matching_keys == {False, True}


@pytest.mark.gpu
@pytest.mark.parametrize("use_fp16_score", (False, True))
@pytest.mark.parametrize("deterministic", (False, True))
def test_tree_indexer_rebases_gather_output(
    deterministic: bool,
    use_fp16_score: bool,
) -> None:
    from msa_v1.indexer_tree import m3_indexer as tree_indexer_module

    q_len = 17
    k_len = 2816
    q_positions = _suffix_q_positions(q_len, k_len)
    block_bases = 4 + torch.arange(q_len, dtype=torch.int32, device="cuda") % 2
    plan = indexer_tree.compile_plan(
        _interval_func(q_positions, k_len, block_bases),
        q_len,
        k_len,
        q_positions,
    )
    assert tree_indexer_module._use_gather_lse(
        plan,
        deterministic=deterministic,
        capability=torch.cuda.get_device_capability(),
    )
    q = torch.randn(q_len, 4, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(k_len, 1, 128, dtype=torch.bfloat16, device="cuda")

    absolute_ids, absolute_lse = _forward_with_timing(
        q,
        k,
        plan,
        deterministic=deterministic,
        use_fp16_score=use_fp16_score,
    )
    output = torch.empty((4, q_len, 16), dtype=torch.int32, device="cuda")
    selected_lse = torch.empty((4, q_len), dtype=torch.float32, device="cuda")
    rebased_ids, rebased_lse = _forward_with_timing(
        q,
        k,
        plan,
        block_bases=block_bases,
        topk_indices=output,
        selected_lse=selected_lse,
        deterministic=deterministic,
        use_fp16_score=use_fp16_score,
    )

    assert rebased_ids.data_ptr() == output.data_ptr()
    assert rebased_lse.data_ptr() == selected_lse.data_ptr()
    expected_ids = absolute_ids - block_bases.view(1, q_len, 1)
    if deterministic:
        assert torch.equal(rebased_ids, expected_ids)
    else:
        assert torch.equal(
            torch.sort(rebased_ids, dim=-1).values,
            torch.sort(expected_ids, dim=-1).values,
        )
    assert torch.equal(rebased_lse.view(torch.int32), absolute_lse.view(torch.int32))
    assert torch.all(rebased_ids >= 0)
    for head in range(4):
        for q_idx in range(q_len):
            expected_local = (q_positions[q_idx] // 128 - block_bases[q_idx]).item()
            assert int(rebased_ids[head, q_idx, -1]) == expected_local

    topk_compile_keys = set(tree_indexer_module._TOPK_COMPILE_CACHE)
    lse_compile_keys = set(tree_indexer_module._LSE_COMPILE_CACHE)
    alternate_bases = block_bases - 1
    alternate_ids, _ = _forward_with_timing(
        q,
        k,
        plan,
        block_bases=alternate_bases,
        topk_indices=output,
        selected_lse=selected_lse,
        deterministic=deterministic,
        use_fp16_score=use_fp16_score,
    )
    assert set(tree_indexer_module._TOPK_COMPILE_CACHE) == topk_compile_keys
    assert set(tree_indexer_module._LSE_COMPILE_CACHE) == lse_compile_keys
    alternate_expected = absolute_ids - alternate_bases.view(1, q_len, 1)
    if deterministic:
        assert torch.equal(alternate_ids, alternate_expected)
    else:
        assert torch.equal(
            torch.sort(alternate_ids, dim=-1).values,
            torch.sort(alternate_expected, dim=-1).values,
        )


@pytest.mark.gpu
@pytest.mark.parametrize("n_func", (1, 3))
def test_tree_indexer_plan_rejects_invalid_intervals(n_func: int) -> None:
    q_len = 65
    k_len = 512
    func = torch.full(
        (1, 1, n_func, q_len + 256),
        k_len,
        dtype=torch.int32,
        device="cuda",
    )
    if n_func == 1:
        func[0, 0, 0, 7] = k_len + 1
    else:
        func[0, 0, 0, :q_len] = 128
        func[0, 0, 1, :q_len] = 256
        func[0, 0, 2, :q_len] = 384
        func[0, 0, 2, 11] = 255

    with pytest.raises(ValueError, match="arbitrary_func intervals"):
        indexer_tree.compile_plan(func, q_len, k_len)


@pytest.mark.gpu
def test_tree_indexer_leading_empty_interval_matches_prefix_plan() -> None:
    torch.manual_seed(37)
    q_len = 4096
    k_len = q_len
    prefix_func = _causal_func(q_len, k_len)
    interval_func = torch.zeros(
        (1, 1, 3, q_len + 256),
        dtype=torch.int32,
        device="cuda",
    )
    interval_func[0, 0, 2, :q_len] = torch.arange(
        1,
        q_len + 1,
        dtype=torch.int32,
        device="cuda",
    )
    prefix_plan = indexer_tree.compile_plan(prefix_func, q_len, k_len)
    interval_plan = indexer_tree.compile_plan(interval_func, q_len, k_len)
    q = torch.randn(q_len, 4, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(k_len, 1, 128, dtype=torch.bfloat16, device="cuda")
    prefix_ids, prefix_lse = indexer_tree.forward(q, k, prefix_plan)
    interval_ids, interval_lse = indexer_tree.forward(q, k, interval_plan)

    torch.testing.assert_close(
        torch.sort(interval_ids, dim=-1).values,
        torch.sort(prefix_ids, dim=-1).values,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(interval_lse, prefix_lse, rtol=2e-3, atol=2e-3)


@pytest.mark.gpu
@pytest.mark.parametrize("use_fp16_score", (False, True))
def test_tree_indexer_gather_lse_matches_causal_reference_for_odd_tile_counts(
    use_fp16_score: bool,
) -> None:
    from msa_v1.indexer_tree import m3_indexer as tree_indexer_module

    torch.manual_seed(43)
    q_len = 2305
    k_len = q_len
    func = torch.zeros(
        (1, 1, 3, q_len + 256),
        dtype=torch.int32,
        device="cuda",
    )
    func[0, 0, 2, :q_len] = torch.arange(
        1,
        q_len + 1,
        dtype=torch.int32,
        device="cuda",
    )
    plan = indexer_tree.compile_plan(func, q_len, k_len)
    assert tree_indexer_module._use_gather_lse(
        plan,
        deterministic=False,
        capability=torch.cuda.get_device_capability(),
    )
    q = torch.randn(q_len, 4, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(k_len, 1, 128, dtype=torch.bfloat16, device="cuda")
    ids, selected_lse = indexer_tree.forward(
        q,
        k,
        plan,
        use_fp16_score=use_fp16_score,
    )
    ids_ref, lse_ref = causal_indexer_reference(
        q,
        k,
        use_fp16_score=use_fp16_score,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        torch.sort(ids, dim=-1).values,
        torch.sort(ids_ref, dim=-1).values,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(selected_lse, lse_ref, rtol=2e-3, atol=2e-3)


@pytest.mark.gpu
@pytest.mark.parametrize("invalid", (False, True), ids=("valid", "invalid-interval"))
def test_tree_indexer_plan_long_query_fallback(invalid: bool) -> None:
    q_len = 16_385
    k_len = q_len
    func = _causal_func(q_len, k_len)
    if invalid:
        func[0, 0, 0, q_len - 1] = k_len + 1

        with pytest.raises(ValueError, match="arbitrary_func intervals"):
            indexer_tree.compile_plan(func, q_len, k_len)
        return

    plan = indexer_tree.compile_plan(func, q_len, k_len)
    assert plan.q_len == q_len
    assert plan.k_len == k_len
    assert plan.max_plan_tiles > 0


@pytest.mark.gpu
def test_non_deterministic_causal_indexer_and_attention_integration() -> None:
    torch.manual_seed(23)
    q_len = 65
    k_len = 2304
    func = torch.full((1, 1, 1, q_len + 256), k_len, dtype=torch.int32, device="cuda")
    func[0, 0, 0, :q_len] = torch.arange(q_len, dtype=torch.int32, device="cuda") + (
        k_len - q_len + 1
    )
    plan = indexer_tree.compile_plan(func, q_len, k_len)
    q_index = torch.randn(q_len, 4, 128, dtype=torch.bfloat16, device="cuda")
    k_index = torch.randn(k_len, 1, 128, dtype=torch.bfloat16, device="cuda")
    ids, selected_lse = indexer_tree.forward(q_index, k_index, plan)
    ids_ref, lse_ref = causal_indexer_reference(q_index, k_index)
    compiled_ids_ref, compiled_lse_ref = causal_indexer_reference_compiled(
        q_index,
        k_index,
    )
    if not torch.equal(
        torch.sort(compiled_ids_ref, dim=-1).values,
        torch.sort(ids_ref, dim=-1).values,
    ):
        warnings.warn("compiled Tree indexer reference ids differ from eager reference")
    if not torch.allclose(compiled_lse_ref, lse_ref, atol=2e-3, rtol=2e-3):
        warnings.warn(
            "compiled Tree indexer reference LSE differs from eager reference"
        )

    torch.cuda.synchronize()
    assert ids.shape == (4, q_len, 16)
    assert ids.dtype == torch.int32 and ids.is_cuda
    assert selected_lse.shape == (4, q_len)
    assert selected_lse.dtype == torch.float32 and selected_lse.is_cuda
    assert torch.all((ids >= 0) | (ids == -1))
    for head in range(4):
        for q_idx in range(q_len):
            valid = ids[head, q_idx][ids[head, q_idx] >= 0]
            valid_ref = ids_ref[head, q_idx][ids_ref[head, q_idx] >= 0]
            assert set(valid.cpu().tolist()) == set(valid_ref.cpu().tolist())
            local_block = (k_len - q_len + q_idx) // 128
            assert int(valid[-1]) == local_block
    torch.testing.assert_close(selected_lse, lse_ref, atol=2e-3, rtol=2e-3)

    cu_q = torch.tensor((0, q_len), dtype=torch.int32, device="cuda")
    cu_k = torch.tensor((0, k_len), dtype=torch.int32, device="cuda")
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
        q_len, 64, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    k = torch.randn(
        k_len, 4, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    v = torch.randn_like(k, requires_grad=True)
    out = attention.forward(q, k, v, metadata)
    grads = torch.autograd.grad(out, (q, k, v), torch.randn_like(out))
    assert all(torch.isfinite(tensor).all() for tensor in (out, *grads))


@pytest.mark.gpu
@pytest.mark.parametrize("q_len,k_len", ((33, 512), (17, 2304)))
@pytest.mark.parametrize("lse_temperature", (0.5, 2.0))
def test_tree_indexer_lse_temperature(
    q_len: int,
    k_len: int,
    lse_temperature: float,
) -> None:
    torch.manual_seed(41)
    func = torch.full((1, 1, 1, q_len + 256), k_len, dtype=torch.int32, device="cuda")
    func[0, 0, 0, :q_len] = torch.arange(q_len, dtype=torch.int32, device="cuda") + (
        k_len - q_len + 1
    )
    plan = indexer_tree.compile_plan(func, q_len, k_len)
    q = torch.randn(q_len, 4, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(k_len, 1, 128, dtype=torch.bfloat16, device="cuda")

    ids_default, lse_default = indexer_tree.forward(q, k, plan)
    ids, selected_lse = indexer_tree.forward(
        q,
        k,
        plan,
        lse_temperature=lse_temperature,
    )
    ids_ref, lse_ref = causal_indexer_reference(
        q,
        k,
        lse_temperature=lse_temperature,
    )
    assert torch.equal(ids_default, ids)
    assert not torch.equal(lse_default, selected_lse)
    torch.testing.assert_close(
        torch.sort(ids, dim=-1).values,
        torch.sort(ids_ref, dim=-1).values,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(selected_lse, lse_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.gpu
def test_tree_indexer_unit_and_invalid_temperature() -> None:
    q_len = 17
    k_len = 257
    func = torch.full((1, 1, 1, q_len + 256), k_len, dtype=torch.int32, device="cuda")
    func[0, 0, 0, :q_len] = torch.arange(q_len, dtype=torch.int32, device="cuda") + (
        k_len - q_len + 1
    )
    plan = indexer_tree.compile_plan(func, q_len, k_len)
    q = torch.randn(q_len, 4, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(k_len, 1, 128, dtype=torch.bfloat16, device="cuda")

    ids_default, lse_default = indexer_tree.forward(q, k, plan)
    ids_one, lse_one = indexer_tree.forward(q, k, plan, lse_temperature=1.0)
    assert torch.equal(ids_default, ids_one)
    assert torch.equal(lse_default, lse_one)
    for value in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="lse_temperature"):
            indexer_tree.forward(q, k, plan, lse_temperature=value)
    for value in (True, torch.tensor(1.0, device="cuda")):
        with pytest.raises(TypeError, match="lse_temperature"):
            indexer_tree.forward(q, k, plan, lse_temperature=value)
