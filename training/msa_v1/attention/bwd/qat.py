import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from msa_v1._common import utils


@cute.jit
def load_row_quant_scale(
    tScS: cute.Tensor,
    mPQuantScale: cute.Tensor,
    mK2qQSplitIndices: cute.Tensor,
    head_idx: Int32,
    row_start: Int32,
    iter_idx: Int32,
    k2q_count: Int32,
    q_offset: Int32,
    scale_component: cutlass.Constexpr[int],
    q_per_tile: cutlass.Constexpr[int],
    qhead_per_kvhead: cutlass.Constexpr[int],
) -> Float32:
    """Load one helper-produced quantization scale for one lane's Q row."""
    lane_idx = cute.arch.lane_idx()
    rows_per_stage = cute.size(tScS, mode=[0])
    coord_lane = cutlass.min(lane_idx, Int32(rows_per_stage - 1))
    row_idx = cute.get(tScS[coord_lane], mode=[1])
    token_idx = row_idx // Int32(qhead_per_kvhead)
    edge_idx = iter_idx * Int32(q_per_tile) + token_idx
    scale_value = Float32(0.0)
    if lane_idx < Int32(rows_per_stage) and edge_idx < k2q_count:
        qsplit = mK2qQSplitIndices[head_idx, row_start + edge_idx]
        q_idx = qsplit & Int32(0x00FF_FFFF)
        split_idx = (qsplit >> Int32(24)) & Int32(0xFF)
        q_abs = q_offset + q_idx
        head_in_kv = row_idx - token_idx * Int32(qhead_per_kvhead)
        head_abs = head_idx * Int32(qhead_per_kvhead) + head_in_kv
        scale_value = mPQuantScale[
            scale_component, split_idx, q_abs, head_abs
        ]
    return scale_value


@cute.jit
def scale_apply_exp2_fake_quant_e4m3(
    tSrS: cute.Tensor,
    tSrPQuant: cute.Tensor,
    row_max_scaled_lane: Float32,
    row_scale_lane: Float32,
    scale_log2: Float32,
) -> None:
    """Reconstruct logical and RTN probabilities for attention-level STE.

    The stored max is the negative P448 exponent bias; the stored row scale
    compensates the 448 factor when normalizing both probabilities.
    ``tSrS`` is updated in place with the unquantized global probability and
    ``tSrPQuant`` receives its E4M3-RTN forward value. Keeping the two values
    in the existing score/P fragments avoids another exp2 or a workspace.
    """
    assert tSrS.element_type is Float32, "tSrS must be Float32"
    assert cute.size(tSrPQuant) == cute.size(tSrS), (
        "tSrPQuant and tSrS must have the same size"
    )
    frg_tile = 4
    assert cute.size(tSrS) % frg_tile == 0, "tSrS size must be divisible by 4"
    tSrS_frg = cute.logical_divide(tSrS, cute.make_layout(frg_tile))
    tSrPQuant_frg = cute.logical_divide(
        tSrPQuant, cute.make_layout(frg_tile)
    )
    tRowScale = cute.make_rmem_tensor(frg_tile, Float32)
    for j in cutlass.range_constexpr(cute.size(tSrS_frg, mode=[1])):
        for k in cutlass.range_constexpr(0, frg_tile, 2):
            idx0 = j * frg_tile + k
            idx1 = idx0 + 1
            row_max_scaled0 = utils.shuffle_sync(
                row_max_scaled_lane, offset=idx0
            )
            row_max_scaled1 = utils.shuffle_sync(
                row_max_scaled_lane, offset=idx1
            )
            tRowScale[k] = utils.shuffle_sync(row_scale_lane, offset=idx0)
            tRowScale[k + 1] = utils.shuffle_sync(
                row_scale_lane, offset=idx1
            )
            tSrS[idx0], tSrS[idx1] = cute.arch.fma_packed_f32x2(
                (tSrS[idx0], tSrS[idx1]),
                (scale_log2, scale_log2),
                (-row_max_scaled0, -row_max_scaled1),
            )
            tSrS[idx0] = cute.math.exp2(tSrS[idx0], fastmath=True)
            tSrS[idx1] = cute.math.exp2(tSrS[idx1], fastmath=True)
        tPQuant = cute.make_rmem_tensor(frg_tile, Float32)
        tPQuant.store(
            tSrS_frg[None, j]
            .load()
            .to(cutlass.Float8E4M3FN)
            .to(Float32)
        )
        for k in cutlass.range_constexpr(0, frg_tile, 2):
            tPQuant[k], tPQuant[k + 1] = cute.arch.mul_packed_f32x2(
                (tPQuant[k], tPQuant[k + 1]),
                (tRowScale[k], tRowScale[k + 1]),
            )
            tSrS_frg[k, j], tSrS_frg[k + 1, j] = cute.arch.mul_packed_f32x2(
                (tSrS_frg[k, j], tSrS_frg[k + 1, j]),
                (tRowScale[k], tRowScale[k + 1]),
            )
        tSrPQuant_frg[None, j].store(
            tPQuant.load().to(tSrPQuant.element_type)
        )


@cute.jit
def reconstruct_attention_ste_probabilities(
    tScS: cute.Tensor,
    tSrS: cute.Tensor,
    tSrPQuant: cute.Tensor,
    tSsPStats: cute.Tensor,
    mPQuantScale: cute.Tensor,
    mK2qQSplitIndices: cute.Tensor,
    head_idx: Int32,
    row_start: Int32,
    iter_idx: Int32,
    k2q_count: Int32,
    q_offset: Int32,
    softmax_scale_log2: Float32,
    stage_p_stats: cutlass.Constexpr[bool],
    q_per_tile: cutlass.Constexpr[int],
    qhead_per_kvhead: cutlass.Constexpr[int],
) -> None:
    """Reconstruct split-local logical and E4M3-RTN probabilities."""
    if cutlass.const_expr(stage_p_stats):
        row_max_scaled_lane = Float32(0.0)
        row_scale_lane = Float32(0.0)
        lane_idx = cute.arch.lane_idx()
        rows_per_stage = cute.size(tScS, mode=[0])
        row_owner = cutlass.min(lane_idx, Int32(rows_per_stage - 1))
        if lane_idx < Int32(rows_per_stage):
            row_max_scaled_lane = tSsPStats[row_owner]
            row_scale_lane = cute.make_tensor(
                tSsPStats.iterator + q_per_tile * qhead_per_kvhead,
                tSsPStats.layout,
            )[row_owner]
    else:
        row_scale_lane = load_row_quant_scale(
            tScS,
            mPQuantScale,
            mK2qQSplitIndices,
            head_idx,
            row_start,
            iter_idx,
            k2q_count,
            q_offset,
            1,
            q_per_tile,
            qhead_per_kvhead,
        )
        row_max_scaled_lane = load_row_quant_scale(
            tScS,
            mPQuantScale,
            mK2qQSplitIndices,
            head_idx,
            row_start,
            iter_idx,
            k2q_count,
            q_offset,
            0,
            q_per_tile,
            qhead_per_kvhead,
        )
    scale_apply_exp2_fake_quant_e4m3(
        tSrS,
        tSrPQuant,
        row_max_scaled_lane,
        row_scale_lane,
        softmax_scale_log2,
    )


class SparseAttentionQatBackwardMixin:
    """Device helpers used only by the BF16 probability-QAT backward path."""

    @cute.jit
    def _load_sparse_p_stats_vec(
        self,
        mPQuantScale: cute.Tensor,
        mK2qQSplitIndices: cute.Tensor,
        sPStats: cute.Tensor,
        head_idx: Int32,
        row_start: Int32,
        iter_idx: Int32,
        k2q_count: Int32,
        q_offset: Int32,
        load_warp_lane: Int32,
        async_stats_atom: cute.CopyAtom,
    ) -> None:
        """Load row max and row scale through the stats pipeline."""
        assert self.qhead_per_kvhead == 16
        heads_per_lane = 4
        lanes_per_token = self.qhead_per_kvhead // heads_per_lane
        token_idx = load_warp_lane // Int32(lanes_per_token)
        head_chunk = load_warp_lane % Int32(lanes_per_token)
        edge_idx = iter_idx * Int32(self.q_per_tile) + token_idx
        edge_idx = cutlass.min(edge_idx, k2q_count - Int32(1))
        qsplit = mK2qQSplitIndices[head_idx, row_start + edge_idx]
        q_idx = qsplit & Int32(0x00FF_FFFF)
        split_idx = (qsplit >> Int32(24)) & Int32(0xFF)
        q_abs = q_offset + q_idx
        head_abs = (
            head_idx * Int32(self.qhead_per_kvhead)
            + head_chunk * Int32(heads_per_lane)
        )
        dst_row = (
            token_idx * Int32(self.qhead_per_kvhead)
            + head_chunk * Int32(heads_per_lane)
        )
        vec_layout = cute.make_layout(heads_per_lane)
        for component in cutlass.range_constexpr(2):
            gPStats = mPQuantScale[component, split_idx, q_abs, None]
            gPStats_ptr = cute.make_ptr(
                Float32,
                gPStats.iterator.toint() + head_abs * Int32(4),
                mem_space=gPStats.iterator.memspace,
                assumed_align=16,
            )
            sPStats_ptr = cute.make_ptr(
                Float32,
                sPStats.iterator.toint()
                + (Int32(component * self.tile_m) + dst_row) * Int32(4),
                mem_space=sPStats.iterator.memspace,
                assumed_align=16,
            )
            gPStats_vec = cute.make_tensor(gPStats_ptr, vec_layout)
            sPStats_vec = cute.make_tensor(sPStats_ptr, vec_layout)
            cute.copy(async_stats_atom, gPStats_vec, sPStats_vec)
