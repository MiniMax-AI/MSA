"""Fused logical-page scheduling, following DeepGEMM's paged metadata design."""

import cutlass
import cutlass.cute as cute  # noqa: PLR0402

# isort: split
import cuda.bindings.driver as cuda

# Original tokens covered by one logical block and one output score column.
_BLOCK_TOKENS = 128


class DecodeIndexerPlanSm100:
    """Generate a complete immutable-for-consumers plan in one CTA."""

    def __init__(
        self,
        *,
        pooling_factor=1,
        pages_per_block=1,
        blocks_per_tile=1,
        include_partial_block=False,
    ):
        self.pooling_factor = pooling_factor
        self.entries_per_block = _BLOCK_TOKENS // pooling_factor
        self.pages_per_block = pages_per_block
        self.blocks_per_tile = blocks_per_tile
        self.include_partial_block = include_partial_block

    @cute.jit
    def __call__(
        self,
        mLengths: cute.Tensor,
        mSnapshot: cute.Tensor,
        mScheduler: cute.Tensor,
        mValidPages: cute.Tensor,
        query_length: cutlass.Int32,
        heads: cutlass.Int32,
        max_pages: cutlass.Int32,
        workers: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mLengths,
            mSnapshot,
            mScheduler,
            mValidPages,
            query_length,
            heads,
            max_pages,
            workers,
        ).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        mLengths: cute.Tensor,
        mSnapshot: cute.Tensor,
        mScheduler: cute.Tensor,
        mValidPages: cute.Tensor,
        query_length: cutlass.Int32,
        heads: cutlass.Int32,
        max_pages: cutlass.Int32,
        workers: cutlass.Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        batch = cutlass.Int32(mLengths.shape[0])
        if cutlass.const_expr(self.pages_per_block != 1):
            max_pages = (max_pages + self.pages_per_block - 1) // self.pages_per_block
        if tid == 0:
            mScheduler[0] = 0
        for request in cutlass.range(tid, batch, 256):
            length = mLengths[request]
            mSnapshot[request] = length
            if cutlass.const_expr(self.include_partial_block):
                complete = length // self.pooling_factor
                count = (
                    complete + self.entries_per_block - 1
                ) // self.entries_per_block
            else:
                count = (length - 1) // 128
            if count < 0:
                count = cutlass.Int32(0)
            if count > max_pages:  # noqa: PLR1730
                count = max_pages
            if cutlass.const_expr(self.blocks_per_tile != 1):
                count = (count + self.blocks_per_tile - 1) // self.blocks_per_tile
            mScheduler[request + 1] = count
        cute.arch.sync_threads()

        # One warp scans successive chunks; no batch-sized shared allocation.
        if tid < 32:
            carry = cutlass.Int32(0)
            for base in cutlass.range(0, batch, 32):
                request = base + lane
                value = cutlass.Int32(0)
                if request < batch:
                    value = mScheduler[request + 1]
                for shift in cutlass.range_constexpr(5):
                    offset = 1 << shift
                    previous = cute.arch.shuffle_sync_up(
                        value, offset=offset, mask_and_clamp=0
                    )
                    if lane >= offset:
                        value += previous
                value += carry
                if request < batch:
                    mScheduler[request + 1] = value
                carry = cute.arch.shuffle_sync(value, 31)
        cute.arch.sync_threads()

        total = mScheduler[batch]
        quotient = total // workers
        remainder = total % workers
        start_offset = batch + workers + 2
        for worker in cutlass.range(tid, workers + 1, 256):
            extra = remainder
            if worker < remainder:
                extra = worker
            begin = worker * quotient + extra
            mScheduler[batch + 1 + worker] = begin
            if worker < workers:
                lower = cutlass.Int32(0)
                upper = batch
                while lower < upper:
                    middle = (lower + upper) // 2
                    if mScheduler[middle + 1] <= begin:
                        lower = middle + 1
                    else:
                        upper = middle
                logical_page = begin - mScheduler[lower]
                end_extra = remainder
                if worker + 1 < remainder:
                    end_extra = worker + 1
                end = (worker + 1) * quotient + end_extra
                request_end = begin
                if lower < batch:
                    request_end = mScheduler[lower + 1]
                mScheduler[start_offset + worker] = lower
                mScheduler[start_offset + workers + worker] = logical_page
                segment = end - begin
                if request_end < end:
                    segment = request_end - begin
                if segment > 16:
                    segment = cutlass.Int32(16)
                mScheduler[start_offset + 2 * workers + worker] = segment

        rows = batch * query_length
        for row in cutlass.range(tid, rows, 256):
            request = row // query_length
            token = row % query_length
            if cutlass.const_expr(self.include_partial_block):
                complete = (
                    mSnapshot[request] - query_length + token + 1
                ) // self.pooling_factor
                valid = (
                    complete + self.entries_per_block - 1
                ) // self.entries_per_block
            else:
                valid = (mSnapshot[request] - query_length + token) // 128 + 1
            if valid < 1:
                valid = cutlass.Int32(1)
            if valid > max_pages:  # noqa: PLR1730
                valid = max_pages
            for head in cutlass.range(heads):
                mValidPages[head * rows + row] = valid
