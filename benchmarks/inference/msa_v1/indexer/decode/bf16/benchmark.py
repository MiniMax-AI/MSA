"""BF16 measurements using the same input and timing protocol as Q8KV8."""

from functools import partial

import torch

from benchmarks.inference.msa_v1.indexer.decode.q8kv8.benchmark import (
    MAX_CV as MAX_CV,
)
from benchmarks.inference.msa_v1.indexer.decode.q8kv8.benchmark import (
    MAX_TIMING_ATTEMPTS as MAX_TIMING_ATTEMPTS,
)
from benchmarks.inference.msa_v1.indexer.decode.q8kv8.benchmark import (
    _make_slot as _make_dense_slot,
)
from benchmarks.inference.msa_v1.indexer.decode.q8kv8.benchmark import (
    _resolve_slots as _resolve_slots,
)
from benchmarks.inference.msa_v1.indexer.decode.q8kv8.benchmark import (
    _time_graphs as _time_graphs,
)
from benchmarks.inference.msa_v1.indexer.decode.q8kv8.benchmark import main
from inference.msa_v1.indexer.decode.bf16 import (
    BatchDecodeIndexerWithPagedKVCacheWrapper as BatchDecodeIndexerWithPagedKVCacheWrapper,
)

PAGE_BYTES = 128 * 128 * 2
_make_slot = partial(_make_dense_slot, dtype=torch.bfloat16)


if __name__ == "__main__":
    main(precision="bf16")
