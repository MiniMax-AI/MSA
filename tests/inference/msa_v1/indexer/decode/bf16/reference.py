"""Independent FP32 reference for scaled BF16 page-max scores."""

import torch

from tests.inference.msa_v1.indexer.decode.q8kv8.reference import (
    indexer_gemm_reference as _page_max_reference,
)
from tests.inference.msa_v1.indexer.decode.q8kv8.reference import (
    make_inputs as _make_inputs,
)


def indexer_gemm_reference(q, k_cache, page_table, seq_lens):
    return _page_max_reference(q, k_cache, page_table, seq_lens) / (128**0.5)


def make_inputs(*args, **kwargs):
    return _make_inputs(*args, **kwargs, dtype=torch.bfloat16)
