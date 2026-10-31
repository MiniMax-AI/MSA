"""Compatibility import retained throughout minimax-msa 0.1.x."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "inference.msa_v1.indexer.prefill.q8kv8.indexer_gemm_sm100"
)
