"""Compatibility import retained throughout minimax-msa 0.1.x."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module(
    "inference.msa_v1.attention.prefill._common.prepare_k2q_csr"
)
