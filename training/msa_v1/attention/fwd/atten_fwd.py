"""Compatibility import retained throughout minimax-msa 0.1.x."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("msa_v1.attention.fwd.atten_fwd_sm100")
