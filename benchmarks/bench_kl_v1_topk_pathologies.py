"""Compatibility import retained throughout minimax-msa 0.1.x."""

import importlib
import sys

if __name__ == "__main__":
    import runpy
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("benchmarks.benchmark_kl_v1_topk_pathologies", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module(
        "benchmarks.benchmark_kl_v1_topk_pathologies"
    )
