"""CUDA Graph E2E benchmark for BF16 sparse decode attention."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from benchmarks.inference.msa_v1.attention.decode._benchmark import main

if __name__ == "__main__":
    main("bf16")
