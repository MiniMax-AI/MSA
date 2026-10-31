"""Check installed wheel origins and resources from outside the source checkout."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from importlib.metadata import distribution
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--resources-only", action="store_true")
    args = parser.parse_args()
    dist = distribution("minimax-msa")
    if dist.version != args.version:
        raise RuntimeError(f"Expected {args.version}, installed {dist.version}")
    direct_url = dist.read_text("direct_url.json")
    if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable"):
        raise RuntimeError("Editable installs are not release acceptance")
    installed = {Path(dist.locate_file(p)).resolve() for p in dist.files or []}
    modules = ["inference", "inference.msa_v1._build_utils"]
    if not args.resources_only:
        modules += [
            "msa_v1",
            "inference.msa_v1.indexer.prefill.q8kv8",
            "inference.msa_v1.attention.prefill.q8kv4",
            "inference.msa_v1.attention.decode.q8kv4",
            "inference.dequant",
        ]
    for name in modules:
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() not in installed:
            raise RuntimeError(
                f"{name} was imported outside the wheel: {module.__file__}"
            )
    from inference.msa_v1._build_utils import cutlass_root

    root = cutlass_root()
    if (root / "provenance.json").resolve() not in installed:
        raise RuntimeError(f"CUTLASS resolved outside the wheel: {root}")
    print(
        json.dumps(
            {
                "python": sys.version,
                "version": dist.version,
                "cutlass": json.loads((root / "provenance.json").read_text()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
