"""Verify canonical and legacy imports in an installed wheel, in both orders."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path


def module_name(path):
    return path.removeprefix("training/").removesuffix(".py").replace("/", ".")


def check(mapping, order, symbols=None):
    dist = distribution("minimax-msa")
    direct_url = dist.read_text("direct_url.json")
    if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable"):
        raise RuntimeError("Alias acceptance requires an installed wheel")
    installed = {Path(dist.locate_file(p)).resolve() for p in dist.files or []}
    for old, new in (symbols or {}).items():
        entries = [old, new]
        if order == "new-first":
            entries.reverse()
        objects = []
        for entry in entries:
            path, attribute = entry.split(":")
            module = importlib.import_module(module_name(path))
            assert Path(module.__file__).resolve() in installed, module.__file__
            objects.append(getattr(module, attribute))
        assert objects[0] is objects[1], (old, new)
    count = 0
    for old, new in mapping.items():
        if not old.startswith(("training/", "inference/")):
            continue
        names = [module_name(old), module_name(new)]
        if order == "new-first":
            names.reverse()
        first, second = [importlib.import_module(name) for name in names]
        assert first is second, (old, new)
        for name in names:
            parent, attribute = name.rsplit(".", 1)
            assert getattr(importlib.import_module(parent), attribute) is first
        assert Path(first.__file__).resolve() in installed, first.__file__
        marker = object()
        first._msa_alias_acceptance_marker = marker
        try:
            assert second._msa_alias_acceptance_marker is marker
        finally:
            del first._msa_alias_acceptance_marker
        count += 1
    # Also catch dependencies imported through a source checkout or PYTHONPATH.
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in {"msa_v1", "inference"}:
            source = getattr(module, "__file__", None)
            if source:
                assert Path(source).resolve() in installed, (name, source)
    print(
        json.dumps(
            {
                "version": dist.version,
                "order": order,
                "aliases": count,
                "symbols": len(symbols or {}),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--order", choices=("old-first", "new-first"))
    args = parser.parse_args()
    if args.order:
        manifest = json.loads(args.manifest.read_text())
        check(manifest["files"], args.order, manifest.get("symbols"))
    else:
        for order in ("old-first", "new-first"):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--manifest",
                    str(args.manifest.resolve()),
                    "--order",
                    order,
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
