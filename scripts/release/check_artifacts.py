"""Validate release archives and write a manifest bound to their exact bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.version import Version


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_archives(dist: Path) -> dict:
    wheels = list(dist.glob("*.whl"))
    sources = list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one wheel and one sdist")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        metadata_names = [n for n in names if n.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError("Wheel must contain one distribution")
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        if metadata["Name"] != "minimax-msa":
            raise ValueError("Unexpected distribution name")
        version = str(Version(metadata["Version"]))
        required = [
            "msa_v1/__init__.py",
            "inference/__init__.py",
            "inference/_vendor/cutlass/include/cutlass/cutlass.h",
            "inference/_vendor/cutlass/include/cute/tensor.hpp",
            "inference/_vendor/cutlass/LICENSE.txt",
            "inference/_vendor/cutlass/provenance.json",
            "inference/msa_v1/attention/prefill/q8kv4/csrc/templates/prefill_attention_inst.cu.jinja",
        ]
        for name in required:
            if name not in names:
                raise ValueError(f"Missing wheel resource: {name}")
        for name in names:
            parts = Path(name).parts
            if any(
                p in {"agent", ".git", "__pycache__", "tests", "benchmarks", "msa_v2", "high_tps"}
                for p in parts
            ):
                raise ValueError(f"Unexpected wheel content: {name}")
            if Path(name).suffix in {".pyc", ".so", ".o", ".cubin", ".ptx", ".log"}:
                raise ValueError(f"Unexpected generated/binary content: {name}")
        provenance = json.loads(archive.read("inference/_vendor/cutlass/provenance.json"))
    with tarfile.open(sources[0]) as archive:
        names = archive.getnames()
        if any({"msa_v2", "high_tps"}.intersection(Path(name).parts) for name in names):
            raise ValueError("sdist exceeds the public MSA v1 scope")
        root = names[0].split("/")[0]
        for relative in (
            "setup.py",
            "pyproject.toml",
            "release/cutlass.json",
            "third_party/cutlass/include/cutlass/cutlass.h",
            "third_party/cutlass/LICENSE.txt",
        ):
            if f"{root}/{relative}" not in names:
                raise ValueError(f"Missing sdist resource: {relative}")
        metadata_file = archive.extractfile(f"{root}/PKG-INFO")
        if (
            metadata_file is None
            or BytesParser().parse(metadata_file)["Version"] != version
        ):
            raise ValueError("sdist and wheel versions differ")
    return {
        "version": version,
        "cutlass": provenance,
        "artifacts": {p.name: sha256(p) for p in (wheels[0], sources[0])},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tag")
    args = parser.parse_args()
    manifest = check_archives(args.dist)
    if args.tag and args.tag != f"v{manifest['version']}":
        parser.error("Tag does not match wheel metadata")
    manifest["commit"] = args.commit
    (args.dist / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
