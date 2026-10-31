"""Bundle pinned public CUTLASS headers without importing GPU dependencies."""

import hashlib
import json
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent
CUTLASS = ROOT / "third_party/cutlass"
PIN = ROOT / "release/cutlass.json"


def cutlass_files():
    files = [CUTLASS / "LICENSE.txt"]
    for directory in ("include", "tools/util/include"):
        files.extend(p for p in (CUTLASS / directory).rglob("*") if p.is_file())
    return sorted(files)


def verify_cutlass():
    pin = json.loads(PIN.read_text())
    digest = hashlib.sha256()
    for path in cutlass_files():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("Initialize the pinned third_party/cutlass submodule")
        relative = path.relative_to(CUTLASS).as_posix().encode()
        content = path.read_bytes()
        digest.update(relative + b"\0" + len(content).to_bytes(8, "little") + content)
    if digest.hexdigest() != pin["sha256"]:
        raise RuntimeError(
            "CUTLASS resources differ from release/cutlass.json; initialize the "
            "pinned submodule or review and update the pin before building"
        )


class BuildPy(build_py):
    def run(self):
        verify_cutlass()
        super().run()
        destination = Path(self.build_lib) / "inference/_vendor/cutlass"
        # Remove stale resources when a developer reuses a build directory.
        if destination.exists():
            shutil.rmtree(destination)
        for source in cutlass_files():
            target = destination / source.relative_to(CUTLASS)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        shutil.copyfile(PIN, destination / "provenance.json")


class SourceDistribution(sdist):
    def run(self):
        verify_cutlass()
        super().run()


if __name__ == "__main__":
    setup(cmdclass={"build_py": BuildPy, "sdist": SourceDistribution})
