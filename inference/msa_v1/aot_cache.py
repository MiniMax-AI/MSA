"""Persistent AOT cache for MSA v1 inference CuTe DSL kernels.

Saves compiled TVM FFI kernels as .o files on first compile and loads them on
subsequent runs to skip JIT compilation.

Environment variables:
    MSA_V1_AOT_CACHE: Override cache directory
        (default: ~/.cache/minfer/msa_v1)
    MSA_V1_AOT_DISABLE=1: Disable AOT cache entirely
"""

import hashlib
import logging
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import cutlass
import cutlass.cute as cute

try:
    import tvm_ffi
except ImportError:
    tvm_ffi = None

logger = logging.getLogger(__name__)

_AOT_CACHE_ENV = "MSA_V1_AOT_CACHE"
_AOT_DISABLE_ENV = "MSA_V1_AOT_DISABLE"
_AOT_CACHE_DIR = os.environ.get(
    _AOT_CACHE_ENV,
    os.path.expanduser("~/.cache/minfer/msa_v1"),
)
_AOT_DISABLE = os.environ.get(_AOT_DISABLE_ENV, "0") == "1"
_CUTLASS_DSL_VERSION = getattr(cutlass, "__version__", "unknown")
_CUDA_BACKEND_VERSION = str(cutlass.CUDA_VERSION)
_TVM_FFI_VERSION = getattr(tvm_ffi, "__version__", "unavailable")
_SOURCE_ROOT = Path(__file__).resolve().parent

_loaded_modules: dict[str, object] = {}


def _iter_package_sources(source_root: Path):
    """Yield importable package sources in deterministic path order."""

    for src in sorted(source_root.rglob("*.py")):
        relative = src.relative_to(source_root)
        if not src.stem.isidentifier():
            continue
        if any(not part.isidentifier() for part in relative.parts[:-1]):
            continue
        yield src, relative


def _hash_source_tree(source_root: Path) -> str:
    """Hash inference MSA v1 sources and runtime ABI stamps."""

    h = hashlib.sha256()
    h.update(f"py={sys.version_info.major}.{sys.version_info.minor}".encode())
    h.update(f"cutlass={_CUTLASS_DSL_VERSION}".encode())
    h.update(f"cuda_backend={_CUDA_BACKEND_VERSION}".encode())
    h.update(f"tvm_ffi={_TVM_FFI_VERSION}".encode())
    for src, relative in _iter_package_sources(source_root):
        h.update(relative.as_posix().encode())
        content = src.read_bytes()
        h.update(len(content).to_bytes(8, "little"))
        h.update(content)
    return h.hexdigest()


@lru_cache(maxsize=1)
def _compute_source_fingerprint() -> str:
    """Return the process-stable fingerprint for MSA v1 inference sources."""

    return _hash_source_tree(_SOURCE_ROOT)


def _key_to_path(key: tuple) -> str:
    h = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    name = str(key[0]).replace("/", "_")
    return os.path.join(
        _AOT_CACHE_DIR,
        _compute_source_fingerprint(),
        f"{name}_{h}",
    )


def try_load_aot(key: tuple):
    if _AOT_DISABLE:
        return None
    obj_path = _key_to_path(key) + ".o"
    if not os.path.isfile(obj_path):
        if "FMHA_SM100_ALLOW_JIT" not in os.environ:
            raise RuntimeError(
                f"JIT compilation is disabled and AOT kernel {key!r} was not found; "
                "set FMHA_SM100_ALLOW_JIT=1 to allow JIT compilation"
            )
        return None
    func_name = str(key[0])
    try:
        if obj_path not in _loaded_modules:
            _loaded_modules[obj_path] = cute.runtime.load_module(
                obj_path, enable_tvm_ffi=True
            )
        loaded = getattr(_loaded_modules[obj_path], func_name)
        logger.debug("AOT cache hit: %s", func_name)
        return loaded
    except Exception as e:
        if "FMHA_SM100_ALLOW_JIT" not in os.environ:
            raise RuntimeError(
                f"JIT compilation is disabled and AOT kernel {key!r} could not "
                f"be loaded from {obj_path}"
            ) from e
        logger.warning("Failed to load AOT kernel from %s: %s", obj_path, e)
        return None


def save_aot(key: tuple, compiled) -> None:
    if _AOT_DISABLE:
        return
    if not hasattr(compiled, "export_to_c"):
        return
    obj_path = _key_to_path(key) + ".o"
    os.makedirs(os.path.dirname(obj_path), exist_ok=True)
    tmp_path = obj_path + f".tmp.{os.getpid()}"
    func_name = str(key[0])
    try:
        t0 = time.time()
        compiled.export_to_c(tmp_path, function_name=func_name)
        os.replace(tmp_path, obj_path)
        dt = time.time() - t0
        logger.debug("Saved AOT kernel %s to %s in %.1fs", func_name, obj_path, dt)
    except Exception as e:
        logger.warning("Failed to save AOT kernel %s: %s", func_name, e)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def compile_or_load(key: tuple, compile_fn, log_prefix: str = "aot_cache"):
    """Load an AOT kernel or invoke ``compile_fn`` and persist its result."""

    loaded = try_load_aot(key)
    if loaded is not None:
        return loaded
    logger.debug("[%s] AOT cache miss: %s; starting cute.compile", log_prefix, key[0])
    t0 = time.time()
    compiled = compile_fn()
    logger.debug(
        "[%s] cute.compile done: %s in %.1fs",
        log_prefix,
        key[0],
        time.time() - t0,
    )
    save_aot(key, compiled)
    return compiled
