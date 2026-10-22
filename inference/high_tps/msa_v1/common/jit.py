"""Lazy JIT compilation for the SM100/SM103 M3 Decode MegaKernel."""

import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import jinja2

from inference.msa_v1._build_utils import cutlass_root

_CACHE_ABI = (
    "msa_v1_m3_decode_megakernel_q8kv4_sparse16_valid_pages_split8_direct_grid_compact_params_"
    "attention_projection_mxfp8_output_projection_tp4_rmsnorm_"
    "post_attention_norm_mxfp8_shared_fc12_mxfp8_routed_schedule_nvfp4_"
    "routed_fc1_nvfp4_dynamic_task_scheduler"
)


def _compute_cache_base() -> Path:
    explicit = os.environ.get("CACHE_DIR")
    if explicit:
        base = Path(explicit).expanduser().resolve()
    else:
        base = Path(os.path.expanduser("~/.cache/minimax_msa"))

    return base / _CACHE_ABI


CACHE_BASE = _compute_cache_base()
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CSRC_DIR = _PROJECT_ROOT / "csrc"
_SOURCE_OWNERS = {"mega_atten": "attn"}


def _variant_cache_paths(variant_name: str) -> tuple[Path, Path]:
    """Return stable cache paths whose components fit POSIX NAME_MAX."""
    encoded_name = variant_name.encode()
    digest = hashlib.sha256(encoded_name).hexdigest()
    directory_name = (
        variant_name
        if len(encoded_name) <= 255
        else f"{variant_name[:96]}_{digest[:16]}"
    )
    shared_object_name = (
        f"{variant_name}.so" if len(encoded_name) + len(".so") <= 255 else "kernel.so"
    )
    cache_dir = CACHE_BASE / directory_name
    return cache_dir, cache_dir / shared_object_name


@contextmanager
def _cache_file_lock(name: str):
    """Serialize cache updates across local processes."""
    CACHE_BASE.mkdir(parents=True, exist_ok=True)
    lock_key = hashlib.sha256(name.encode()).hexdigest()
    lock_path = CACHE_BASE / f".{lock_key}.lock"
    with open(lock_path, "a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _shared_object_is_usable(path: Path) -> bool:
    """Reject missing or obviously truncated ELF shared objects."""
    try:
        if path.stat().st_size < 4096:
            return False
        with open(path, "rb") as source:
            return source.read(4) == b"\x7fELF"
    except OSError:
        return False


def _build_fingerprint(variant_name: str, variant_config: dict) -> str:
    """Hash local compile inputs without changing the stable cache ABI."""
    digest = hashlib.sha256()
    digest.update(variant_name.encode())
    for key, value in sorted(variant_config.items()):
        digest.update(f"\0{key}={value}".encode())

    owner = _SOURCE_OWNERS[variant_config["kernel_mode"]]
    inputs = {Path(__file__).resolve()}
    pending = list((_CSRC_DIR / owner / "kernel" / "templates").glob("*.jinja"))
    pending.append(_CSRC_DIR / owner / "kernel/include/m3_decode_sm100.cuh")
    while pending:
        path = pending.pop()
        if path in inputs:
            continue
        inputs.add(path)
        for include in re.findall(
            r'^#include "([^"\n]+)"', path.read_text(), re.MULTILINE
        ):
            dependency = _CSRC_DIR / include
            if dependency.is_file():
                pending.append(dependency)
    for path in sorted(inputs):
        digest.update(f"\0{path.relative_to(_PROJECT_ROOT)}\0".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cached_variant_is_current(so_path: Path, fingerprint: str) -> bool:
    if not _shared_object_is_usable(so_path):
        return False
    try:
        return (
            so_path.parent / ".build_fingerprint"
        ).read_text().strip() == fingerprint
    except OSError:
        return False


def _get_tvm_ffi_include() -> str:
    try:
        import tvm_ffi

        tvm_dir = Path(tvm_ffi.__path__[0])
        inc = tvm_dir / "include"
        if inc.exists():
            return str(inc)
        inc2 = tvm_dir.parent / "include"
        if inc2.exists():
            return str(inc2)
    except ImportError:
        pass
    raise RuntimeError("Cannot find TVM-FFI include directory; install apache-tvm-ffi")


def _get_cuda_home() -> str:
    if "CUDA_HOME" in os.environ:
        return os.environ["CUDA_HOME"]
    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).resolve().parent.parent)
    for path in ["/usr/local/cuda", "/opt/cuda"]:
        if os.path.isdir(path):
            return path
    raise RuntimeError("Cannot find CUDA toolkit. Set CUDA_HOME.")


@lru_cache(maxsize=1)
def _require_qmul4_ptx_support() -> None:
    with _cache_file_lock("qmul4_ptx_probe"):
        cuda_home = _get_cuda_home()
        nvcc = os.path.join(cuda_home, "bin", "nvcc")
        nvcc_version = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, check=True
        ).stdout
        toolchain_key = hashlib.sha256(
            (str(Path(nvcc).resolve()) + "\n" + nvcc_version).encode()
        ).hexdigest()[:16]
        probe_dir = CACHE_BASE / "_probes" / "qmul4_ptx" / toolchain_key
        probe_dir.mkdir(parents=True, exist_ok=True)
        src = probe_dir / "qmul4_probe.cu"
        obj = probe_dir / "qmul4_probe.o"
        log = probe_dir / "qmul4_probe.log"
        source = r"""
#include <stdint.h>

extern "C" __global__ void qmul4_probe(uint32_t* out, uint16_t a, uint32_t b) {
  uint32_t dst;
  asm volatile(
      "{ .reg .b16 value;\n"
      "  mov.b16 value, %1;\n"
      "  mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, value, %2; }\n"
      : "=r"(dst)
      : "h"(a), "r"(b));
  out[threadIdx.x] = dst;
}
"""
        if (
            obj.exists()
            and obj.stat().st_size > 0
            and src.exists()
            and src.read_text() == source
        ):
            return
        src.write_text(source)
        staged_obj = probe_dir / f".qmul4_probe.{os.getpid()}.o.tmp"
        cmd = [
            nvcc,
            "-std=c++20",
            "-O3",
            "-gencode=arch=compute_100a,code=sm_100a",
            "-c",
            str(src),
            "-o",
            str(staged_obj),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        log.write_text(
            f"cmd: {' '.join(cmd)}\n\nstdout:\n{result.stdout}"
            f"\n\nstderr:\n{result.stderr}\n"
        )
        if result.returncode != 0:
            staged_obj.unlink(missing_ok=True)
            raise RuntimeError(
                "The CUDA toolchain does not support the required qmul4 PTX "
                f"instruction. See {log} for the compiler diagnostic."
            )
        os.replace(staged_obj, obj)


def _get_nvcc_flags(cache_dir: Path) -> str:
    tvm_include = _get_tvm_ffi_include()
    kernel_include = str(_CSRC_DIR)
    cutlass = cutlass_root()
    cutlass_include = str(cutlass / "include")
    cutlass_util_include = str(cutlass / "tools" / "util" / "include")
    arch_flags = [
        "-gencode=arch=compute_100a,code=sm_100a",
        "-gencode=arch=compute_103a,code=sm_103a",
    ]
    nvcc_flags = [
        "-O3",
        "-lineinfo",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        *arch_flags,
        "-static-global-template-stub=false",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
        "-DFLASHINFER_ENABLE_FP8_E8M0",
        "-DFLASHINFER_ENABLE_FP4_E2M1",
        "-DFLASHINFER_ENABLE_F16",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-Xcudafe",
        "--diag_suppress=2908",
        f"-I{kernel_include}",
        f"-I{cutlass_include}",
        f"-I{cutlass_util_include}",
        f"-I{tvm_include}",
        f"-I{cache_dir}",
        "-use_fast_math",
        "-DNDEBUG",
        "-Xptxas",
        "-O3",
        "-Xcompiler",
        "-fPIC",
    ]
    return " ".join(nvcc_flags)


class _VariantWrapper:
    def __init__(self, fn):
        self._fn = fn

    def run(self, *args):
        return self._fn(*args)


class M3DecodeVariantManager:
    def __init__(self):
        self._loaded = {}
        self._lock = threading.Lock()
        self._inst_templates = {}
        self._run_templates = {}

    def _load_templates(self):
        if not self._inst_templates:
            filenames = {
                "mega_atten": "m3_decode_mega_atten_sm100_run.cu.jinja",
            }
            for mode, filename in filenames.items():
                templates = _CSRC_DIR / _SOURCE_OWNERS[mode] / "kernel/templates"
                self._inst_templates[mode] = jinja2.Template(
                    (templates / "m3_decode_sm100_inst.cu.jinja").read_text()
                )
                self._run_templates[mode] = jinja2.Template(
                    (templates / filename).read_text()
                )

    def _variant_info(
        self,
        *,
        kernel_mode: str = "mega_atten",
        wait_deadline: bool = False,
        enable_placement: bool = False,
    ):
        if kernel_mode != "mega_atten":
            raise ValueError(f"unsupported M3 Decode kernel mode: {kernel_mode}")
        return (
            "m3_decode_mega_atten_sparse16_runtime_kv_exact_topk_"
            "attention_projection_output_projection_tp4_postnorm"
            + ("_cluster_placement" if enable_placement else "_canonical_placement")
            + ("_wait_deadline" if wait_deadline else ""),
            {
                "kernel_mode": kernel_mode,
                "wait_deadline": "1" if wait_deadline else "0",
                "enable_placement": "1" if enable_placement else "0",
            },
        )

    def get_m3_decode_mega_atten_variant(
        self, *, wait_deadline: bool = False, enable_placement: bool = False
    ):
        variant_name, variant_config = self._variant_info(
            kernel_mode="mega_atten", wait_deadline=wait_deadline,
            enable_placement=enable_placement,
        )
        return self._get_or_compile(variant_name, variant_config)

    def _get_or_compile(self, variant_name: str, variant_config: dict):
        cached = self._loaded.get(variant_name)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._loaded.get(variant_name)
            if cached is not None:
                return cached

            _, so_path = _variant_cache_paths(variant_name)
            with _cache_file_lock(f"{variant_name}.compile"):
                fingerprint = _build_fingerprint(variant_name, variant_config)
                if not _cached_variant_is_current(so_path, fingerprint):
                    if "ALLOW_JIT" not in os.environ:
                        raise RuntimeError(
                            "JIT disabled and a current M3 Decode MegaKernel was not "
                            f"found: {so_path}. The cached binary is missing, invalid, "
                            "or was built from different kernel sources."
                        )
                    self._compile_only(variant_name, variant_config, fingerprint)

                import tvm_ffi

                module = tvm_ffi.load_module(str(so_path))
            fn = getattr(module, f"run_{variant_name}")
            self._loaded[variant_name] = _VariantWrapper(fn)
            return self._loaded[variant_name]

    def _compile_only(self, variant_name: str, variant_config: dict, fingerprint: str):
        _require_qmul4_ptx_support()
        cache_dir, so_path = _variant_cache_paths(variant_name)
        fingerprint_path = cache_dir / ".build_fingerprint"
        if _cached_variant_is_current(so_path, fingerprint):
            return
        so_path.unlink(missing_ok=True)
        fingerprint_path.unlink(missing_ok=True)

        cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_templates()

        params = {
            "variant_name": variant_name,
            "func_name": "launch_" + variant_name,
            **variant_config,
        }
        inst_cu = cache_dir / "m3_decode_inst.cu"
        run_cu = cache_dir / "m3_decode_run.cu"
        params["source_owner"] = _SOURCE_OWNERS[variant_config["kernel_mode"]]
        inst_source = self._inst_templates[variant_config["kernel_mode"]].render(
            **params
        )
        run_source = self._run_templates[variant_config["kernel_mode"]].render(**params)
        # A changed dependency fingerprint invalidates both translation units,
        # including the host wrapper when only a shared ABI header changed.
        inst_cu.write_text(inst_source)
        run_cu.write_text(run_source)

        staged_so = self._write_ninja(cache_dir, variant_name, inst_cu, run_cu)
        self._run_ninja(cache_dir, "M3 Decode MegaKernel variant")
        if not _shared_object_is_usable(staged_so):
            staged_so.unlink(missing_ok=True)
            raise RuntimeError(
                "M3 Decode MegaKernel build produced an invalid shared object: "
                f"{staged_so}"
            )
        os.replace(staged_so, so_path)
        staged_fingerprint = cache_dir / f".build_fingerprint.{os.getpid()}.tmp"
        staged_fingerprint.write_text(fingerprint + "\n")
        os.replace(staged_fingerprint, fingerprint_path)

    def _write_ninja(
        self,
        cache_dir: Path,
        variant_name: str,
        inst_cu: Path,
        run_cu: Path,
    ) -> Path:
        cuda_home = _get_cuda_home()
        nvcc = os.path.join(cuda_home, "bin", "nvcc")
        nvcc_flags = _get_nvcc_flags(cache_dir)

        staged_so = cache_dir / f".link.{os.getpid()}.so.tmp"
        staged_so.unlink(missing_ok=True)
        inst_obj = cache_dir / "m3_decode_inst.o"
        run_obj = cache_dir / "m3_decode_run.o"

        ninja_content = f"""ninja_required_version = 1.5

nvcc = {nvcc}
nvcc_flags = {nvcc_flags}

rule nvcc_compile
  command = $nvcc $nvcc_flags -c $in -o $out
  description = Compiling $in

rule nvcc_link
  command = $nvcc -shared $in -o $out -lcuda
  description = Linking $out

build {inst_obj}: nvcc_compile {inst_cu}
build {run_obj}: nvcc_compile {run_cu}
build {staged_so}: nvcc_link {inst_obj} {run_obj}
"""
        (cache_dir / "build.ninja").write_text(ninja_content)
        return staged_so

    @staticmethod
    def _run_ninja(cache_dir: Path, label: str):
        result = subprocess.run(
            ["ninja", "-j1"],
            cwd=str(cache_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{label} compilation failed:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )


_m3_decode_variant_manager = M3DecodeVariantManager()


def get_m3_decode_mega_atten_variant(
    *, wait_deadline: bool = False, enable_placement: bool = False
):
    return _m3_decode_variant_manager.get_m3_decode_mega_atten_variant(
        wait_deadline=wait_deadline, enable_placement=enable_placement
    )
