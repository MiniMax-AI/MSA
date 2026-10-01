# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Per-variant lazy JIT compilation for FMHA varlen kernels.

Each FMHA variant (dtype x tile x sparse x page x split_kv x pack_factor) and each fixed module
(plan, sparse top-k, reductions, indexers) is compiled independently on first use and cached under
~/.cache/minfer/fmha_sm100/ (``MINFER_FMHA_CACHE_DIR``). The cache checks every library against
the content of the files it was compiled from (``_jit_cache``), so a kernel change rebuilds only
the libraries that include the changed files; nothing needs clearing by hand.
"""

import itertools
import logging
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jinja2

from . import _jit_cache

logger = logging.getLogger(__name__)

def _compute_cache_base():
    explicit = os.environ.get("MINFER_FMHA_CACHE_DIR")
    if explicit:
        return Path(explicit)
    base = Path(os.path.expanduser("~/.cache/minfer/fmha_sm100"))
    # Different build configs get separate cache dirs to avoid conflicts
    suffix = ""
    if os.environ.get("GPU_TRACE") is not None:
        suffix += "_gpu_trace"
    if os.environ.get("SM_TIMING") is not None:
        suffix += "_sm_timing"
    if os.environ.get("FMHA_GMEM_CHECK") is not None:
        suffix += "_gmem_check"
    if suffix:
        base = base.parent / (base.name + suffix)
    return base


CACHE_BASE = _compute_cache_base()


# Kernel sources and CUTLASS headers are shipped inside the package directory
# so that JIT compilation works from both editable and wheel installs.
_PACKAGE_DIR = Path(__file__).resolve().parent
_FMHA_VARLEN_DIR = _PACKAGE_DIR / "csrc"
_CUTLASS_DIR = _PACKAGE_DIR / "cutlass"
_CUTLASS_INCLUDE = _CUTLASS_DIR / "include"
_CUTLASS_UTIL_INCLUDE = _CUTLASS_DIR / "tools" / "util" / "include"

_PACK_FACTORS = [1,2,4,6,8,16]
# _PACK_FACTORS = [1, 6]

# DLPack dtype codes (must match tvm_ffi_utils.h encode_dlpack_dtype)
_BFLOAT16_CODE = (4 << 16) | (16 << 8) | 1      # 266241
_FLOAT8_E4M3FN_CODE = (12 << 16) | (8 << 8) | 1  # 788481

_FMHA_SM100_DISPATCH = [
    ("int64_t dtype_code", [
        (_BFLOAT16_CODE,      {"dtype_in": "nv_bfloat16",   "cutlass_dtype_out": "cutlass::bfloat16_t"}),
        (_FLOAT8_E4M3FN_CODE, {"dtype_in": "__nv_fp8_e4m3", "cutlass_dtype_out": "cutlass::bfloat16_t"}),
    ]),
    ("int qo_tile_size", [
        (128, {"tile_q": "_128", "tile_kv": "_256", "thread_shape": "_1, _2, _1"}),
        (256, {"tile_q": "_256", "tile_kv": "_128", "thread_shape": "_2, _1, _1"}),
    ]),
    ("bool single_wg", [
        ("true",  {"single_wg": "true"}),
        ("false", {"single_wg": "false"}),
    ]),
    ("int sparse_mode", [
        (0,    {"sparse_mode": "Sparse"}),
        (1,    {"sparse_mode": "Full"}),
        (2,    {"sparse_mode": "OnlyScore"}),
        (None, {"sparse_mode": "Off"}),
    ]),
    ("int page_size", [
        (-1,  {"page_size": -1}),
        (128, {"page_size": 128}),
        # (256, {"page_size": 256}),
    ]),
    ("bool split_kv", [
        ("false", {"is_split_kv": "false"}),
        ("true",  {"is_split_kv": "true"}),
    ]),
    ("int pack_factor", [(i, {"pack_factor": i}) for i in _PACK_FACTORS]),
]

# NVFP4 has a separate cache identity; existing FP8 names and ABI stay stable.
_FMHA_SM100_KV_DTYPE = [
    # (runtime value, jinja/build params)
    ("fp8",   {"kv_mode": 0, "kv_suffix": ""}),
    ("nvfp4", {"kv_mode": 3, "kv_suffix": "_nvfp4_v1"}),
]


def _kv_dtype_idx(kv_dtype):
    """Select the ordinary kernel or the explicit NVFP4 cache reader."""
    if kv_dtype in (None, "fp8"):
        return 0
    if kv_dtype == "nvfp4":
        return 1
    raise ValueError(f"unknown kv_dtype {kv_dtype!r}; expected 'fp8' or 'nvfp4'")


_FMHA_SM100_IMPOSSIBLE = lambda p: (
    (p.get("tile_q") == "_256" and p.get("single_wg") == "true") or
    (p.get("tile_q") == "_256" and p.get("is_split_kv") == "true") or
    (p.get("page_size") == -1 and p.get("sparse_mode") == "Sparse") or
    (p.get("pack_factor", 1) > 1 and p.get("tile_q") == "_256") or
    # NVFP4 uses the FP8 single-softmax-warpgroup decoder with page-128 tiles.
    (p.get("kv_mode", 0) >= 3 and (
        p.get("page_size") != 128 or p.get("single_wg") != "true" or
        p.get("dtype_in") != "__nv_fp8_e4m3"))
)


def _dlpack_dtype_code(torch_dtype):
    """Encode a torch dtype as a DLPack int64 code."""
    import torch
    _map = {
        torch.float16: (2 << 16) | (16 << 8) | 1,
        torch.bfloat16: (4 << 16) | (16 << 8) | 1,
        torch.float32: (2 << 16) | (32 << 8) | 1,
        torch.float8_e4m3fn: (12 << 16) | (8 << 8) | 1,
        torch.float8_e5m2: (13 << 16) | (8 << 8) | 1,
    }
    return _map[torch_dtype]


def _variant_key_from_runtime(dtype_code, qo_tile_size, single_wg,
                               sparse_mode, page_size, split_kv, pack_factor,
                               kv_dtype=None):
    """Compute a variant key string from runtime parameters."""
    dims = _FMHA_SM100_DISPATCH

    def _match_idx(dim_values, runtime_val):
        # Convert Python bool to string to match dispatch table ("true"/"false")
        if isinstance(runtime_val, bool):
            runtime_val = "true" if runtime_val else "false"
        for idx, (match_val, _) in enumerate(dim_values):
            if match_val is None:
                return idx
            if match_val == runtime_val:
                return idx
        return len(dim_values) - 1

    runtime_vals = [dtype_code, qo_tile_size, single_wg, sparse_mode,
                    page_size, split_kv, pack_factor]
    indices = []
    params = {}
    for (_, dim_values), rv in zip(dims, runtime_vals):
        idx = _match_idx(dim_values, rv)
        indices.append(idx)
        _, tparams = dim_values[idx]
        params.update(tparams)

    # name is byte-for-byte what it was before this axis existed.
    kv_idx = _kv_dtype_idx(kv_dtype)
    params.update(_FMHA_SM100_KV_DTYPE[kv_idx][1])

    if _FMHA_SM100_IMPOSSIBLE(params):
        raise ValueError(f"Impossible FMHA variant combination: {params}")

    suffix = params["kv_suffix"]
    func_name = "fmha_sm100_" + "_".join(str(i) for i in indices) + suffix
    variant_name = "_".join(str(i) for i in indices) + suffix
    params["func_name"] = func_name
    params["variant_name"] = variant_name
    return variant_name, params


def _variant_params_from_name(variant_name):
    """Inverse of ``_variant_key_from_runtime``: the build params of a variant name."""
    kv_idx = next((i for i, (_, p) in enumerate(_FMHA_SM100_KV_DTYPE)
                   if p["kv_suffix"] and variant_name.endswith(p["kv_suffix"])), 0)
    suffix = _FMHA_SM100_KV_DTYPE[kv_idx][1]["kv_suffix"]
    fields = variant_name[:len(variant_name) - len(suffix)].split("_")
    if len(fields) != len(_FMHA_SM100_DISPATCH):
        raise ValueError(f"Not an FMHA variant name: {variant_name!r}")
    params = {}
    for (_, dim_values), field in zip(_FMHA_SM100_DISPATCH, fields):
        params.update(dim_values[int(field)][1])
    params.update(_FMHA_SM100_KV_DTYPE[kv_idx][1])
    if _FMHA_SM100_IMPOSSIBLE(params):
        raise ValueError(f"Impossible FMHA variant: {variant_name!r}")
    params["func_name"] = "fmha_sm100_" + variant_name
    params["variant_name"] = variant_name
    return params


def _get_tvm_ffi_include():
    """Find TVM-FFI include directory."""
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


def _get_cuda_home():
    """Find CUDA toolkit root."""
    if "CUDA_HOME" in os.environ:
        return os.environ["CUDA_HOME"]
    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).resolve().parent.parent)
    for p in ["/usr/local/cuda", "/opt/cuda"]:
        if os.path.isdir(p):
            return p
    raise RuntimeError("Cannot find CUDA toolkit. Set CUDA_HOME.")


_namespace_instance = None


def _namespace():
    """The cache namespace of the selected toolchain: ``CACHE_BASE/v2/<toolchain>/``."""
    global _namespace_instance
    if _namespace_instance is None:
        _namespace_instance = _jit_cache.Namespace(CACHE_BASE, _get_cuda_home())
    return _namespace_instance


def _get_nvcc_flags(fmha=True, kv_mode=0, fast_math=True):
    tvm_include = _get_tvm_ffi_include()
    fmha_include = str(_FMHA_VARLEN_DIR / "include")
    cutlass_include = str(_CUTLASS_INCLUDE)
    cutlass_util_include = str(_CUTLASS_UTIL_INCLUDE)
    nvcc_flags = [
        "-O3", "-std=c++20",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-gencode=arch=compute_100a,code=sm_100a",
        "-gencode=arch=compute_103a,code=sm_103a",
        "-gencode=arch=compute_100f,code=sm_100f",
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
        "-Xcudafe --diag_suppress=2908",
        f"-I{fmha_include}",
        f"-I{cutlass_include}",
        f"-I{cutlass_util_include}",
        f"-I{tvm_include}",
        # csrc/ itself: fmha_sm100_params.h, tvm_ffi_utils.h and gmem_bounds_check.h. Sources
        # compile from the package, so the cache records the files the compiler really read.
        f"-I{_FMHA_VARLEN_DIR}",
        "-DNDEBUG", "-Xptxas", "-O1" if fmha else "-O3",
        "-Xcompiler", "-fPIC",
    ]
    if fast_math:
        nvcc_flags.append("-use_fast_math")
    if os.environ.get("GPU_TRACE") is not None:
        nvcc_flags.append("-DGPU_TRACE_ENABLED")
    if os.environ.get("SM_TIMING") is not None:
        nvcc_flags.append("-DSM_TIMING_ENABLED")
    if os.environ.get("FMHA_GMEM_CHECK") is not None:
        nvcc_flags.append("-DFMHA_GMEM_BOUNDS_CHECK")
    # Header templates need the mode macro. Ordinary variants keep their flags.
    if kv_mode:
        nvcc_flags.append(f"-DMSA_NVFP4_KV_MODE={int(kv_mode)}")
        # The macro changes FMHA template bodies without changing their C++
        # names. Hide weak symbols so FP8/FP4 load order cannot interpose them.
        nvcc_flags += ["-Xcompiler", "-fvisibility=hidden"]
    return " ".join(nvcc_flags)


def _nvcc():
    return os.path.join(_get_cuda_home(), "bin", "nvcc")


# Compile and link rules of every library; part of each recipe, so changing them rebuilds.
_NINJA_RULES = """rule nvcc_compile
  command = $nvcc $nvcc_flags -MMD -MF $out.d -c $in -o $out
  description = Compiling $in
  depfile = $out.d
  deps = gcc

rule nvcc_link
  command = $nvcc -shared $in -o $out -lcuda
  description = Linking $out
"""


def _recipe(name, nvcc_flags, templates=(), **key):
    """The cache recipe of one library: its name, compiler, flags and build rules, plus ``key``
    (template parameters). File contents are checked separately, per record."""
    return _jit_cache.Recipe(_namespace(), name, {"nvcc": _nvcc(), "nvcc_flags": nvcc_flags,
                                                  "rules": _NINJA_RULES, **key},
                             templates=templates)


def _build_library(recipe, label, nvcc_flags, sources, jobs=1):
    """Return the library of ``recipe``, compiling it when no record matches the current files.

    ``sources(build_dir)`` returns the translation units, writing generated ones into
    ``build_dir`` first. Each compiles to its own object with a depfile, so the cache learns
    every header it read, and the objects link into one shared library.
    """
    def builder(build_dir):
        logger.info("JIT compiling %s", label)
        library = build_dir / f"{recipe.name}.so"
        objects = [(build_dir / f"{Path(source).stem}.o", source) for source in sources(build_dir)]
        compiles = "\n".join(f"build {obj}: nvcc_compile {source}" for obj, source in objects)
        _jit_cache.write_if_changed(build_dir / "build.ninja", f"""ninja_required_version = 1.5

nvcc = {_nvcc()}
nvcc_flags = {nvcc_flags}

{_NINJA_RULES}
{compiles}
build {library}: nvcc_link {" ".join(str(obj) for obj, _ in objects)}
""")
        _jit_cache.run_ninja(build_dir, label, jobs=jobs)
        return library

    return recipe.build(builder)


class _VariantWrapper:
    """Wraps a TVM-FFI function to look like a module with .run()."""
    def __init__(self, fn):
        self._fn = fn

    def run(self, *args):
        return self._fn(*args)


_FMHA_TEMPLATES = (_FMHA_VARLEN_DIR / "fmha_sm100_inst.jinja",
                   _FMHA_VARLEN_DIR / "fmha_sm100_variant_run.cu.jinja")


class FMHAVariantManager:
    """Builds and loads the FMHA variant kernels, one cached library per variant."""

    def __init__(self):
        self._loaded = {}
        self._lock = threading.Lock()
        self._inst_template = None
        self._run_template = None
        self._requested = set()

    def _load_templates(self):
        if self._inst_template is None:
            inst_path, run_path = _FMHA_TEMPLATES
            self._run_template = jinja2.Template(run_path.read_text())
            self._inst_template = jinja2.Template(inst_path.read_text())

    def _recipe(self, variant_name, params):
        nvcc_flags = _get_nvcc_flags(kv_mode=params.get("kv_mode", 0))
        return _recipe(f"fmha_{variant_name}", nvcc_flags, templates=_FMHA_TEMPLATES,
                       params=params), nvcc_flags

    def get_variant(self, dtype_code, qo_tile_size, single_wg,
                    sparse_mode, page_size, split_kv, pack_factor,
                    kv_dtype=None):
        variant_name, params = _variant_key_from_runtime(
            dtype_code, qo_tile_size, single_wg,
            sparse_mode, page_size, split_kv, pack_factor, kv_dtype)
        self._requested.add(variant_name)

        cached = self._loaded.get(variant_name)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._loaded.get(variant_name)
            if cached is not None:
                return cached
            import tvm_ffi
            module = tvm_ffi.load_module(str(self.compile_locked(variant_name, params)))
            self._loaded[variant_name] = _VariantWrapper(getattr(module, f"run_{variant_name}"))
            return self._loaded[variant_name]

    def is_cached(self, variant_name):
        """Whether a variant loads without compiling: a record matches the current sources."""
        recipe, _ = self._recipe(variant_name, _variant_params_from_name(variant_name))
        return recipe.lookup() is not None

    def compile_locked(self, variant_name, params):
        """Return the variant's library, compiling it under its cache lock (the lock every
        process takes) when no record matches. Loads nothing."""
        recipe, nvcc_flags = self._recipe(variant_name, params)

        def sources(build_dir):
            self._load_templates()
            inst_cu = build_dir / f"fmha_sm100_inst_{variant_name}.cu"
            run_cu = build_dir / f"fmha_sm100_run_{variant_name}.cu"
            _jit_cache.write_if_changed(inst_cu, self._inst_template.render(**params))
            _jit_cache.write_if_changed(run_cu, self._run_template.render(**params))
            return [inst_cu, run_cu]

        # The kernel instantiation and the host launcher are independent objects.
        return _build_library(recipe, f"FMHA variant {variant_name}", nvcc_flags, sources, jobs=2)


_variant_manager = FMHAVariantManager()


def requested_fmha_variants():
    """Names of the variants this process has asked ``get_fmha_variant`` for, sorted."""
    return sorted(_variant_manager._requested)


def _build_all(builds, max_workers):
    """Run independent builds at once; each waits on its own ninja, so threads suffice."""
    if builds:
        workers = max_workers or min(len(builds), os.cpu_count() or 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda build: build(), builds))


def prebuild_fmha_variants(variant_names, max_workers=None):
    """Compile the named variants that are not cached yet, in parallel.

    Lazy JIT builds one variant at a time when a call first needs it (about a minute each), so
    a test sweep over a cold cache spends most of its time compiling serially. This builds every
    missing variant at once, one ninja per variant, into the same cache and under the same locks
    as ``get_fmha_variant``, so concurrent processes stay safe. Returns the names compiled.
    """
    manager = _variant_manager
    missing = []
    for name in dict.fromkeys(variant_names):
        params = _variant_params_from_name(name)
        if manager._recipe(name, params)[0].lookup() is None:
            missing.append((name, params))
    if missing:
        logger.info("Pre-building %d FMHA variants", len(missing))
    _build_all([lambda item=item: manager.compile_locked(*item) for item in missing], max_workers)
    return [name for name, _ in missing]


def get_fmha_variant(dtype_code, qo_tile_size, single_wg,
                     sparse_mode, page_size, split_kv, pack_factor,
                     kv_dtype=None):
    """Get a compiled FMHA variant module. Thread-safe, lazy compilation.

    None/'fp8' retains the ordinary cache identity and compiler flags.
    'nvfp4' selects the page-128 packed cache in a separate variant.
    """
    return _variant_manager.get_variant(
        dtype_code, qo_tile_size, single_wg,
        sparse_mode, page_size, split_kv, pack_factor, kv_dtype)


# ============================================================================
# Fixed native modules: plan, sparse top-k, split-KV reductions, indexers
# ============================================================================

# key: (library name, translation units under csrc/, _get_nvcc_flags options, extra flags)
_MODULES = {
    "plan": ("fmha_sm100_plan", ("fmha_sm100_plan.cu",), {}, ""),
    "sparse_topk": ("sparse_topk_select", ("sparse_topk_select.cu",), {}, ""),
    "reduction": ("fmha_sm100_reduction", ("fmha_sm100_reduction.cu",), {}, ""),
    # The device-scale NVFP4 reduction ABI: hidden weak symbols keep it apart from the other.
    "reduction_nvfp4": ("fmha_sm100_reduction_nvfp4_v1", ("fmha_sm100_reduction.cu",), {},
                        " -Xcompiler -fvisibility=hidden"),
    # The ported indexer kernels were validated with IEEE division and denormals.
    "indexer_topk_select": ("indexer_topk_select", ("indexer_topk_select.cu",),
                            {"fast_math": False}, ""),
    "q8kv4_indexer_decode": ("q8kv4_indexer_decode", ("q8kv4_indexer_decode.cu",),
                             {"fast_math": False}, ""),
}
# The modules get_indexer_module loads.
_INDEXER_MODULE_SOURCES = ("indexer_topk_select", "q8kv4_indexer_decode")
_modules = {}
_modules_lock = threading.Lock()


def _module_recipe(key):
    name, sources, options, extra_flags = _MODULES[key]
    nvcc_flags = _get_nvcc_flags(False, **options) + extra_flags
    return _recipe(name, nvcc_flags), nvcc_flags, [_FMHA_VARLEN_DIR / s for s in sources]


def build_module(key):
    """Return the library of one of ``_MODULES``, compiling it when no record matches."""
    recipe, nvcc_flags, sources = _module_recipe(key)
    return _build_library(recipe, f"{recipe.name} module", nvcc_flags, lambda _: sources)


def prebuild_modules(keys=tuple(_MODULES), max_workers=None):
    """Compile the fixed modules that are not cached yet, in parallel, without loading them
    (an image build needs no GPU). Returns the keys compiled."""
    missing = [key for key in keys if _module_recipe(key)[0].lookup() is None]
    _build_all([lambda key=key: build_module(key) for key in missing], max_workers)
    return missing


def _load_module(key):
    module = _modules.get(key)
    if module is None:
        with _modules_lock:
            module = _modules.get(key)
            if module is None:
                import tvm_ffi
                module = tvm_ffi.load_module(str(build_module(key)))
                if key == "sparse_topk":
                    module.sparse_topk_select_init()
                _modules[key] = module
    return module


def get_plan_fn():
    """Get the plan function. JIT compiles on first call."""
    return _load_module("plan")


def get_sparse_topk_module():
    """Get the sparse_topk_select module. JIT compiles on first call."""
    return _load_module("sparse_topk")


def get_reduction_module(nvfp4=False):
    """Load the ordinary or device-scale NVFP4 split-KV reduction ABI."""
    return _load_module("reduction_nvfp4" if nvfp4 else "reduction")


def get_indexer_module(name):
    """Load one of ``_INDEXER_MODULE_SOURCES``. JIT compiles on first call."""
    if name not in _INDEXER_MODULE_SOURCES:
        raise KeyError(f"unknown indexer module {name!r}")
    return _load_module(name)
