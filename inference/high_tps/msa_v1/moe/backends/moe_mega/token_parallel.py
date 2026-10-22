"""Token-parallel single-kernel MoE for Q8 on TP4."""

import hashlib
import importlib.util
import os
import shutil
import socket
import subprocess
import sys
import sysconfig
import threading
from pathlib import Path

import torch
import torch.distributed as dist

from ....common import jit as _jit

_SOURCE_DIRECTORY = Path(__file__).resolve().parents[3] / "csrc" / "moe_token_parallel"
_SOURCES = ("api/moe_token_parallel_api.cpp", "src/moe_token_parallel.cu")
# torch >= 2.14 headers require C++20
_CPP_FLAGS = ("-O3", "-std=c++20")
# The kernel follows the exact-IEEE contract except where its code requests
# approximate or flush-to-zero operations explicitly.
_CUDA_FLAGS = (
    "-O3",
    "-std=c++20",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "-gencode=arch=compute_100a,code=sm_100a",
    "-gencode=arch=compute_103a,code=sm_103a",
)
_LINK_FLAGS = ("-lcuda",)
_modules = {}
_module_lock = threading.Lock()


def _toolchain_identity():
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError("token-parallel MoE requires a CUDA toolkit")
    compiler = Path(CUDA_HOME) / "bin" / "nvcc"
    host_compiler = shutil.which(os.environ.get("CXX", "c++"))
    if host_compiler is None:
        raise RuntimeError("token-parallel MoE requires a C++ compiler")
    versions = [
        subprocess.run([str(tool), "--version"], capture_output=True, text=True, check=True).stdout
        for tool in (compiler, host_compiler)
    ]
    return (
        str(compiler.resolve()),
        versions[0],
        str(Path(host_compiler).resolve()),
        versions[1],
        str(sysconfig.get_config_var("SOABI")),
        str(torch.__version__),
        str(torch.version.cuda),
        str(torch._C._GLIBCXX_USE_CXX11_ABI),
    )


def _build_digest(identity):
    """Invalidate the native module for every source, flag and toolchain change."""
    digest = hashlib.sha256()
    for value in (*identity, *_CPP_FLAGS, *_CUDA_FLAGS, *_LINK_FLAGS):
        digest.update(value.encode() + b"\0")
    for path in sorted(p for p in _SOURCE_DIRECTORY.rglob("*") if p.is_file()):
        digest.update(path.relative_to(_SOURCE_DIRECTORY).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _import_library(name, library):
    specification = importlib.util.spec_from_file_location(name, library)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load the token-parallel MoE extension {library}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _load_or_build(digest):
    from torch.utils.cpp_extension import load

    name = "msa_v1_moe_token_parallel_" + digest[:20]
    cache_directory = _jit.CACHE_BASE / "moe_token_parallel" / digest
    library = cache_directory / (name + ".so")
    with _jit._cache_file_lock(name + ".compile"):
        if _jit._cached_variant_is_current(library, digest):
            return _import_library(name, library)
        if "ALLOW_JIT" not in os.environ:
            raise RuntimeError(
                "JIT disabled and a current token-parallel MoE kernel was not found: "
                f"{library}. The cached binary is missing, invalid, or was built from "
                "different kernel sources."
            )
        fingerprint = cache_directory / ".build_fingerprint"
        cache_directory.mkdir(parents=True, exist_ok=True)
        library.unlink(missing_ok=True)
        fingerprint.unlink(missing_ok=True)
        module = load(
            name=name,
            sources=[str(_SOURCE_DIRECTORY / source) for source in _SOURCES],
            extra_include_paths=[str(_SOURCE_DIRECTORY)],
            extra_cflags=list(_CPP_FLAGS),
            extra_cuda_cflags=list(_CUDA_FLAGS),
            extra_ldflags=list(_LINK_FLAGS),
            build_directory=str(cache_directory),
            with_cuda=True,
            verbose=False,
        )
        if not _jit._shared_object_is_usable(library):
            raise RuntimeError("token-parallel MoE build produced an invalid shared object")
        staged = cache_directory / f".build_fingerprint.{os.getpid()}.tmp"
        staged.write_text(digest + "\n")
        os.replace(staged, fingerprint)
        return module


def _get_module():
    digest = _build_digest(_toolchain_identity())
    with _module_lock:
        if digest not in _modules:
            _modules[digest] = _load_or_build(digest)
        return _modules[digest]


def _gather_errors(group, error):
    errors = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, error, group=group)
    return [item for item in errors if item]


class Workspace:
    """Context-owned private IPC arena, shared only by sequential plans."""

    def __init__(self, context):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("workspace creation is not allowed during capture")
        self.device = context.device
        self.group = context.group
        self.closed = False
        self.module = None
        self.native = None
        error = None
        try:
            self.module = _get_module()
            self.native = self.module.Workspace(self.device.index, context.rank)
        except Exception as exception:  # noqa: BLE001 - propagate failure to every rank
            error = str(exception)
        if errors := _gather_errors(self.group, error):
            if self.native is not None:
                self.native.close()
            raise RuntimeError("token-parallel MoE workspace creation failed: " + str(errors))
        peers = [None] * 4
        dist.all_gather_object(
            peers,
            (socket.gethostname(), self.native.device_uuid(), self.native.ipc_handle()),
            group=self.group,
        )
        if len({peer[0] for peer in peers}) != 1:
            raise ValueError("token-parallel MoE requires all TP ranks on one node")
        if len({peer[1] for peer in peers}) != 4:
            raise ValueError("token-parallel MoE requires four distinct CUDA devices")
        try:
            self.native.connect([peer[2] for peer in peers])
        except Exception as exception:  # noqa: BLE001 - propagate failure to every rank
            error = str(exception)
        if errors := _gather_errors(self.group, error):
            self.native.close_imports()
            dist.barrier(group=self.group)
            self.native.close()
            raise RuntimeError("token-parallel MoE IPC connection failed: " + str(errors))

    def close(self):
        """Collectively close peer mappings first, then free every owner arena."""
        if self.closed:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("workspace destruction is not allowed during capture")
        with torch.cuda.device(self.device):
            torch.cuda.synchronize()
            dist.barrier(group=self.group)
            error = None
            try:
                self.native.close_imports()
            except Exception as exception:  # noqa: BLE001 - propagate failure to every rank
                error = str(exception)
            if errors := _gather_errors(self.group, error):
                raise RuntimeError("token-parallel MoE IPC close failed: " + str(errors))
            dist.barrier(group=self.group)
            self.native.close()
        self.closed = True


class TokenParallelPlan:
    """Bound tensors and immutable launch metadata for one fixed Q8 input."""

    def __init__(self, workspace, weights, fc2_alpha, hidden, residual, output):
        self.workspace = workspace
        self.native = workspace.module.Plan(
            workspace.native,
            hidden,
            residual,
            output,
            weights.router_weight,
            weights.router_correction_bias,
            weights.shared_fc1_weight,
            weights.shared_fc1_weight_sf,
            weights.shared_fc2_weight,
            weights.shared_fc2_weight_sf,
            weights.routed_fc1_weight,
            weights.routed_fc1_weight_sf,
            weights.routed_fc2_weight,
            weights.routed_fc2_weight_sf,
            weights.routed_fc1_alpha,
            fc2_alpha,
        )

    def run(self):
        if self.workspace.closed:
            raise RuntimeError("token-parallel MoE workspace is closed")
        self.native.run()
