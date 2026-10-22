"""MiniMax-M3.1 DSpark draft backbone on TP4: the vocab-parallel embedding, the five dense
decoder layers and the final norm of one request's seven-row draft block in one launch per rank."""

import hashlib
import importlib.util
import os
import shutil
import socket
import subprocess
import sys
import sysconfig
import threading
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.distributed as dist

from ..common import jit as _jit

LAYERS = 5
TP = 4
ROWS = 7
HIDDEN = 6144
HEAD_DIM = 128
ROTARY_DIM = 64
LOCAL_Q_HEADS = 16
QKV_ROWS = LOCAL_Q_HEADS * HEAD_DIM + 2 * HEAD_DIM  # 2304: q | k | v
O_PROJ_K = LOCAL_Q_HEADS * HEAD_DIM  # 2048
INTER = 3072  # per rank
WINDOW = 4096  # keys a row attends (HF sliding_window 4096, window_left 4095)
PAGE = 128
SCALE_BLOCK = 32

_SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "csrc" / "draft_backbone"
_SOURCES = ("api/draft_backbone_api.cpp", "src/draft_backbone.cu")
# torch >= 2.14 headers require C++20
_CPP_FLAGS = ("-O3", "-std=c++20")
# The kernel follows the exact-IEEE contract except where its code requests
# approximate operations explicitly.
_CUDA_FLAGS = (
    "-O3",
    "-std=c++20",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "-gencode=arch=compute_100a,code=sm_100a",
    "-gencode=arch=compute_103a,code=sm_103a",
)
_modules = {}
_module_lock = threading.Lock()


def _toolchain_identity():
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError("the draft backbone requires a CUDA toolkit")
    compiler = Path(CUDA_HOME) / "bin" / "nvcc"
    host_compiler = shutil.which(os.environ.get("CXX", "c++"))
    if host_compiler is None:
        raise RuntimeError("the draft backbone requires a C++ compiler")
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
    for value in (*identity, *_CPP_FLAGS, *_CUDA_FLAGS):
        digest.update(value.encode() + b"\0")
    for path in sorted(p for p in _SOURCE_DIRECTORY.rglob("*") if p.is_file()):
        digest.update(path.relative_to(_SOURCE_DIRECTORY).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _import_library(name, library):
    specification = importlib.util.spec_from_file_location(name, library)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load the draft backbone extension {library}")
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

    name = "msa_v1_draft_backbone_" + digest[:20]
    cache_directory = _jit.CACHE_BASE / "draft_backbone" / digest
    library = cache_directory / (name + ".so")
    with _jit._cache_file_lock(name + ".compile"):
        if _jit._cached_variant_is_current(library, digest):
            return _import_library(name, library)
        if "ALLOW_JIT" not in os.environ:
            raise RuntimeError(
                "JIT disabled and a current draft backbone kernel was not found: "
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
            build_directory=str(cache_directory),
            with_cuda=True,
            verbose=False,
        )
        if not _jit._shared_object_is_usable(library):
            raise RuntimeError("draft backbone build produced an invalid shared object")
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


class DraftTpContext:
    """Collectively created private IPC arena, shared only by sequential plans.

    Call close on every rank after all uses and graphs have finished.
    """

    def __init__(self, group=None):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("context creation is not allowed during capture")
        if not dist.is_initialized() or dist.get_world_size(group) != TP:
            raise ValueError("an initialized single-node TP4 process group is required")
        self.group = group
        self.rank = dist.get_rank(group)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.closed = False
        self.module = None
        self.native = None
        error = None
        try:
            self.module = _get_module()
            self.native = self.module.Workspace(self.device.index, self.rank)
        except Exception as exception:
            error = str(exception)
        if errors := _gather_errors(self.group, error):
            if self.native is not None:
                self.native.close()
            raise RuntimeError("draft backbone workspace creation failed: " + str(errors))
        peers = [None] * TP
        dist.all_gather_object(
            peers,
            (socket.gethostname(), self.native.device_uuid(), self.native.ipc_handle()),
            group=self.group,
        )
        if len({peer[0] for peer in peers}) != 1:
            raise ValueError("the draft backbone requires all TP ranks on one node")
        if len({peer[1] for peer in peers}) != TP:
            raise ValueError("the draft backbone requires four distinct CUDA devices")
        try:
            self.native.connect([peer[2] for peer in peers])
        except Exception as exception:
            error = str(exception)
        if errors := _gather_errors(self.group, error):
            self.native.close_imports()
            dist.barrier(group=self.group)
            self.native.close()
            raise RuntimeError("draft backbone IPC connection failed: " + str(errors))

    def close(self):
        """Collectively close peer mappings first, then free every owner arena."""
        if self.closed:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("context destruction is not allowed during capture")
        with torch.cuda.device(self.device):
            torch.cuda.synchronize()
            dist.barrier(group=self.group)
            error = None
            try:
                self.native.close_imports()
            except Exception as exception:
                error = str(exception)
            if errors := _gather_errors(self.group, error):
                raise RuntimeError("draft backbone IPC close failed: " + str(errors))
            dist.barrier(group=self.group)
            self.native.close()
        self.closed = True


def create_draft_tp_context(group=None) -> DraftTpContext:
    """Collectively create the TP4 context; all four ranks call together."""
    return DraftTpContext(group)


@dataclass
class DraftLayerWeights:
    """One decoder layer's rank-local operands, checkpoint (row-major) layouts."""

    input_norm: torch.Tensor  # bf16 [6144]: raw Gemma w of input_layernorm (applied as 1 + w)
    post_norm: torch.Tensor  # bf16 [6144]: raw Gemma w of post_attention_layernorm
    q_norm: torch.Tensor  # bf16 [128]: raw w of q_norm
    k_norm: torch.Tensor  # bf16 [128]: raw w of k_norm
    qkv: torch.Tensor  # e4m3 [2304, 6144]: q (16 heads) | k | v rows of this rank
    qkv_sf: torch.Tensor  # u8 [2304, 192]: UE8M0 per 32 K-values
    o: torch.Tensor  # e4m3 [6144, 2048]
    o_sf: torch.Tensor  # u8 [6144, 64]
    gate_up: torch.Tensor  # e4m3 [6144, 6144]: gate 3072 | up 3072 rows of this rank
    gate_up_sf: torch.Tensor  # u8 [6144, 192]
    down: torch.Tensor  # e4m3 [6144, 3072]
    down_sf: torch.Tensor  # u8 [6144, 96]

    def tensors(self):
        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class DraftBackboneWeights:
    layers: list  # LAYERS x DraftLayerWeights
    embed: torch.Tensor  # bf16 [vocab_rows, 6144]: this rank's vocab shard
    vocab_start: int  # first token id of the shard
    final_norm: torch.Tensor  # bf16 [6144]: raw Gemma w of the final norm
    eps: float


@dataclass
class DraftBackboneInputs:
    """Graph-stable step inputs, rewritten in place before every replay."""

    input_ids: torch.Tensor  # int64 [7]
    positions: torch.Tensor  # int64 [7]
    slot_mapping: torch.Tensor  # int64 [7]: cache slot of each row, -1 skips the write
    page_table: torch.Tensor  # int32 [max_pages]
    kv_len: torch.Tensor  # int32 [1]: cached tokens including the block
    cos_sin: torch.Tensor  # f32 [positions, 64]: cos | sin
    k_caches: list  # LAYERS x bf16 [slots, 128]
    v_caches: list  # LAYERS x bf16 [slots, 128]


class DraftBackbonePlan:
    """Bound tensors for one fixed set of step buffers; ``run`` is graph-capturable.

    ``out`` (bf16 [7, 6144]) receives the final-normed hidden, identical on every
    rank, and every call writes the block's K/V rows into the caches.
    Plans of one context run in stream order.
    """

    def __init__(self, context: DraftTpContext, weights: DraftBackboneWeights,
                 inputs: DraftBackboneInputs, out: torch.Tensor):
        if context.closed:
            raise RuntimeError("draft backbone context is closed")
        if len(weights.layers) != LAYERS:
            raise ValueError(f"the draft backbone has {LAYERS} layers")
        self.context = context
        layers = [
            layer.tensors() + [inputs.k_caches[i], inputs.v_caches[i]]
            for i, layer in enumerate(weights.layers)
        ]
        step = [
            inputs.input_ids,
            weights.embed,
            out,
            weights.final_norm,
            inputs.positions,
            inputs.slot_mapping,
            inputs.page_table,
            inputs.kv_len,
            inputs.cos_sin,
        ]
        self.native = context.module.Plan(
            context.native, layers, step, int(weights.vocab_start), float(weights.eps)
        )
        self.out = out

    def run(self) -> torch.Tensor:
        if self.context.closed:
            raise RuntimeError("draft backbone context is closed")
        self.native.run()
        return self.out
