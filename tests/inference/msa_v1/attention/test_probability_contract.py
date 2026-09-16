"""Exercise production probability primitives against the decoded FlashInfer formula.

The independent oracle follows the SM103 sparse FP8 cubin used by local FlashInfer
0.6.18 (553c2280a88581c7ede39aa571a54d1232854b81): FP32 fused bias, fused score
scaling, hardware EX2, then SATFINITE E4M3 conversion.
"""

import ctypes
import importlib.metadata
import logging
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from packaging.version import Version

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cute.runtime import from_dlpack
import torch

from inference.msa_v1._build_utils import require_cuda_version
from inference.msa_v1.attention.prefill._common.dsl.softmax import (
    SoftmaxSm100 as InferSoftmax,
)
from msa_v1._common.softmax import SoftmaxSm100 as TrainSoftmax
from msa_v1.attention.bwd.qat import scale_apply_exp2_fake_quant_e4m3
from msa_v1.attention.fwd.qat import SparseAttentionQatForwardMixin

pytestmark = pytest.mark.gpu


@contextmanager
def _kernel_run():
    watchdog = threading.Timer(30.0, lambda: os._exit(124))
    watchdog.start()
    try:
        yield
    finally:
        watchdog.cancel()


class ProbabilityProbe:
    def __init__(self, qat):
        self.qat = qat

    @cute.jit
    def __call__(self, scores, maxima, scales, output, unquantized):
        self.kernel(scores, maxima, scales, output, unquantized).launch(
            grid=(cute.ceil_div(scores.shape[0], 32), 1, 1), block=(32, 1, 1)
        )

    @cute.kernel
    def kernel(
        self,
        scores: cute.Tensor,
        maxima: cute.Tensor,
        scales: cute.Tensor,
        output: cute.Tensor,
        unquantized: cute.Tensor,
    ):
        row = cute.arch.block_idx()[0] * 32 + cute.arch.thread_idx()[0]
        if row < scores.shape[0]:
            values = cute.make_rmem_tensor(128, Float32)
            p = cute.make_rmem_tensor(
                128, cutlass.BFloat16 if const_expr(self.qat) else cutlass.Float8E4M3FN
            )
            for col in cutlass.range_constexpr(128):
                values[col] = scores[row, col]
            if const_expr(self.qat):
                softmax = TrainSoftmax.create(scales[row])
            else:
                softmax = InferSoftmax.create(scales[row])
            softmax.scale_subtract_rowmax(values, maxima[row], fp8_probability=True)
            if const_expr(self.qat):
                softmax.apply_exp2_convert(
                    values,
                    p,
                    sparse_attn_p_mode=True,
                    ex2_emu_freq=0,
                    ex2_emu_start_frg=1,
                )
            else:
                softmax.apply_exp2_convert(
                    values,
                    p,
                    ex2_emu_freq=0,
                    ex2_emu_start_frg=1,
                )
            # Subword conversions require packed vectors in the minimum DSL version.
            p_output = cute.make_rmem_tensor(128, cutlass.Float8E4M3FN)
            if const_expr(self.qat):
                p_output.store(p.load().to(Float32).to(cutlass.Float8E4M3FN))
            else:
                p_output.store(p.load())
            for col in cutlass.range_constexpr(128):
                output[row, col] = p_output[col]
                unquantized[row, col] = values[col]


class BackwardProbabilityProbe(SparseAttentionQatForwardMixin):
    def __init__(self):
        self.scale_fp8_p = True

    @cute.jit
    def __call__(self, scores, maxima, scales, quantized, logical):
        self.kernel(scores, maxima, scales, quantized, logical).launch(
            grid=(scores.shape[0] // 4, 4, 1), block=(32, 1, 1)
        )

    @cute.kernel
    def kernel(
        self,
        scores: cute.Tensor,
        maxima: cute.Tensor,
        scales: cute.Tensor,
        quantized: cute.Tensor,
        logical: cute.Tensor,
    ):
        base = cute.arch.block_idx()[0] * 4
        lane = cute.arch.thread_idx()[0]
        col = cute.arch.block_idx()[1] * 32 + lane
        scale_log2 = scales[base]
        shifted_max = self._probability_max_log2(maxima[base + lane % 4], scale_log2)
        values = cute.make_rmem_tensor(4, Float32)
        p = cute.make_rmem_tensor(4, cutlass.BFloat16)
        for row in cutlass.range_constexpr(4):
            values[row] = scores[base + row, col]
        scale_apply_exp2_fake_quant_e4m3(
            values, p, shifted_max, Float32(1.0 / 448.0), scale_log2
        )
        for row in cutlass.range_constexpr(4):
            quantized[base + row, col] = p[row]
            logical[base + row, col] = values[row]


def test_p448_contract(tmp_path):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    assert Version(importlib.metadata.version("nvidia-cutlass-dsl")) >= Version("4.5.2")
    logging.info(
        "DSL %s %s",
        importlib.metadata.version("nvidia-cutlass-dsl"),
        cutlass.__file__,
    )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in {
        (10, 0),
        (10, 3),
    }:
        pytest.skip("P448 acceptance requires SM100 or SM103")
    require_cuda_version((13, 4), component="P448 probability contract")
    root = tmp_path
    source = root / "p448_reference.cu"
    source.write_text(
        r"""
#include <cuda_runtime.h>
#include <cuda_fp8.h>
__device__ float fma_ftz(float a, float b, float c) {
  float r;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__global__ void reference(float const* s, float const* m, float const* scale,
                          unsigned char* p, float* unquantized, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float bias = fma_ftz(-m[i / 128], scale[i / 128], 8.80735492706298828125f);
  float x = fma_ftz(s[i], scale[i / 128], bias);
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(x) : "f"(x));
  unquantized[i] = x;
  p[i] = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
extern "C" int run(float const* s, float const* m, float const* scale,
                    unsigned char* p, float* u, int n) {
  reference<<<(n + 255) / 256, 256>>>(s, m, scale, p, u, n);
  return int(cudaGetLastError());
}
"""
    )
    started = time.perf_counter()
    subprocess.run(
        [
            str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"),
            "-arch=sm_%d%da" % torch.cuda.get_device_capability(),
            "-O3",
            "--shared",
            "-Xcompiler=-fPIC",
            str(source),
            "-o",
            str(root / "p448_reference.so"),
        ],
        check=True,
    )
    logging.info("REFERENCE_COMPILE_SECONDS %.3f", time.perf_counter() - started)
    lib = ctypes.CDLL(str(root / "p448_reference.so"))
    lib.run.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int]
    torch.manual_seed(1701)
    rows = 1024
    scores = torch.randn(rows, 128, device="cuda") * 12
    maxima = scores.max(-1).values + torch.rand(rows, device="cuda") * 4
    scales = torch.exp2(torch.linspace(-12, 8, rows, device="cuda"))
    # Include maxima, masks, zero scale, exact E4M3 midpoints, and underflow.
    scores[:, 0] = maxima
    scores[:, -1] = -torch.inf
    scales[0] = 0.0
    scores[0, -1] = 0.0
    scores[1].fill_(-torch.inf)
    maxima[1] = 0.0
    midpoints = (
        torch.arange(1, 128, device="cuda", dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .float()
    )
    midpoints = (midpoints[:-1] + midpoints[1:]) * 0.5
    midpoints = midpoints[torch.isfinite(midpoints)]
    for offset, direction in enumerate((-1, 0, 1)):
        row = offset + 8
        maxima[row] = 0
        scales[row] = 1
        exponents = torch.log2(midpoints) - float(torch.tensor(448.0).log2())
        if direction:
            exponents = torch.nextafter(
                exponents, torch.full_like(exponents, direction * torch.inf)
            )
        scores[row, : exponents.numel()] = exponents
    # Backward fragments share the attention scale across their query rows.
    scales = scales[::4].repeat_interleave(4)
    scales[:4] = 1.0
    scales[4:8] = 0.0
    scales[8:12] = 1.0
    scores[4:8, -1] = 0.0
    reference = torch.empty_like(scores, dtype=torch.float8_e4m3fn)
    reference_p = torch.empty_like(scores)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with _kernel_run():
        assert (
            lib.run(
                scores.data_ptr(),
                maxima.data_ptr(),
                scales.data_ptr(),
                reference.data_ptr(),
                reference_p.data_ptr(),
                scores.numel(),
            )
            == 0
        )
        torch.cuda.synchronize()
    logging.info("REFERENCE_RUN_MS %.3f", (time.perf_counter() - started) * 1e3)
    for qat in (False, True):
        output = torch.empty_like(reference)
        p = torch.empty_like(scores)
        args = [
            from_dlpack(t)
            for t in (scores, maxima, scales, output.view(torch.uint8), p)
        ]
        args[3].element_type = cutlass.Float8E4M3FN
        started = time.perf_counter()
        fn = cute.compile(ProbabilityProbe(qat), *args)
        logging.info("COMPILE_SECONDS qat=%s %.3f", qat, time.perf_counter() - started)
        for repeat in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            with _kernel_run():
                fn(*args)
                torch.cuda.synchronize()
            logging.info(
                "RUN_MS qat=%s %.3f", qat, (time.perf_counter() - started) * 1e3
            )
            different = int(
                (output.view(torch.uint8) != reference.view(torch.uint8)).sum()
            )
            p_different = int(
                (p.view(torch.int32) != reference_p.view(torch.int32)).sum()
            )
            assert different == 0 and p_different == 0, (
                f"qat={qat}, repeat={repeat}: E4M3 mismatches={different}, "
                f"FP32 mismatches={p_different}"
            )
    quantized = torch.empty_like(scores, dtype=torch.bfloat16)
    logical = torch.empty_like(scores)
    args = [from_dlpack(t) for t in (scores, maxima, scales, quantized, logical)]
    started = time.perf_counter()
    fn = cute.compile(BackwardProbabilityProbe(), *args)
    logging.info("BACKWARD_COMPILE_SECONDS %.3f", time.perf_counter() - started)
    expected_quantized = (reference.float() * (1.0 / 448)).to(torch.bfloat16)
    expected_logical = reference_p * (1.0 / 448)
    for _ in range(3):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with _kernel_run():
            fn(*args)
            torch.cuda.synchronize()
        logging.info("BACKWARD_RUN_MS %.3f", (time.perf_counter() - started) * 1e3)
        assert torch.equal(
            quantized.view(torch.int16), expected_quantized.view(torch.int16)
        )
        assert torch.equal(
            logical.view(torch.int32), expected_logical.view(torch.int32)
        )
    _verify_cpp(root, scores, maxima, scales, reference, reference_p)


def _verify_cpp(root, scores, maxima, scales, reference, reference_p):
    # Extract the current arithmetic verbatim; replace only the surrounding data movement.
    # The independent CUDA oracle above never imports these production helpers.
    repo = Path(__file__).resolve().parents[4]
    base = repo / "inference/msa_v1/attention"
    prefill = base / "prefill/q8kv4/csrc/include"
    decode = base / "decode/q8kv4/csrc/include/sm100"
    pf_source = (prefill / "sm100/common/prefill_attention_math.cuh").read_text()
    de_source = (
        decode / "collective/sm100_fmha_softmax_tma_warpspecialized.hpp"
    ).read_text()
    de_constant = de_source.split("static constexpr float kLog2E4m3Scale =", 1)[
        1
    ].split(";", 1)[0]
    pf_body = pf_source.split("  float2 const packed_scale =", 1)[1].split(
        "  sum0 = add_packed_f32x2(sum0, sum1);", 1
    )[0]
    pf_body = "  float2 const packed_scale =" + pf_body
    de_body = de_source.split("  CUTLASS_DEVICE static void make_p_and_store(", 1)[1]
    de_body = de_body.split("float scale_softmax_log2) {", 1)[1].split(
        "    store_transposed_smem_8b_16x128(", 1
    )[0]
    source = (
        r"""
    #include <cuda_runtime.h>
    #include <cuda_fp8.h>
    #include "sm100/common/prefill_attention_math.cuh"
    #include "sm100_fmha_fp4_transform.cuh"
    namespace pf = minimax::msa_v1::attention::prefill::q8kv4::detail;
    __global__ void prefill_probe(float const* scores, float const* maxima,
                                 float const* scales, unsigned char* out, float* raw) {
      using namespace pf;
      int row = blockIdx.x * blockDim.x + threadIdx.x;
      auto rS = cute::make_tensor<float>(cute::make_shape(cute::Int<128>{}));
      for (int i=0; i<128; ++i) rS(i) = scores[row*128+i];
      float maximum_safe = maxima[row];
      float softmax_scale_log2 = scales[row];
    """
        + pf_body
        + r"""
      for (int i=0; i<32; ++i) reinterpret_cast<uint32_t*>(out)[row*32+i] = rP(i);
      for (int i=0; i<128; ++i) raw[row*128+i] = rS(i);
    }
    __global__ void decode_probe(float const* scores, float const* maxima,
                                float const* scales, unsigned char* out, float* raw) {
      using namespace cutlass::fmha::collective;
      int row = blockIdx.x;
      int col = threadIdx.x * 16;
      float qk[16], new_max[4];
      for (int i=0; i<16; ++i) qk[i] = scores[row*128+col+i];
      for (int i=0; i<4; ++i) new_max[i] = maxima[row];
      float scale_softmax_log2 = scales[row];
      constexpr float kLog2E4m3Scale = PRODUCTION_LOG2_E4M3_SCALE;
      using PackedPFragment = uint32_t[4];
    """
        + de_body
        + r"""
      for (int i=0; i<4; ++i) reinterpret_cast<uint32_t*>(out)[row*32+col/4+i] = regs_p[i];
      for (int i=0; i<16; ++i) raw[row*128+col+i] = qk[i];
    }
    extern "C" int run(int decode, float const* s, float const* m, float const* scale,
                       unsigned char* p, float* raw, int rows) {
      if (decode) decode_probe<<<rows, 8>>>(s,m,scale,p,raw);
      else prefill_probe<<<rows/32, 32>>>(s,m,scale,p,raw);
      return int(cudaGetLastError());
    }
    """
    )
    path = root / "cpp_p448_probe.cu"
    path.write_text(source.replace("PRODUCTION_LOG2_E4M3_SCALE", de_constant))
    for mode in (0, 1):
        library = root / f"cpp_p448_probe_{mode}.so"
        started = time.perf_counter()
        command = [
            str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"),
            "-arch=sm_%d%da" % torch.cuda.get_device_capability(),
            "-O3",
            "-std=c++20",
            "--shared",
            "-Xcompiler=-fPIC",
            "--expt-relaxed-constexpr",
            "-DMINIMAX_MSA_Q8KV4_HAS_QMUL4=1",
            "-I" + str(repo / "third_party/cutlass/include"),
            "-I" + str(prefill),
            "-I" + str(decode / "collective"),
            "-I" + str(decode / "common"),
            str(path),
            "-o",
            str(library),
        ]
        if mode == 1:
            command.append("-use_fast_math")
        subprocess.run(command, check=True)
        logging.info("CPP_COMPILE_SECONDS %.3f", time.perf_counter() - started)
        lib = ctypes.CDLL(str(library))
        lib.run.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 5 + [ctypes.c_int]
        out = torch.empty_like(reference)
        raw = torch.empty_like(reference_p)
        for repeat in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            with _kernel_run():
                assert (
                    lib.run(
                        mode,
                        scores.data_ptr(),
                        maxima.data_ptr(),
                        scales.data_ptr(),
                        out.data_ptr(),
                        raw.data_ptr(),
                        scores.shape[0],
                    )
                    == 0
                )
                torch.cuda.synchronize()
            logging.info("CPP_RUN_MS %.3f", (time.perf_counter() - started) * 1e3)
            differences = (
                (out.view(torch.uint8) != reference.view(torch.uint8)).sum().item()
            )
            raw_differences = (
                (raw.view(torch.int32) != reference_p.view(torch.int32)).sum().item()
            )
            assert differences == 0 and raw_differences == 0, (
                f"decode={mode}, repeat={repeat}: E4M3 mismatches={differences}, "
                f"FP32 mismatches={raw_differences}"
            )
