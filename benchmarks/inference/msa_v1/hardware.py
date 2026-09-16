"""Public peak specifications used only for MBU/MFU diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PeakSpec:
    hbm_tb_s: float
    dense_fp8_tflops: float
    dense_bf16_tflops: float
    source: str


_DENSE_FP8_FMAS_PER_CYCLE_PER_SM = 8192
_DENSE_BF16_FMAS_PER_CYCLE_PER_SM = 4096
_PEAKS = {
    "NVIDIA B200": PeakSpec(
        hbm_tb_s=8.0,
        dense_fp8_tflops=(_DENSE_FP8_FMAS_PER_CYCLE_PER_SM * 148 * 1.83e9 * 2 / 1.0e12),
        dense_bf16_tflops=(
            _DENSE_BF16_FMAS_PER_CYCLE_PER_SM * 148 * 1.83e9 * 2 / 1.0e12
        ),
        source="public B200/GB200 SM100 specification",
    ),
    "NVIDIA GB200": PeakSpec(
        hbm_tb_s=8.0,
        dense_fp8_tflops=(_DENSE_FP8_FMAS_PER_CYCLE_PER_SM * 148 * 1.83e9 * 2 / 1.0e12),
        dense_bf16_tflops=(
            _DENSE_BF16_FMAS_PER_CYCLE_PER_SM * 148 * 1.83e9 * 2 / 1.0e12
        ),
        source="public B200/GB200 SM100 specification",
    ),
    "NVIDIA GB300": PeakSpec(
        hbm_tb_s=8.0,
        dense_fp8_tflops=(_DENSE_FP8_FMAS_PER_CYCLE_PER_SM * 152 * 2.07e9 * 2 / 1.0e12),
        dense_bf16_tflops=(
            _DENSE_BF16_FMAS_PER_CYCLE_PER_SM * 152 * 2.07e9 * 2 / 1.0e12
        ),
        source="public SM100 rates with GB300 device SM count and maximum SM clock",
    ),
    # B300 SXM6 (air-cooled, x86 host): device attributes report 148 SMs, 2.032 GHz max SM
    # clock, HBM3e 7680-bit x 3.996 GHz DDR = 7.67 TB/s (the public 8.0 TB/s figure assumes
    # the full 8192-bit bus). Tensor rates use the public SM100 per-SM rates.
    "NVIDIA B300 SXM6 AC": PeakSpec(
        hbm_tb_s=7.67,
        dense_fp8_tflops=(
            _DENSE_FP8_FMAS_PER_CYCLE_PER_SM * 148 * 2.032e9 * 2 / 1.0e12
        ),
        dense_bf16_tflops=(
            _DENSE_BF16_FMAS_PER_CYCLE_PER_SM * 148 * 2.032e9 * 2 / 1.0e12
        ),
        source="public SM100 rates with B300 SXM6 device SM count, clock and memory bus",
    ),
    # Rubin (SM107) engineering sample reports the generic pre-release name.
    # Provisional: HBM peak is the DRAM interface peak of this sample (15360-bit
    # bus, DDR at the 3.186 GHz maximum memory clock = 3840 B/cycle = 12.23 TB/s,
    # which is also the Nsight Compute DRAM peak). The 4.995 GHz reported by
    # cudaDevAttrMemoryClockRate overstates it (19.2 TB/s) and is not used.
    # Tensor Core peak assumes 2x the SM100 per-SM FP8 rate and 1x the SM100
    # per-SM BF16 rate at 208 SMs / 2.424 GHz.
    "NVIDIA Graphics Device": PeakSpec(
        hbm_tb_s=12.23,
        dense_fp8_tflops=(
            2 * _DENSE_FP8_FMAS_PER_CYCLE_PER_SM * 208 * 2.424e9 * 2 / 1.0e12
        ),
        dense_bf16_tflops=(
            _DENSE_BF16_FMAS_PER_CYCLE_PER_SM * 208 * 2.424e9 * 2 / 1.0e12
        ),
        source="provisional SM107 estimate (ncu DRAM peak, SM100-scaled tensor rates); not a public specification",
    ),
}


def peak_spec(device_name: str) -> PeakSpec:
    try:
        return _PEAKS[device_name]
    except KeyError as error:
        raise ValueError(
            f"no public MBU/MFU peak specification is registered for {device_name!r}"
        ) from error


def roofline_metrics(
    *,
    device_name: str,
    useful_flops: int,
    logical_bytes: int,
    latency_us: float,
    compute_dtype: str = "fp8",
) -> dict[str, float | str]:
    """Return logical-work MBU/MFU diagnostics without changing E2E acceptance."""

    spec = peak_spec(device_name)
    if compute_dtype == "fp8":
        peak_tflops = spec.dense_fp8_tflops
    elif compute_dtype == "bf16":
        peak_tflops = spec.dense_bf16_tflops
    else:
        raise ValueError("compute_dtype must be 'fp8' or 'bf16'")
    observed_tflops = useful_flops / latency_us / 1.0e6
    observed_tb_s = logical_bytes / latency_us / 1.0e6
    arithmetic_intensity = useful_flops / logical_bytes
    ridge_point = peak_tflops / spec.hbm_tb_s
    bound = "memory" if arithmetic_intensity < ridge_point else "compute"
    mbu = observed_tb_s / spec.hbm_tb_s
    mfu = observed_tflops / peak_tflops
    return {
        "peak_hbm_tb_s": spec.hbm_tb_s,
        f"peak_dense_{compute_dtype}_tflops": peak_tflops,
        "peak_spec_source": spec.source,
        "logical_arithmetic_intensity_flops_per_byte": arithmetic_intensity,
        "roofline_ridge_flops_per_byte": ridge_point,
        "roofline_bound": bound,
        "logical_mbu": mbu,
        "logical_mfu": mfu,
        "roofline_efficiency": mbu if bound == "memory" else mfu,
    }


__all__ = ["PeakSpec", "peak_spec", "roofline_metrics"]
