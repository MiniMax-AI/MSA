"""Canonical RL-rollout benchmark cases for MSA v1 decode operators."""

from __future__ import annotations

from dataclasses import dataclass

import torch

DEFAULT_SEED = 1701
BATCH_SIZES = (8, 32, 64, 128)
SEQ_LENGTHS = (1_000, 4_000, 5_000, 10_000, 50_000, 100_000, 200_000)
LOW_LATENCY_BATCH_SIZES = (1, 2, 4)
LOW_LATENCY_SEQ_LENGTHS = (1_000, 4_000, 8_000, 32_000)
Q_LEN_PER_REQ = 8

_WEIGHTS = {
    1_000: (30, 90, 180, 300),
    4_000: (40, 120, 240, 400),
    5_000: (50, 150, 300, 500),
    10_000: (140, 350, 490, 420),
    50_000: (500, 800, 500, 200),
    100_000: (1260, 980, 420, 140),
    200_000: (910, 350, 112, 28),
}


@dataclass(frozen=True)
class DecodeBenchmarkCase:
    batch_size: int
    nominal_seq_len: int
    weight: int
    q_len_per_req: int = Q_LEN_PER_REQ
    seed: int = DEFAULT_SEED

    @property
    def name(self) -> str:
        return f"b{self.batch_size}_s{self.nominal_seq_len}_q{self.q_len_per_req}"

    @property
    def batch(self) -> int:
        """Compatibility spelling used by the existing benchmark storage."""

        return self.batch_size

    @property
    def hard_gate(self) -> bool:
        return True


FULL_CASES = tuple(
    DecodeBenchmarkCase(
        batch_size=batch_size,
        nominal_seq_len=seq_len,
        weight=_WEIGHTS[seq_len][batch_index],
    )
    for seq_len in SEQ_LENGTHS
    for batch_index, batch_size in enumerate(BATCH_SIZES)
)
# Zero weight keeps this independent latency suite out of production aggregates.
LOW_LATENCY_CASES = tuple(
    DecodeBenchmarkCase(batch_size=batch_size, nominal_seq_len=seq_len, weight=0)
    for seq_len in LOW_LATENCY_SEQ_LENGTHS
    for batch_size in LOW_LATENCY_BATCH_SIZES
)
SMOKE_CASES = tuple(
    case
    for case in FULL_CASES
    if case.nominal_seq_len in (1_000, 100_000, 200_000) and case.batch_size in (8, 128)
)


def benchmark_cases(suite: str) -> tuple[DecodeBenchmarkCase, ...]:
    if suite == "smoke":
        return SMOKE_CASES
    if suite == "full":
        return FULL_CASES
    if suite == "low-latency":
        return LOW_LATENCY_CASES
    raise ValueError("suite must be 'smoke', 'full', or 'low-latency'")


def make_seq_lens(case: DecodeBenchmarkCase) -> torch.Tensor:
    """Return deterministic lengths; a singleton uses the exact nominal length."""

    generator = torch.Generator().manual_seed(
        case.seed + case.batch_size * 1009 + case.nominal_seq_len
    )
    pair_count = case.batch_size // 2
    minimum = max(1, case.nominal_seq_len // 20)
    maximum = max(minimum, case.nominal_seq_len // 4)
    deltas = torch.randint(
        minimum,
        maximum + 1,
        (pair_count,),
        generator=generator,
        dtype=torch.int64,
    )
    parts = [case.nominal_seq_len - deltas, case.nominal_seq_len + deltas]
    if case.batch_size % 2:
        # Odd batch: one request at the nominal length keeps the mean exact.
        parts.append(torch.tensor([case.nominal_seq_len], dtype=torch.int64))
    values = torch.cat(parts)
    values = values[torch.randperm(case.batch_size, generator=generator)]
    assert int(values.sum()) == case.batch_size * case.nominal_seq_len
    assert int(values.min()) >= case.q_len_per_req
    return values.to(torch.int32)


assert len(FULL_CASES) == 28
assert sum(case.weight for case in FULL_CASES) == 10_000
assert all(case.weight > 0 for case in FULL_CASES)


__all__ = [
    "BATCH_SIZES",
    "DEFAULT_SEED",
    "FULL_CASES",
    "LOW_LATENCY_BATCH_SIZES",
    "LOW_LATENCY_CASES",
    "LOW_LATENCY_SEQ_LENGTHS",
    "Q_LEN_PER_REQ",
    "SEQ_LENGTHS",
    "SMOKE_CASES",
    "DecodeBenchmarkCase",
    "benchmark_cases",
    "make_seq_lens",
]
