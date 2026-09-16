"""Deterministic canonical correctness cases for MSA v1 decode operators."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

DEFAULT_SEED = 1701
FULL_CASE_COUNT = 256
SMOKE_CASE_COUNT = 96
PRODUCTION_CASE_COUNT = FULL_CASE_COUNT * 3 // 4
ROBUSTNESS_CASE_COUNT = FULL_CASE_COUNT - PRODUCTION_CASE_COUNT
Q_LENGTHS = (1, 2, 4, 8, 16)
MAX_SEQ_LEN = 512 * 1024

_SEQ_BUCKETS = (
    (1_000, 6),
    (4_000, 8),
    (5_000, 10),
    (10_000, 14),
    (50_000, 20),
    (100_000, 28),
    (200_000, 14),
)
_BATCH_ANCHORS = {
    8: (1, 2, 4, 8, 16),
    32: (24, 32, 48),
    64: (64, 96),
    128: (128, 192, 256, 384, 512),
}
_BATCH_GROUP_WEIGHTS = {
    1_000: (5, 15, 30, 50),
    4_000: (5, 15, 30, 50),
    5_000: (5, 15, 30, 50),
    10_000: (10, 25, 35, 30),
    50_000: (25, 40, 25, 10),
    100_000: (45, 35, 15, 5),
    200_000: (65, 25, 8, 2),
}
_Q_LENGTH_CYCLE = (1, 1, 2, 2, 4, 4, 4) + (8,) * 10 + (16, 16, 16)
_PAGE_LAYOUTS = ("disjoint", "permuted", "shared_prefix")
_ROBUST_BATCHES = (1, 2, 7, 8, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512)
_ROBUST_SEQ_LENS = (
    1_000,
    1_023,
    1_024,
    1_025,
    4_095,
    4_096,
    4_097,
    9_999,
    10_000,
    10_001,
    99_999,
    100_000,
    100_001,
    199_999,
    200_000,
    200_001,
    MAX_SEQ_LEN - 1,
    MAX_SEQ_LEN,
)


@dataclass(frozen=True)
class DecodeCorrectnessCase:
    """One shared shape/metadata workload; dtype payloads remain op-specific."""

    case_id: str
    batch_size: int
    q_len_per_req: int
    nominal_seq_len: int
    distribution: str
    page_layout: str
    seed: int
    production: bool
    deterministic: bool = True


def _weighted_cycle(entries: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    return tuple(value for value, weight in entries for _ in range(weight))


def _select_weighted(value: int, weights: tuple[int, ...]) -> int:
    slot = value % sum(weights)
    total = 0
    for index, weight in enumerate(weights):
        total += weight
        if slot < total:
            return index
    raise AssertionError("weighted selection did not terminate")


def _case_id(
    *,
    index: int,
    batch_size: int,
    q_len_per_req: int,
    nominal_seq_len: int,
    distribution: str,
    page_layout: str,
    production: bool,
) -> str:
    payload = (
        f"decode-v1:{index}:{batch_size}:{q_len_per_req}:{nominal_seq_len}:"
        f"{distribution}:{page_layout}:{int(production)}:{DEFAULT_SEED}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"decode_{index:04d}_{digest}"


def _production_cases() -> tuple[DecodeCorrectnessCase, ...]:
    seq_cycle = _weighted_cycle(_SEQ_BUCKETS)
    batch_groups = tuple(_BATCH_ANCHORS)
    cases = []
    for index in range(PRODUCTION_CASE_COUNT):
        nominal_seq_len = seq_cycle[index % len(seq_cycle)]
        group_index = _select_weighted(
            index * 37 + DEFAULT_SEED,
            _BATCH_GROUP_WEIGHTS[nominal_seq_len],
        )
        group = batch_groups[group_index]
        anchors = _BATCH_ANCHORS[group]
        batch_size = anchors[(index * 11 + nominal_seq_len) % len(anchors)]
        q_len_per_req = _Q_LENGTH_CYCLE[index % len(_Q_LENGTH_CYCLE)]
        page_layout = _PAGE_LAYOUTS[index % len(_PAGE_LAYOUTS)]
        if (
            page_layout == "disjoint"
            and batch_size * ((nominal_seq_len + 127) // 128) > 8192
        ):
            page_layout = "permuted"
        distribution = ("narrow", "bimodal", "long_tail")[index % 3]
        cases.append(
            DecodeCorrectnessCase(
                case_id=_case_id(
                    index=index,
                    batch_size=batch_size,
                    q_len_per_req=q_len_per_req,
                    nominal_seq_len=nominal_seq_len,
                    distribution=distribution,
                    page_layout=page_layout,
                    production=True,
                ),
                batch_size=batch_size,
                q_len_per_req=q_len_per_req,
                nominal_seq_len=nominal_seq_len,
                distribution=distribution,
                page_layout=page_layout,
                seed=DEFAULT_SEED,
                production=True,
            )
        )
    return tuple(cases)


def _robustness_cases() -> tuple[DecodeCorrectnessCase, ...]:
    cases = []
    for offset in range(ROBUSTNESS_CASE_COUNT):
        index = PRODUCTION_CASE_COUNT + offset
        batch_size = _ROBUST_BATCHES[(offset * 7) % len(_ROBUST_BATCHES)]
        q_len_per_req = Q_LENGTHS[(offset * 3) % len(Q_LENGTHS)]
        nominal_seq_len = _ROBUST_SEQ_LENS[(offset * 5) % len(_ROBUST_SEQ_LENS)]
        page_layout = _PAGE_LAYOUTS[(offset * 2) % len(_PAGE_LAYOUTS)]
        if (
            page_layout == "disjoint"
            and batch_size * ((nominal_seq_len + 127) // 128) > 8192
        ):
            page_layout = "shared_prefix"
        cases.append(
            DecodeCorrectnessCase(
                case_id=_case_id(
                    index=index,
                    batch_size=batch_size,
                    q_len_per_req=q_len_per_req,
                    nominal_seq_len=nominal_seq_len,
                    distribution="boundary",
                    page_layout=page_layout,
                    production=False,
                ),
                batch_size=batch_size,
                q_len_per_req=q_len_per_req,
                nominal_seq_len=nominal_seq_len,
                distribution="boundary",
                page_layout=page_layout,
                seed=DEFAULT_SEED,
                production=False,
            )
        )
    return tuple(cases)


FULL_CASES = _production_cases() + _robustness_cases()


def _smoke_cases() -> tuple[DecodeCorrectnessCase, ...]:
    mandatory = {
        min(FULL_CASES, key=lambda case: (case.batch_size, case.nominal_seq_len)).case_id,
        max(FULL_CASES, key=lambda case: (case.batch_size, case.nominal_seq_len)).case_id,
    }
    for q_len in Q_LENGTHS:
        mandatory.add(next(case.case_id for case in FULL_CASES if case.q_len_per_req == q_len))
    for page_layout in _PAGE_LAYOUTS:
        mandatory.add(next(case.case_id for case in FULL_CASES if case.page_layout == page_layout))
    ordered = sorted(
        FULL_CASES,
        key=lambda case: hashlib.sha256(
            f"decode-smoke-v1:{case.case_id}".encode()
        ).hexdigest(),
    )
    selected = set(mandatory)
    for case in ordered:
        if len(selected) == SMOKE_CASE_COUNT:
            break
        selected.add(case.case_id)
    return tuple(case for case in FULL_CASES if case.case_id in selected)


SMOKE_CASES = _smoke_cases()


def correctness_cases(suite: str) -> tuple[DecodeCorrectnessCase, ...]:
    if suite == "smoke":
        return SMOKE_CASES
    if suite == "full":
        return FULL_CASES
    raise ValueError("suite must be 'smoke' or 'full'")


def make_seq_lens(case: DecodeCorrectnessCase) -> torch.Tensor:
    """Build an exact-mean deterministic varlen vector for one case."""

    generator = torch.Generator().manual_seed(case.seed ^ int(case.case_id[-8:], 16))
    batch = case.batch_size
    nominal = case.nominal_seq_len
    values = torch.full((batch,), nominal, dtype=torch.int64)
    pair_count = batch // 2
    if pair_count:
        if case.distribution == "boundary":
            max_delta = max(1, min(nominal // 16, 127))
            min_delta = 0
        elif case.distribution == "narrow":
            min_delta, max_delta = max(1, nominal // 32), max(2, nominal // 8)
        elif case.distribution == "bimodal":
            min_delta, max_delta = max(1, nominal // 8), max(2, nominal // 3)
        else:
            min_delta, max_delta = max(1, nominal // 16), max(2, nominal // 2)
        deltas = torch.randint(
            min_delta,
            max_delta + 1,
            (pair_count,),
            generator=generator,
            dtype=torch.int64,
        )
        values[:pair_count] -= deltas
        values[pair_count : 2 * pair_count] += deltas
    values.clamp_(min=case.q_len_per_req, max=MAX_SEQ_LEN)
    values = values[torch.randperm(batch, generator=generator)]
    return values.to(torch.int32)


assert len(FULL_CASES) == FULL_CASE_COUNT
assert len(SMOKE_CASES) == SMOKE_CASE_COUNT
assert sum(case.production for case in FULL_CASES) == PRODUCTION_CASE_COUNT


__all__ = [
    "DEFAULT_SEED",
    "FULL_CASES",
    "FULL_CASE_COUNT",
    "MAX_SEQ_LEN",
    "Q_LENGTHS",
    "SMOKE_CASES",
    "SMOKE_CASE_COUNT",
    "DecodeCorrectnessCase",
    "correctness_cases",
    "make_seq_lens",
]
