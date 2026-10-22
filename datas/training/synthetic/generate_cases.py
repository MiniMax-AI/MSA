#!/usr/bin/env python3
"""Generate supplemental synthetic CP workloads."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DATA_DIR = Path(__file__).resolve().parent
SOURCE_GLOBAL_SEQLEN = 196608
SOURCE_CASE_COUNT = 99
GENERATED_CASE_COUNT = 128
DEFAULT_SEED = 20260813
DEFAULT_JITTER_BASIS_POINTS = 500

SCENARIOS = (
    (128 * 1024, 16, "cp_cases_128k_cp16.json"),
    (256 * 1024, 32, "cp_cases_256k_cp32.json"),
    (512 * 1024, 64, "cp_cases_512k_cp64.json"),
)


@dataclass(frozen=True)
class Profile:
    weights: tuple[int, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DATA_DIR / "cp32_source_profiles.json",
        help="JSON file containing the canonical 99 CP32 source profiles",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="directory for the three generated JSON files",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--jitter-basis-points",
        type=int,
        default=DEFAULT_JITTER_BASIS_POINTS,
        help="per-sequence multiplicative jitter for the 29 bootstrap profiles",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that existing outputs are byte-for-byte reproducible",
    )
    return parser.parse_args()


def _load_source_cases(source_path: Path) -> tuple[tuple[int, ...], ...]:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != SOURCE_CASE_COUNT:
        raise RuntimeError(
            f"source profile file must contain {SOURCE_CASE_COUNT} cases"
        )

    cases = []
    for sample_id, value in enumerate(payload):
        if not isinstance(value, dict) or set(value) != {"global_cu_seqlens"}:
            raise RuntimeError(
                f"source sample_id={sample_id} fields must be exactly "
                "['global_cu_seqlens']"
            )
        raw_offsets = value["global_cu_seqlens"]
        if not isinstance(raw_offsets, list) or any(
            isinstance(offset, bool) or not isinstance(offset, int)
            for offset in raw_offsets
        ):
            raise RuntimeError(
                f"source sample_id={sample_id} offsets must be an integer array"
            )
        offsets = tuple(raw_offsets)
        if offsets[0] != 0 or offsets[-1] != SOURCE_GLOBAL_SEQLEN:
            raise RuntimeError(f"source sample_id={sample_id} has invalid endpoints")
        if any(begin >= end for begin, end in zip(offsets, offsets[1:])):
            raise RuntimeError(f"source sample_id={sample_id} is not strictly increasing")
        cases.append(offsets)
    return tuple(cases)


def _sequence_lengths(cu_seqlens: Sequence[int]) -> tuple[int, ...]:
    return tuple(end - begin for begin, end in zip(cu_seqlens, cu_seqlens[1:]))


def _make_profiles(
    source_cases: tuple[tuple[int, ...], ...],
    *,
    seed: int,
    jitter_basis_points: int,
) -> tuple[Profile, ...]:
    if not 0 <= jitter_basis_points < 10000:
        raise ValueError("jitter_basis_points must be in [0, 10000)")

    profiles = [
        Profile(
            weights=_sequence_lengths(cu_seqlens),
        )
        for cu_seqlens in source_cases
    ]

    source_sample_ids = list(range(SOURCE_CASE_COUNT))
    random.Random(seed).shuffle(source_sample_ids)
    bootstrap_count = GENERATED_CASE_COUNT - SOURCE_CASE_COUNT
    for bootstrap_index, source_sample_id in enumerate(source_sample_ids[:bootstrap_count]):
        case_id = SOURCE_CASE_COUNT + bootstrap_index
        profile_seed = seed + case_id * 1000003 + source_sample_id * 10007
        profile_rng = random.Random(profile_seed)
        source_lengths = _sequence_lengths(source_cases[source_sample_id])
        jittered_weights = tuple(
            length
            * (
                10000
                + profile_rng.randint(-jitter_basis_points, jitter_basis_points)
            )
            for length in source_lengths
        )
        profiles.append(
            Profile(
                weights=jittered_weights,
            )
        )

    if len(profiles) != GENERATED_CASE_COUNT:
        raise RuntimeError(f"expected {GENERATED_CASE_COUNT} profiles")
    return tuple(profiles)


def _scale_lengths(weights: Sequence[int], target_total: int) -> tuple[int, ...]:
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("profile weights must be positive")
    weight_sum = sum(weights)
    scaled_lengths = []
    remainders = []
    for sequence_id, weight in enumerate(weights):
        scaled_length, remainder = divmod(weight * target_total, weight_sum)
        scaled_lengths.append(scaled_length)
        remainders.append((remainder, sequence_id))

    residual = target_total - sum(scaled_lengths)
    for _, sequence_id in sorted(remainders, key=lambda item: (-item[0], item[1]))[
        :residual
    ]:
        scaled_lengths[sequence_id] += 1

    if any(length <= 0 for length in scaled_lengths):
        raise RuntimeError("scaling produced a zero-length sequence")
    if sum(scaled_lengths) != target_total:
        raise RuntimeError("scaled profile does not match target total")
    return tuple(scaled_lengths)


def _make_cu_seqlens(lengths: Sequence[int]) -> tuple[int, ...]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return tuple(offsets)


def _make_case(
    profile: Profile,
    *,
    global_seqlen: int,
) -> dict[str, list[int]]:
    document_lengths = _scale_lengths(profile.weights, global_seqlen)
    return {
        "global_cu_seqlens": list(_make_cu_seqlens(document_lengths)),
    }


def _make_cases(
    *,
    global_seqlen: int,
    profiles: tuple[Profile, ...],
) -> list[dict[str, list[int]]]:
    cases = [
        _make_case(
            profile,
            global_seqlen=global_seqlen,
        )
        for profile in profiles
    ]
    case_keys = {
        tuple(case["global_cu_seqlens"])
        for case in cases
    }
    if len(case_keys) != len(cases):
        raise RuntimeError("generated case set contains duplicate cases")
    return cases


def _render_cases(cases: list[dict[str, list[int]]]) -> str:
    return json.dumps(cases, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    args = _parse_args()
    source_cases = _load_source_cases(args.source)
    profiles = _make_profiles(
        source_cases,
        seed=args.seed,
        jitter_basis_points=args.jitter_basis_points,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for global_seqlen, cp_size, filename in SCENARIOS:
        if global_seqlen // cp_size != 8192:
            raise RuntimeError("each CP scenario must contain 8192 Q tokens per rank")
        cases = _make_cases(
            global_seqlen=global_seqlen,
            profiles=profiles,
        )
        output_path = args.output_dir / filename
        rendered = _render_cases(cases)
        if args.check:
            if not output_path.is_file():
                raise RuntimeError(f"missing generated file: {output_path}")
            if output_path.read_text(encoding="utf-8") != rendered:
                raise RuntimeError(f"generated file is stale: {output_path}")
            action = "verified"
        else:
            output_path.write_text(rendered, encoding="utf-8")
            action = "wrote"
        print(f"{action} {output_path}: {len(cases)} cases")


if __name__ == "__main__":
    main()
