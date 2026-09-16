"""Sanitized CP metadata shared by MSA v1 tests and benchmarks."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

TRAINING_DIR = Path(__file__).resolve().parent
REAL_DIR = TRAINING_DIR / "real"
SYNTHETIC_DIR = TRAINING_DIR / "synthetic"
MANIFEST_PATH = REAL_DIR / "cp16_192k_cases_v1.jsonl"
TEST_SELECTION_PATH = REAL_DIR / "cp16_192k_test_selection_v1.json"
BENCHMARK_SELECTION_PATH = REAL_DIR / "cp16_192k_benchmark_selection_v1.json"

DEFAULT_SCENARIO = "192k_cp16"
EXPECTED_REAL_CASES = 1253
EXPECTED_REAL_CALLS = 1500
CHUNK_SIZE = 1024


@dataclass(frozen=True)
class SparseSpec:
    name: str
    block_size: int
    topk_capacity: int
    index_heads: int


MSA_V1_SPEC = SparseSpec("msa_v1", 128, 16, 4)


@dataclass(frozen=True)
class CpScenario:
    name: str
    total_tokens: int
    cp_size: int
    filename: str
    synthetic: bool = False

    @property
    def chunk_size(self) -> int:
        return CHUNK_SIZE

    @property
    def num_chunks(self) -> int:
        return self.total_tokens // self.chunk_size

    @property
    def chunks_per_rank(self) -> int:
        return self.num_chunks // self.cp_size

    @property
    def q_tokens_per_rank(self) -> int:
        return self.chunks_per_rank * self.chunk_size


OFFICIAL_SCENARIOS = {
    DEFAULT_SCENARIO: CpScenario(DEFAULT_SCENARIO, 192 * 1024, 16, MANIFEST_PATH.name)
}
SYNTHETIC_SCENARIOS = {
    "128k_cp16": CpScenario(
        "128k_cp16", 128 * 1024, 16, "cp_cases_128k_cp16.json", True
    ),
    "256k_cp32": CpScenario(
        "256k_cp32", 256 * 1024, 32, "cp_cases_256k_cp32.json", True
    ),
    "512k_cp64": CpScenario(
        "512k_cp64", 512 * 1024, 64, "cp_cases_512k_cp64.json", True
    ),
}
# Default callers only see production workloads. Synthetic inputs require opt-in.
SCENARIOS = OFFICIAL_SCENARIOS


@dataclass(frozen=True)
class TokenRange:
    begin: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True)
class GlobalCpCase:
    case_id: int
    global_cu_seqlens: tuple[int, ...]
    metrics: Mapping[str, int]

    @property
    def document_lengths(self) -> tuple[int, ...]:
        return _sequence_lengths(self.global_cu_seqlens)

    @property
    def total_tokens(self) -> int:
        return self.global_cu_seqlens[-1]


@dataclass(frozen=True)
class CpFragment:
    document_id: int
    local_begin: int
    local_end: int
    global_begin: int

    @property
    def global_end(self) -> int:
        return self.global_begin + self.local_end - self.local_begin

    @property
    def length(self) -> int:
        return self.local_end - self.local_begin


@dataclass(frozen=True)
class CpRankCase:
    scenario: str
    case_id: int
    rank: int
    cp_size: int
    sparse_spec: SparseSpec
    chunk_ids: tuple[int, ...]
    fragments: tuple[CpFragment, ...]
    cu_seqlens_q: tuple[int, ...]
    cu_seqlens_kv: tuple[int, ...]
    fragment_indices: tuple[int, ...]
    q_source_ranges: tuple[TokenRange, ...]
    kv_source_ranges: tuple[TokenRange, ...]
    causal_elements: int

    @property
    def total_q(self) -> int:
        return self.cu_seqlens_q[-1]

    @property
    def total_kv(self) -> int:
        return sum(source.length for source in self.kv_source_ranges)

    @property
    def max_seqlen_q(self) -> int:
        return max(_sequence_lengths(self.cu_seqlens_q))

    @property
    def max_seqlen_kv(self) -> int:
        return max(
            end - self.cu_seqlens_kv[self.fragment_indices[fragment_id]]
            for fragment_id, end in enumerate(self.cu_seqlens_kv[1:])
        )

    @property
    def total_kv_rows(self) -> int:
        block_size = self.sparse_spec.block_size
        return sum(
            (
                end
                - self.cu_seqlens_kv[self.fragment_indices[fragment_id]]
                + block_size
                - 1
            )
            // block_size
            for fragment_id, end in enumerate(self.cu_seqlens_kv[1:])
        )

    @property
    def sparse_attention_elements(self) -> int:
        return sum(
            _fragment_sparse_causal_area(
                fragment.local_begin,
                fragment.local_end,
                block_size=self.sparse_spec.block_size,
                topk_capacity=self.sparse_spec.topk_capacity,
            )
            for fragment in self.fragments
        )


@dataclass(frozen=True)
class TestSelection:
    selection_id: int
    case_id: int
    rank: int
    structured: bool


@dataclass(frozen=True)
class BenchmarkSelection:
    benchmark_case_id: int
    case_id: int
    rank: int
    representative_weight: int
    stratum: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_lengths(cu_seqlens: Sequence[int]) -> tuple[int, ...]:
    return tuple(end - begin for begin, end in zip(cu_seqlens, cu_seqlens[1:]))


def _shape_metrics(cu_seqlens: tuple[int, ...]) -> dict[str, int]:
    lengths = _sequence_lengths(cu_seqlens)
    ordered = sorted(lengths)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) // 2
    )
    return {
        "total_tokens": cu_seqlens[-1],
        "num_sequences": len(lengths),
        "min_seqlen": min(lengths),
        "median_seqlen": median,
        "max_seqlen": max(lengths),
        "causal_elements": sum(length * (length + 1) // 2 for length in lengths),
    }


def _parse_cu_seqlens(value: object, *, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(offset, bool) or not isinstance(offset, int) for offset in value
    ):
        raise ValueError(f"{context}: cu_seqlens must be an integer array")
    offsets = tuple(value)
    if len(offsets) < 2 or offsets[0] != 0:
        raise ValueError(f"{context}: cu_seqlens must start at zero")
    if any(begin >= end for begin, end in zip(offsets, offsets[1:])):
        raise ValueError(f"{context}: cu_seqlens must strictly increase")
    return offsets


def _parse_real_case(value: object, line_number: int) -> GlobalCpCase:
    if not isinstance(value, dict) or set(value) != {
        "case_id",
        "cu_seqlens",
        "metrics",
    }:
        raise ValueError(f"manifest line {line_number}: unexpected fields")
    case_id = value["case_id"]
    if isinstance(case_id, bool) or not isinstance(case_id, int):
        raise ValueError(f"manifest line {line_number}: case_id must be an integer")
    cu_seqlens = _parse_cu_seqlens(value["cu_seqlens"], context=f"case_id={case_id}")
    metrics = value["metrics"]
    expected = _shape_metrics(cu_seqlens)
    if not isinstance(metrics, dict) or metrics != expected:
        raise ValueError(f"case_id={case_id}: shape metrics are inconsistent")
    return GlobalCpCase(case_id, cu_seqlens, expected)


@cache
def load_real_cases() -> tuple[GlobalCpCase, ...]:
    cases = tuple(
        _parse_real_case(json.loads(line), line_number)
        for line_number, line in enumerate(
            MANIFEST_PATH.read_text(encoding="utf-8").splitlines(), 1
        )
        if line
    )
    if len(cases) != EXPECTED_REAL_CASES:
        raise ValueError(f"real manifest must contain {EXPECTED_REAL_CASES} cases")
    if [case.case_id for case in cases] != list(range(len(cases))):
        raise ValueError("real manifest case IDs must be contiguous")
    if len({case.global_cu_seqlens for case in cases}) != len(cases):
        raise ValueError("real manifest contains duplicate shapes")
    scenario = OFFICIAL_SCENARIOS[DEFAULT_SCENARIO]
    if any(case.total_tokens != scenario.total_tokens for case in cases):
        raise ValueError("real manifest contains a non-192K case")
    return cases


@cache
def load_synthetic_cases(name: str) -> tuple[GlobalCpCase, ...]:
    try:
        scenario = SYNTHETIC_SCENARIOS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown synthetic scenario {name!r}: expected {tuple(SYNTHETIC_SCENARIOS)}"
        ) from exc
    path = SYNTHETIC_DIR / scenario.filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 128:
        raise ValueError(f"{path} must contain 128 synthetic cases")
    cases = []
    for case_id, value in enumerate(payload):
        if not isinstance(value, dict) or set(value) != {"global_cu_seqlens"}:
            raise ValueError(f"{path}: synthetic case {case_id} has unexpected fields")
        cu_seqlens = _parse_cu_seqlens(
            value["global_cu_seqlens"], context=f"synthetic case_id={case_id}"
        )
        cases.append(GlobalCpCase(case_id, cu_seqlens, _shape_metrics(cu_seqlens)))
    if any(case.total_tokens != scenario.total_tokens for case in cases):
        raise ValueError(f"{path} contains a case with the wrong total length")
    return tuple(cases)


def load_global_cases(
    name: str = DEFAULT_SCENARIO,
    *,
    allow_synthetic: bool = False,
) -> tuple[GlobalCpCase, ...]:
    if name in OFFICIAL_SCENARIOS:
        return load_real_cases()
    if name in SYNTHETIC_SCENARIOS:
        if not allow_synthetic:
            raise ValueError(
                f"{name!r} is synthetic; pass allow_synthetic=True for supplemental use"
            )
        return load_synthetic_cases(name)
    raise ValueError(f"unknown training scenario: {name!r}")


def _scenario(name: str, *, allow_synthetic: bool) -> CpScenario:
    if name in OFFICIAL_SCENARIOS:
        return OFFICIAL_SCENARIOS[name]
    if name in SYNTHETIC_SCENARIOS and allow_synthetic:
        return SYNTHETIC_SCENARIOS[name]
    if name in SYNTHETIC_SCENARIOS:
        raise ValueError(
            f"{name!r} is synthetic; pass allow_synthetic=True for supplemental use"
        )
    raise ValueError(f"unknown training scenario: {name!r}")


def _chunk_causal_area(
    chunk_id: int,
    chunk_size: int,
    global_cu_seqlens: Sequence[int],
) -> int:
    chunk_begin = chunk_id * chunk_size
    chunk_end = chunk_begin + chunk_size
    area = 0
    for document_begin, document_end in zip(global_cu_seqlens, global_cu_seqlens[1:]):
        q_begin = max(chunk_begin, document_begin)
        q_end = min(chunk_end, document_end)
        if q_begin >= q_end:
            continue
        local_begin = q_begin - document_begin
        local_end = q_end - document_begin
        height = local_end - local_begin
        area += (local_begin + 1 + local_end) * height // 2
    return area


def magi_minheap_partitions(
    global_cu_seqlens: Sequence[int],
    cp_size: int,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[tuple[int, ...], ...]:
    return _magi_minheap_partitions_cached(
        tuple(global_cu_seqlens), cp_size, chunk_size
    )


@cache
def _magi_minheap_partitions_cached(
    global_cu_seqlens: tuple[int, ...],
    cp_size: int,
    chunk_size: int,
) -> tuple[tuple[int, ...], ...]:
    total_tokens = global_cu_seqlens[-1]
    if total_tokens % chunk_size:
        raise ValueError("total tokens must be divisible by chunk_size")
    num_chunks = total_tokens // chunk_size
    bucket_limit = math.ceil(num_chunks / cp_size)
    workloads = [
        _chunk_causal_area(chunk_id, chunk_size, global_cu_seqlens)
        for chunk_id in range(num_chunks)
    ]
    sorted_chunk_ids = sorted(range(num_chunks), key=workloads.__getitem__)[::-1]
    bucket_counts = [0] * cp_size
    partitions = [[] for _ in range(cp_size)]
    heap = [(0, rank) for rank in range(cp_size)]
    heapq.heapify(heap)
    for chunk_id in sorted_chunk_ids:
        while heap:
            workload, rank = heapq.heappop(heap)
            if bucket_counts[rank] < bucket_limit:
                break
        else:
            raise RuntimeError("Magi MinHeap dispatch ran out of ranks")
        partitions[rank].append(chunk_id)
        bucket_counts[rank] += 1
        heapq.heappush(heap, (workload + workloads[chunk_id], rank))
    return tuple(tuple(sorted(partition)) for partition in partitions)


def _rank_fragments(
    chunk_ids: Sequence[int],
    global_cu_seqlens: Sequence[int],
    *,
    chunk_size: int,
) -> tuple[CpFragment, ...]:
    fragments: list[CpFragment] = []
    for chunk_id in chunk_ids:
        chunk_begin = chunk_id * chunk_size
        chunk_end = chunk_begin + chunk_size
        for document_id, (document_begin, document_end) in enumerate(
            zip(global_cu_seqlens, global_cu_seqlens[1:])
        ):
            q_begin = max(chunk_begin, document_begin)
            q_end = min(chunk_end, document_end)
            if q_begin >= q_end:
                continue
            local_begin = q_begin - document_begin
            local_end = q_end - document_begin
            if (
                fragments
                and fragments[-1].document_id == document_id
                and fragments[-1].local_end == local_begin
            ):
                previous = fragments[-1]
                fragments[-1] = CpFragment(
                    document_id,
                    previous.local_begin,
                    local_end,
                    previous.global_begin,
                )
            else:
                fragments.append(
                    CpFragment(document_id, local_begin, local_end, q_begin)
                )
    return tuple(fragments)


def _make_cu_seqlens(lengths: Iterable[int]) -> tuple[int, ...]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return tuple(offsets)


def make_rank_case(
    case: GlobalCpCase,
    scenario: CpScenario,
    rank: int,
    sparse_spec: SparseSpec,
) -> CpRankCase:
    if not 0 <= rank < scenario.cp_size:
        raise ValueError(f"rank must be in [0, {scenario.cp_size})")
    if case.total_tokens != scenario.total_tokens:
        raise ValueError("global case length does not match scenario")
    chunk_ids = magi_minheap_partitions(
        case.global_cu_seqlens,
        scenario.cp_size,
        chunk_size=scenario.chunk_size,
    )[rank]
    fragments = _rank_fragments(
        chunk_ids, case.global_cu_seqlens, chunk_size=scenario.chunk_size
    )
    cu_seqlens_q = _make_cu_seqlens(fragment.length for fragment in fragments)
    if cu_seqlens_q[-1] != scenario.q_tokens_per_rank:
        raise RuntimeError(
            f"case={case.case_id} rank={rank} has Q={cu_seqlens_q[-1]}, "
            f"expected {scenario.q_tokens_per_rank}"
        )

    cu_seqlens_kv = [0]
    fragment_indices = []
    kv_source_ranges = []
    fragment_id = 0
    while fragment_id < len(fragments):
        document_id = fragments[fragment_id].document_id
        document_fragment_begin = fragment_id
        packed_kv_base = cu_seqlens_kv[-1]
        document_begin = case.global_cu_seqlens[document_id]
        max_local_end = 0
        while (
            fragment_id < len(fragments)
            and fragments[fragment_id].document_id == document_id
        ):
            local_end = fragments[fragment_id].local_end
            max_local_end = max(max_local_end, local_end)
            cu_seqlens_kv.append(packed_kv_base + local_end)
            fragment_indices.append(document_fragment_begin)
            fragment_id += 1
        kv_source_ranges.append(
            TokenRange(document_begin, document_begin + max_local_end)
        )

    causal_elements = sum(
        _chunk_causal_area(chunk_id, scenario.chunk_size, case.global_cu_seqlens)
        for chunk_id in chunk_ids
    )
    return CpRankCase(
        scenario=scenario.name,
        case_id=case.case_id,
        rank=rank,
        cp_size=scenario.cp_size,
        sparse_spec=sparse_spec,
        chunk_ids=chunk_ids,
        fragments=fragments,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=tuple(cu_seqlens_kv),
        fragment_indices=tuple(fragment_indices),
        q_source_ranges=tuple(
            TokenRange(fragment.global_begin, fragment.global_end)
            for fragment in fragments
        ),
        kv_source_ranges=tuple(kv_source_ranges),
        causal_elements=causal_elements,
    )


def iter_rank_cases(
    name: str = DEFAULT_SCENARIO,
    *,
    sparse_spec: SparseSpec = MSA_V1_SPEC,
    allow_synthetic: bool = False,
) -> Iterator[CpRankCase]:
    scenario = _scenario(name, allow_synthetic=allow_synthetic)
    for case in load_global_cases(name, allow_synthetic=allow_synthetic):
        for rank in range(scenario.cp_size):
            yield make_rank_case(case, scenario, rank, sparse_spec)


def _load_selection_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported selection schema in {path}")
    if payload.get("manifest_sha256") != _sha256(MANIFEST_PATH):
        raise ValueError(f"manifest SHA256 mismatch in {path}")
    return payload


@cache
def load_test_selections(tier: str = "full_gpu") -> tuple[TestSelection, ...]:
    payload = _load_selection_payload(TEST_SELECTION_PATH)
    tiers = payload.get("tiers")
    if not isinstance(tiers, dict) or tier not in tiers:
        raise ValueError(f"unknown training test tier: {tier}")
    value = tiers[tier]
    if not isinstance(value, dict):
        raise ValueError(f"invalid training test tier: {tier}")
    if tier == "static_all":
        result = tuple(
            TestSelection(case.case_id, case.case_id, case.case_id % 16, False)
            for case in load_real_cases()
        )
    else:
        entries = value.get("cases")
        if not isinstance(entries, list):
            raise ValueError(f"training test tier {tier} has no cases")
        result = tuple(
            TestSelection(
                selection_id=int(entry["selection_id"]),
                case_id=int(entry["case_id"]),
                rank=int(entry["rank"]),
                structured=bool(entry["structured"]),
            )
            for entry in entries
            if isinstance(entry, dict)
            and set(entry) == {"selection_id", "case_id", "rank", "structured"}
        )
        if len(result) != len(entries):
            raise ValueError(f"training test tier {tier} has invalid entries")
    if len(result) != int(value["case_count"]):
        raise ValueError(f"training test tier {tier} case count changed")
    if [item.selection_id for item in result] != list(range(len(result))):
        raise ValueError(f"training test tier {tier} IDs must be contiguous")
    case_count = len(load_real_cases())
    if any(not 0 <= item.case_id < case_count for item in result):
        raise ValueError(f"training test tier {tier} references an unknown case")
    if any(not 0 <= item.rank < 16 for item in result):
        raise ValueError(f"training test tier {tier} references an invalid rank")
    if tier == "full_gpu" and len(result) != EXPECTED_REAL_CALLS:
        raise ValueError("full_gpu must preserve all 1500 real calls")
    return result


@cache
def load_benchmark_selections() -> tuple[BenchmarkSelection, ...]:
    payload = _load_selection_payload(BENCHMARK_SELECTION_PATH)
    entries = payload.get("cases")
    if not isinstance(entries, list):
        raise ValueError("training benchmark selection has no cases")
    expected_keys = {
        "benchmark_case_id",
        "case_id",
        "rank",
        "representative_weight",
        "stratum",
    }
    result = tuple(
        BenchmarkSelection(
            benchmark_case_id=int(entry["benchmark_case_id"]),
            case_id=int(entry["case_id"]),
            rank=int(entry["rank"]),
            representative_weight=int(entry["representative_weight"]),
            stratum=str(entry["stratum"]),
        )
        for entry in entries
        if isinstance(entry, dict) and set(entry) == expected_keys
    )
    if len(result) != int(payload["case_count"]) or len(result) != len(entries):
        raise ValueError("training benchmark selection case count changed")
    if [item.benchmark_case_id for item in result] != list(range(len(result))):
        raise ValueError("training benchmark IDs must be contiguous")
    if sum(item.representative_weight for item in result) != EXPECTED_REAL_CALLS:
        raise ValueError("training benchmark weights must represent 1500 calls")
    if any(not 0 <= item.case_id < len(load_real_cases()) for item in result):
        raise ValueError("training benchmark references an unknown case")
    if any(not 0 <= item.rank < 16 for item in result):
        raise ValueError("training benchmark references an invalid rank")
    return result


def selected_rank_cases(
    tier: str,
    *,
    sparse_spec: SparseSpec,
) -> tuple[tuple[TestSelection, CpRankCase], ...]:
    scenario = OFFICIAL_SCENARIOS[DEFAULT_SCENARIO]
    case_map = {case.case_id: case for case in load_real_cases()}
    return tuple(
        (
            selection,
            make_rank_case(
                case_map[selection.case_id], scenario, selection.rank, sparse_spec
            ),
        )
        for selection in load_test_selections(tier)
    )


def selected_benchmark_rank_cases(
    *,
    sparse_spec: SparseSpec,
) -> tuple[tuple[BenchmarkSelection, CpRankCase], ...]:
    scenario = OFFICIAL_SCENARIOS[DEFAULT_SCENARIO]
    case_map = {case.case_id: case for case in load_real_cases()}
    return tuple(
        (
            selection,
            make_rank_case(
                case_map[selection.case_id], scenario, selection.rank, sparse_spec
            ),
        )
        for selection in load_benchmark_selections()
    )


def _padding_prefix(token_count: int, block_size: int) -> int:
    full_blocks, tail = divmod(token_count, block_size)
    block_padding = block_size * (block_size - 1) // 2
    tail_padding = tail * block_size - tail * (tail + 1) // 2
    return full_blocks * block_padding + tail_padding


def _capped_sparse_prefix(
    token_count: int,
    *,
    block_size: int,
    topk_capacity: int,
) -> int:
    capacity_tokens = block_size * topk_capacity
    dense_tokens = min(token_count, capacity_tokens)
    area = dense_tokens * (dense_tokens + 1) // 2
    if token_count <= capacity_tokens:
        return area
    padding = _padding_prefix(token_count, block_size) - _padding_prefix(
        capacity_tokens, block_size
    )
    return area + (token_count - capacity_tokens) * capacity_tokens - padding


def _fragment_sparse_causal_area(
    local_begin: int,
    local_end: int,
    *,
    block_size: int,
    topk_capacity: int,
) -> int:
    return _capped_sparse_prefix(
        local_end, block_size=block_size, topk_capacity=topk_capacity
    ) - _capped_sparse_prefix(
        local_begin, block_size=block_size, topk_capacity=topk_capacity
    )


def metadata_digest(cases: Iterable[CpRankCase]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        payload = (
            case.scenario,
            case.case_id,
            case.rank,
            case.chunk_ids,
            case.cu_seqlens_q,
            case.cu_seqlens_kv,
            case.fragment_indices,
        )
        digest.update(repr(payload).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def make_torch_metadata(
    case: CpRankCase,
    *,
    device: object = "cuda",
) -> dict[str, object]:
    import torch

    return {
        "cu_seqlens_q": torch.tensor(
            case.cu_seqlens_q, dtype=torch.int32, device=device
        ),
        "cu_seqlens_kv": torch.tensor(
            case.cu_seqlens_kv, dtype=torch.int32, device=device
        ),
        "fragment_indices": torch.tensor(
            case.fragment_indices, dtype=torch.int32, device=device
        ),
    }


def make_torch_topk(case: CpRankCase, *, device: object = "cuda"):
    import torch

    spec = case.sparse_spec
    topk = torch.full(
        (spec.index_heads, case.total_q, spec.topk_capacity),
        -1,
        dtype=torch.int32,
        device=device,
    )
    slots = torch.arange(spec.topk_capacity, dtype=torch.int32, device=device)
    for fragment_id, fragment in enumerate(case.fragments):
        q_begin = case.cu_seqlens_q[fragment_id]
        q_end = case.cu_seqlens_q[fragment_id + 1]
        visible_tokens = torch.arange(
            fragment.local_begin + 1,
            fragment.local_end + 1,
            dtype=torch.int32,
            device=device,
        )
        visible_blocks = torch.div(
            visible_tokens + spec.block_size - 1,
            spec.block_size,
            rounding_mode="floor",
        )
        selected = visible_blocks.clamp(max=spec.topk_capacity)
        indices = visible_blocks[:, None] - selected[:, None] + slots[None, :]
        valid = slots[None, :] < selected[:, None]
        fragment_topk = torch.where(valid, indices, torch.full_like(indices, -1))
        topk[:, q_begin:q_end] = fragment_topk.unsqueeze(0)
    return topk


def pack_source_tensor(tensor: object, ranges: Sequence[TokenRange]):
    import torch

    return torch.cat([tensor[source.begin : source.end] for source in ranges], dim=0)


__all__ = [
    "BENCHMARK_SELECTION_PATH",
    "CHUNK_SIZE",
    "DEFAULT_SCENARIO",
    "EXPECTED_REAL_CALLS",
    "MSA_V1_SPEC",
    "OFFICIAL_SCENARIOS",
    "SCENARIOS",
    "SYNTHETIC_SCENARIOS",
    "BenchmarkSelection",
    "CpFragment",
    "CpRankCase",
    "CpScenario",
    "GlobalCpCase",
    "SparseSpec",
    "TestSelection",
    "TokenRange",
    "iter_rank_cases",
    "load_benchmark_selections",
    "load_global_cases",
    "load_real_cases",
    "load_synthetic_cases",
    "load_test_selections",
    "magi_minheap_partitions",
    "make_rank_case",
    "make_torch_metadata",
    "make_torch_topk",
    "metadata_digest",
    "pack_source_tensor",
    "selected_benchmark_rank_cases",
    "selected_rank_cases",
]
