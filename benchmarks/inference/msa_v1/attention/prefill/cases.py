"""Real inference prefill benchmark cases shared by attention formats."""

from __future__ import annotations

from dataclasses import dataclass

from datas.inference.cases import (
    InferencePrefillBenchmarkCase,
    InferencePrefillCase,
    load_prefill_benchmark_cases,
    load_prefill_cases,
)

PAGE_SIZE = 128
TOPK = 16
Q_HEADS = 64
HEAD_DIM = 128


@dataclass(frozen=True)
class PrefillCase:
    shape: InferencePrefillCase
    tags: tuple[str, ...] = ()
    representative_weight: int | None = None
    stratum: str | None = None

    @property
    def name(self) -> str:
        return self.shape.case_id

    @property
    def batch(self) -> int:
        return self.shape.batch_size

    @property
    def query_lens(self) -> tuple[int, ...]:
        return self.shape.query_lens

    @property
    def prefix_lens(self) -> tuple[int, ...]:
        return self.shape.prefix_lens

    @property
    def final_kv_lens(self) -> tuple[int, ...]:
        return self.shape.final_kv_lens

    @property
    def total_q(self) -> int:
        return self.shape.total_q

    @property
    def max_query_len(self) -> int:
        return self.shape.max_query_len

    @property
    def max_final_kv(self) -> int:
        return self.shape.max_final_kv

    @property
    def max_cols(self) -> int:
        return self.shape.max_cols

    @property
    def seed(self) -> int:
        return self.shape.seed

    @property
    def num_active_pages(self) -> int:
        return sum((length + PAGE_SIZE - 1) // PAGE_SIZE for length in self.final_kv_lens)

    @property
    def selected_tokens(self) -> int:
        total = 0
        for query_len, prefix_len in zip(
            self.query_lens, self.prefix_lens, strict=True
        ):
            for query_idx in range(query_len):
                position = prefix_len + query_idx
                local_page, local_offset = divmod(position, PAGE_SIZE)
                valid_pages = min(TOPK, local_page + 1)
                total += (valid_pages - 1) * PAGE_SIZE + local_offset + 1
        return total

    @property
    def average_selected_tokens(self) -> float:
        return self.selected_tokens / self.total_q

    @property
    def sparse_density(self) -> float:
        dense_tokens = sum(
            query_len * (prefix_len + 1) + query_len * (query_len - 1) // 2
            for query_len, prefix_len in zip(
                self.query_lens, self.prefix_lens, strict=True
            )
        )
        return self.selected_tokens / dense_tokens

    @property
    def useful_flops(self) -> int:
        return 4 * HEAD_DIM * Q_HEADS * self.selected_tokens


def _from_selection(selection: InferencePrefillBenchmarkCase) -> PrefillCase:
    return PrefillCase(
        selection.case,
        selection.tags,
        selection.representative_weight,
        selection.stratum,
    )


def real_prefill_cases(
    suite: str = "full",
    batch_size: int | None = None,
) -> tuple[PrefillCase, ...]:
    if suite not in {"smoke", "full"}:
        raise ValueError("suite must be 'smoke' or 'full'")
    cases = tuple(
        _from_selection(selection)
        for selection in load_prefill_benchmark_cases(batch_size)
    )
    if suite == "smoke":
        cases = tuple(case for case in cases if "batch_anchor" in case.tags)
    return cases


def warmup_case(batch_size: int | None = None) -> PrefillCase:
    cases = load_prefill_cases()
    if batch_size is not None:
        cases = tuple(case for case in cases if case.batch_size == batch_size)
    return PrefillCase(
        min(cases, key=lambda case: (case.total_q * case.max_final_kv, case.case_id))
    )


__all__ = [
    "HEAD_DIM",
    "PAGE_SIZE",
    "Q_HEADS",
    "TOPK",
    "PrefillCase",
    "real_prefill_cases",
    "warmup_case",
]
