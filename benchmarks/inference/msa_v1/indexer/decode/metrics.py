"""Necessary-traffic model and acceptance for multi-head decode indexers."""

import math

import torch

MINIMUM_MBU_RATIO = 0.99
MINIMUM_Q8KV4_MULTIHEAD_MBU_RATIO = 0.90


def compare_bf16_mbu(candidate, baseline):
    """Compare BF16 against Q8KV8 at identical local head and query counts."""
    if candidate["precision"] != "bf16" or baseline["precision"] != "q8kv8":
        raise ValueError("BF16 acceptance requires a Q8KV8 baseline")
    for key in ("protocol", "device", "device_uuid", "dsl_version", "num_index_heads"):
        if candidate[key] != baseline[key]:
            raise ValueError(f"baseline/candidate {key} mismatch")
    protocol = candidate["protocol"]
    counts = {"full": 28, "low-latency": 12}
    if not protocol["formal"] or protocol["suite"] not in counts:
        raise ValueError(
            "acceptance requires an unfiltered production or low-latency suite"
        )
    for key, value in (
        ("warmup_replays", 5),
        ("timed_replays", 20),
        ("cuda_graph_calls", 120),
    ):
        if protocol[key] != value:
            raise ValueError(f"acceptance requires {key}={value}")
    previous = {row["name"]: row for row in baseline["results"]}
    rows = candidate["results"]
    count = counts[protocol["suite"]]
    if (
        len(previous) != count
        or len(rows) != count
        or set(previous) != {r["name"] for r in rows}
    ):
        raise ValueError("acceptance requires identical complete case sets")
    totals = [0.0, 0.0, 0.0, 0.0]
    for row in rows:
        old = previous[row["name"]]
        if row["weight"] != old["weight"] or row["slots"] != old["slots"]:
            raise ValueError("case weights and cold-cache slots must match")
        for value in (row, old):
            if not all(
                math.isfinite(value[key])
                for key in (
                    "cv",
                    "latency_us",
                    "necessary_bytes",
                    "weight",
                    "reuse_distance_over_l2",
                )
            ):
                raise ValueError("invalid measurement")
            if (
                not 0 <= value["cv"] <= 0.03
                or value["reuse_distance_over_l2"] <= 2
                or value["latency_us"] <= 0
                or value["necessary_bytes"] <= 0
                or value["weight"] < 0
            ):
                raise ValueError("invalid measurement")
        weight = 1.0 if protocol["suite"] == "low-latency" else row["weight"]
        if weight <= 0:
            raise ValueError("production weights must be positive")
        totals[0] += weight * row["necessary_bytes"]
        totals[1] += weight * row["latency_us"]
        totals[2] += weight * old["necessary_bytes"]
        totals[3] += weight * old["latency_us"]
    ratio = (totals[0] / totals[1]) / (totals[2] / totals[3])
    return {
        "passed": ratio >= 0.9,
        "minimum_weighted_effective_mbu_ratio": 0.9,
        "weighted_effective_mbu_ratio": ratio,
        "weighted_latency_ratio": totals[1] / totals[3],
        "weighting": "equal" if protocol["suite"] == "low-latency" else "production",
    }


def necessary_bytes(
    lengths, heads, page_bytes, *, query_length=8, query_element_bytes=1
):
    """Count shared K once per request; do not credit redundant kernel traffic."""
    local = (
        lengths[:, None].to(torch.int64) - query_length + torch.arange(query_length)
    ) // 128
    pages = int(local[:, -1].sum())
    scores = int(local.sum())
    selected_scores = int(local.masked_select(local >= 16).sum())
    batch = lengths.numel()
    return (
        pages * page_bytes
        + batch * query_length * heads * 128 * query_element_bytes
        + pages * 4
        + batch * 4
        + batch * query_length * heads * 4
        + heads * (scores + selected_scores) * 4
        + heads * batch * query_length * 16 * 4
    )


def compare_mbu(candidate, baseline):
    """Compare identical full protocols using aggregate bytes over aggregate time."""
    for key in ("protocol", "device", "device_uuid", "dsl_version", "precision"):
        if candidate[key] != baseline[key]:
            raise ValueError(f"baseline/candidate {key} mismatch")
    if not candidate["protocol"]["formal"]:
        raise ValueError("acceptance requires the full unfiltered manifest")
    old = {row["name"]: row for row in baseline["results"]}
    rows = candidate["results"]
    if len(rows) != 28 or len(old) != 28 or set(old) != {r["name"] for r in rows}:
        raise ValueError("acceptance requires the same 28 cases")
    heads = candidate["num_index_heads"]
    if baseline["num_index_heads"] != 1:
        raise ValueError("baseline must be the original H=1 implementation")
    totals = [0.0, 0.0, 0.0, 0.0]
    regressed = []
    for row in rows:
        previous = old[row["name"]]
        if row["weight"] != previous["weight"] or row["slots"] != previous["slots"]:
            raise ValueError("case weights and cold-cache slots must match")
        for value in (row, previous):
            if (
                not math.isfinite(value["cv"])
                or value["cv"] > 0.03
                or value["latency_us"] <= 0
            ):
                raise ValueError("invalid timing")
        w = row["weight"]
        totals[0] += w * row["necessary_bytes"]
        totals[1] += w * row["latency_us"]
        totals[2] += w * previous["necessary_bytes"]
        totals[3] += w * previous["latency_us"]
        if heads == 1 and row["latency_us"] > previous["latency_us"] * 1.05:
            regressed.append(row["name"])
    ratio = (totals[0] / totals[1]) / (totals[2] / totals[3])
    minimum_ratio = (
        MINIMUM_Q8KV4_MULTIHEAD_MBU_RATIO
        if candidate["precision"] == "q8kv4" and heads > 1
        else MINIMUM_MBU_RATIO
    )
    return {
        "passed": ratio >= minimum_ratio and not regressed,
        "minimum_weighted_effective_mbu_ratio": minimum_ratio,
        "weighted_effective_mbu_ratio": ratio,
        "h1_regressed_cases": regressed,
    }


def compare_low_latency(candidate, baseline):
    """Gate every low-latency case at the same H without a weighted aggregate."""
    from benchmarks.inference.msa_v1.decode.cases import LOW_LATENCY_CASES

    for key in (
        "protocol",
        "device",
        "device_uuid",
        "dsl_version",
        "precision",
        "num_index_heads",
    ):
        if candidate[key] != baseline[key]:
            raise ValueError(f"baseline/candidate {key} mismatch")
    if (
        candidate["protocol"]["suite"] != "low-latency"
        or not candidate["protocol"]["formal"]
    ):
        raise ValueError("acceptance requires the low-latency suite")
    for key, expected_value in (
        ("warmup_replays", 5),
        ("timed_replays", 20),
        ("cuda_graph_calls", 120),
    ):
        if candidate["protocol"].get(key) != expected_value:
            raise ValueError(
                "low-latency acceptance requires the standard timing protocol"
            )
    expected = {case.name for case in LOW_LATENCY_CASES}
    mappings = []
    for payload in (candidate, baseline):
        rows = payload["results"]
        mapping = {row["name"]: row for row in rows}
        if len(rows) != len(expected) or set(mapping) != expected:
            raise ValueError("acceptance requires all 12 low-latency cases")
        for row in rows:
            if (
                not math.isfinite(row["cv"])
                or not 0 <= row["cv"] <= 0.03
                or not math.isfinite(row["latency_us"])
                or row["latency_us"] <= 0
                or row["reuse_distance_over_l2"] < 2
                or not math.isfinite(row["reuse_distance_over_l2"])
            ):
                raise ValueError("invalid timing or cold-cache coverage")
        mappings.append(mapping)
    ratios = {}
    for case in LOW_LATENCY_CASES:
        row, previous = (mapping[case.name] for mapping in mappings)
        for key in (
            "weight",
            "slots",
            "seed",
            "batch_size",
            "nominal_seq_len",
            "necessary_bytes",
        ):
            if row[key] != previous[key]:
                raise ValueError(f"case {case.name} {key} mismatch")
        if row["weight"] != 0:
            raise ValueError("low-latency cases must not carry production weights")
        ratios[case.name] = row["latency_us"] / previous["latency_us"]
    regressed = [name for name, ratio in ratios.items() if ratio > 1.05]
    return {
        "passed": not regressed,
        "maximum_latency_ratio": 1.05,
        "latency_ratios": ratios,
        "regressed_cases": regressed,
    }
