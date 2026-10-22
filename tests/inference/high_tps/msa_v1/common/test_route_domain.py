"""Validate the distinct TopK=4 route domain used by decode fixtures."""

import pytest
import torch

from tests.inference.high_tps.msa_v1.common.m3_decode_fixture import route_ids


@pytest.mark.parametrize("active_experts", [1, 2, 3])
def test_insufficient_active_experts_rejected(active_experts):
    with pytest.raises(ValueError, match=r"active_experts must be in \[4, 32\]"):
        route_ids(active_experts, torch.device("cpu"))


@pytest.mark.parametrize("active_experts", range(4, 33))
@pytest.mark.parametrize("route_pattern", ["balanced", "skewed", "permuted", "random"])
def test_active_expert_route_domain(active_experts, route_pattern):
    routes = route_ids(active_experts, torch.device("cpu"), route_pattern)
    assert routes.shape == (8, 4)
    assert len(torch.unique(routes)) == active_experts
    assert all(len(torch.unique(row)) == 4 for row in routes)


@pytest.mark.parametrize("q_len", range(1, 9))
@pytest.mark.parametrize("route_pattern", ["balanced", "skewed", "permuted", "random"])
def test_valid_query_rows_cover_requested_experts(q_len, route_pattern):
    for active_experts in range(4, min(32, 4 * q_len) + 1):
        routes = route_ids(
            active_experts, torch.device("cpu"), route_pattern, q_len
        )
        assert routes.shape == (8, 4)
        assert len(torch.unique(routes[-q_len:])) == active_experts
        assert all(len(torch.unique(row)) == 4 for row in routes)


def test_unreachable_active_expert_count_rejected():
    with pytest.raises(ValueError, match=r"active_experts must not exceed 4 \* q_len"):
        route_ids(5, torch.device("cpu"), q_len=1)


@pytest.mark.parametrize("active_experts", range(4, 33))
@pytest.mark.parametrize("route_pattern", ["balanced", "skewed", "permuted", "random"])
def test_full_query_routes_preserve_previous_fixture(active_experts, route_pattern):
    routes = torch.arange(32, dtype=torch.int64).remainder(active_experts).view(8, 4)
    if route_pattern == "skewed":
        routes = torch.arange(4, dtype=torch.int64).repeat(8, 1)
        for expert in range(4, active_experts):
            index = expert - 4
            routes[index // 4, index % 4] = expert
    elif route_pattern == "permuted":
        mapping = (torch.arange(active_experts, dtype=torch.int64) * 37 + 11) % 128
        routes = mapping[routes]
    elif route_pattern == "random":
        generator = torch.Generator().manual_seed(20260731 + active_experts)
        mapping = torch.randperm(128, generator=generator)[:active_experts]
        routes = mapping[routes[torch.randperm(8, generator=generator)]]
        routes = torch.stack(
            [row[torch.randperm(4, generator=generator)] for row in routes]
        )
    actual = route_ids(active_experts, torch.device("cpu"), route_pattern, q_len=8)
    assert torch.equal(actual, routes.int())
