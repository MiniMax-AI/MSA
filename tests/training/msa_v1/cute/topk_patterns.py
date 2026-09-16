"""Legal TopK patterns for KL contract tests."""

import math

import torch

BLOCK_K = 128

TOPK = 16

KV_HEADS = 4

DEFAULT_PATTERNS = (
    "recent",
    "spread",
    "pool16",
    "pool32",
    "pool64",
    "pool128",
    "pool256",
    "hot15_tail1",
    "hot8_tail8",
    "random",
    "b0p99_local_recent",
    "b0p99_local_spread",
    "b0p99_local_pool16",
    "b0p99_local_pool32",
    "b0p99_local_pool64",
    "b0p99_local_hot2",
    "b0p99_local_hot4",
    "b0p99_local_hot6",
    "b0p99_local_hot8",
    "b0p99_local_hot12",
    "b0p99_local_hot14",
)


def _coprime_step(pool: int) -> int:
    for candidate in (127, 61, 31, 17, 13, 7, 5, 3, 1):
        if math.gcd(candidate, pool) == 1:
            return candidate
    raise AssertionError("unreachable")


def _build_block0_local_topk(
    pattern: str,
    visible: torch.Tensor,
    q_idx: torch.Tensor,
    heads: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Build TopK with block 0 at ~99% and every query's local block at 100%."""
    q_len = int(visible.numel())
    common_blocks = int(visible.min().item())
    local = visible - 1
    if int(local.min().item()) <= 0:
        raise ValueError("the constrained patterns require every local block > 0")
    if common_blocks < TOPK + 1:
        raise ValueError(
            "the constrained patterns need at least 17 causal-visible blocks"
        )

    # A deterministic pseudo-random 99%-rate mask keeps runs reproducible.
    # It is shared across index heads because the constraint is per query.
    select_block0 = ((q_idx * 73 + seed) % 100) != 0

    if pattern == "b0p99_local_recent":
        candidate_slots = torch.arange(
            TOPK + 2, device=visible.device, dtype=torch.int64
        )
        candidates = local[None, :, None] - 1 - candidate_slots[None, None, :]
        candidates = candidates.expand(KV_HEADS, -1, -1)
    else:
        hot_count = 0
        if pattern == "b0p99_local_spread":
            pool_end = common_blocks - 1
        elif pattern.startswith("b0p99_local_pool"):
            pool_end = min(
                int(pattern.removeprefix("b0p99_local_pool")),
                common_blocks - 1,
            )
        elif pattern.startswith("b0p99_local_hot"):
            hot_count = int(pattern.removeprefix("b0p99_local_hot"))
            if not 0 <= hot_count <= TOPK - 2:
                raise ValueError(f"invalid constrained hot count: {hot_count}")
            pool_end = common_blocks - 1
        else:
            raise ValueError(f"unknown constrained pattern: {pattern}")

        if pool_end < TOPK:
            raise ValueError(f"{pattern} has too few nonzero common blocks")
        hot = torch.arange(1, hot_count + 1, device=visible.device, dtype=torch.int64)
        tail_begin = hot_count + 1
        tail_size = pool_end - hot_count
        if tail_size <= 0:
            raise ValueError(f"{pattern} has no tail candidate blocks")
        step = _coprime_step(tail_size)
        tail_slots = torch.arange(tail_size, device=visible.device, dtype=torch.int64)
        base = (q_idx[:, None] * TOPK + heads[None, :] * 37) % tail_size
        tail = (
            tail_begin
            + (base[:, :, None] + tail_slots[None, None, :] * step) % tail_size
        )
        tail = tail.permute(1, 0, 2)
        hot_candidates = hot[None, None, :].expand(KV_HEADS, q_len, -1)
        candidates = torch.cat((hot_candidates, tail), dim=-1)

    valid = (
        (candidates > 0)
        & (candidates < visible[None, :, None])
        & (candidates != local[None, :, None])
    )
    filler_count = TOPK - 1
    if int(valid.sum(dim=-1).min().item()) < filler_count:
        raise ValueError(f"{pattern} cannot supply {filler_count} unique fillers")
    order = torch.arange(candidates.shape[-1], device=visible.device, dtype=torch.int64)
    priority = order[None, None, :] + (~valid) * (2 * candidates.shape[-1])
    selected = torch.topk(
        priority, k=filler_count, dim=-1, largest=False, sorted=True
    ).indices
    fillers = torch.gather(candidates, dim=-1, index=selected)

    with_block0 = torch.cat(
        (
            torch.zeros((KV_HEADS, q_len, 1), device=visible.device, dtype=torch.int64),
            fillers[:, :, : TOPK - 2],
        ),
        dim=-1,
    )
    chosen_tail = torch.where(
        select_block0[None, :, None],
        with_block0,
        fillers,
    )
    return torch.cat(
        (local[None, :, None].expand(KV_HEADS, -1, -1), chosen_tail),
        dim=-1,
    )


def build_topk(pattern: str, visible: torch.Tensor, seed: int) -> torch.Tensor:
    """Build unique and bottom-right-causal-valid ``[4, Q, 16]`` TopK."""
    q_len = int(visible.numel())
    if int(visible.min().item()) < TOPK:
        raise ValueError("every query must expose at least 16 K blocks")
    q_idx = torch.arange(q_len, device=visible.device, dtype=torch.int64)
    heads = torch.arange(KV_HEADS, device=visible.device, dtype=torch.int64)
    slots = torch.arange(TOPK, device=visible.device, dtype=torch.int64)
    common_blocks = int(visible.min().item())

    if pattern.startswith("b0p99_local_"):
        result = _build_block0_local_topk(pattern, visible, q_idx, heads, seed)
    elif pattern == "recent":
        values = visible[:, None] - TOPK + slots[None, :]
        result = values[None, :, :].expand(KV_HEADS, -1, -1).clone()
    elif pattern == "spread":
        pool = common_blocks
        step = _coprime_step(pool)
        base = (q_idx[:, None] * TOPK + heads[None, :] * 37) % pool
        values = (base[:, :, None] + slots[None, None, :] * step) % pool
        result = values.permute(1, 0, 2).contiguous()
    elif pattern.startswith("pool"):
        requested = int(pattern.removeprefix("pool"))
        pool = min(requested, common_blocks)
        if pool < TOPK:
            raise ValueError(f"{pattern} requires at least {TOPK} common blocks")
        step = _coprime_step(pool)
        base = (q_idx[:, None] * TOPK + heads[None, :] * 19) % pool
        values = (base[:, :, None] + slots[None, None, :] * step) % pool
        result = values.permute(1, 0, 2).contiguous()
    elif pattern == "hot15_tail1":
        if common_blocks <= TOPK:
            raise ValueError("hot15_tail1 requires more than 16 common blocks")
        hot = slots[: TOPK - 1]
        tail = (
            TOPK
            - 1
            + (q_idx[:, None] + heads[None, :] * 53) % (common_blocks - (TOPK - 1))
        )
        result = torch.empty(
            (KV_HEADS, q_len, TOPK), dtype=torch.int64, device=visible.device
        )
        result[:, :, : TOPK - 1] = hot
        result[:, :, TOPK - 1] = tail.transpose(0, 1)
    elif pattern == "hot8_tail8":
        hot_count = TOPK // 2
        if common_blocks < TOPK:
            raise ValueError("hot8_tail8 requires at least 16 common blocks")
        hot = slots[:hot_count]
        pool = common_blocks - hot_count
        step = _coprime_step(pool)
        base = (q_idx[:, None] * hot_count + heads[None, :] * 29) % pool
        tail = (
            hot_count + (base[:, :, None] + slots[None, None, :hot_count] * step) % pool
        )
        result = torch.empty(
            (KV_HEADS, q_len, TOPK), dtype=torch.int64, device=visible.device
        )
        result[:, :, :hot_count] = hot
        result[:, :, hot_count:] = tail.permute(1, 0, 2)
    elif pattern == "random":
        generator = torch.Generator(device=visible.device).manual_seed(seed)
        keys = torch.rand(
            (KV_HEADS, q_len, common_blocks),
            generator=generator,
            device=visible.device,
        )
        result = torch.topk(keys, k=TOPK, dim=-1, largest=False, sorted=False).indices
    else:
        raise ValueError(f"unknown pattern: {pattern}")

    result = result.to(torch.int32).contiguous()
    if tuple(result.shape) != (KV_HEADS, q_len, TOPK):
        raise AssertionError(f"bad TopK shape: {tuple(result.shape)}")
    sorted_values = result.sort(dim=-1).values
    if torch.any(sorted_values[:, :, 1:] == sorted_values[:, :, :-1]).item():
        raise AssertionError(f"{pattern} generated duplicate TopK indices")
    if torch.any(result < 0).item():
        raise AssertionError(f"{pattern} generated negative TopK indices")
    if torch.any(result >= visible[None, :, None]).item():
        raise AssertionError(f"{pattern} generated a non-causal TopK index")
    return result
