"""Construct cache views with identical logical values and distinct storage."""

import torch

CACHE_LAYOUTS = (
    "contiguous",
    "page",
    "head",
    "token",
    "permuted",
    "combined",
    "offset",
)


def scale_cache_view(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    """Vary scale page placement without changing packed in-page storage."""
    scale_layout = layout if layout in ("contiguous", "offset") else "page"
    return cache_view(tensor, scale_layout, alignment_bytes=16)


def cache_view(
    tensor: torch.Tensor, layout: str, *, alignment_bytes: int = 16
) -> torch.Tensor:
    """Keep the channel axis contiguous while varying outer physical layout."""
    if layout in ("singleton_zero", "singleton_unaligned"):
        strides = list(tensor.stride())
        assert 1 in tensor.shape[:-1]
        for axis, size in enumerate(tensor.shape[:-1]):
            if size == 1:
                strides[axis] = 0 if layout == "singleton_zero" else 1
        return tensor.as_strided(tensor.shape, strides)
    if layout not in CACHE_LAYOUTS:
        raise ValueError(f"unknown cache layout: {layout}")
    if layout == "contiguous":
        return tensor.clone(memory_format=torch.contiguous_format)
    shape = tuple(tensor.shape)
    order = list(range(tensor.ndim - 1))
    if layout in ("permuted", "combined"):
        order.reverse()
    strides = [0] * tensor.ndim
    strides[-1] = 1
    alignment = alignment_bytes // tensor.element_size()
    extent = shape[-1]
    for axis in reversed(order):
        if (
            layout == "combined"
            or (layout == "page" and axis == 0)
            or (layout == "token" and axis == tensor.ndim - 2)
            or (layout == "head" and tensor.ndim == 4 and axis == 1)
        ):
            extent = ((extent + alignment - 1) // alignment + 1) * alignment
        strides[axis] = extent
        extent *= shape[axis]
    offset = alignment if layout in ("combined", "offset") else 0
    storage = torch.empty(extent + offset, dtype=tensor.dtype, device=tensor.device)
    storage.view(torch.uint8).fill_(0x7F)
    view = storage.as_strided(shape, strides, offset)
    view.copy_(tensor)
    return view
