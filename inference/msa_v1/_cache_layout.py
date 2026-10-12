"""Host-only validation of paged cache tensor layouts."""

import torch


def normalize_cache_singletons(tensor: torch.Tensor) -> torch.Tensor:
    """Keep unused strides out of CuTe's static broadcast and alignment contracts."""
    strides = list(tensor.stride())
    alignment = 16 // tensor.element_size()
    unused = [
        axis
        for axis, (size, stride) in enumerate(zip(tensor.shape, strides))
        if size == 1 and (stride <= 0 or stride % alignment)
    ]
    if not unused:
        return tensor
    span = max(size * stride for size, stride in zip(tensor.shape, strides) if size > 1)
    canonical_stride = (span + alignment - 1) // alignment * alignment
    for axis in unused:
        strides[axis] = canonical_stride
    return tensor.as_strided(tensor.shape, strides)


def check_cache_layout(tensor: torch.Tensor, *, name: str, alignment: int = 16) -> None:
    """Validate channel-contiguous, aligned cache views without reading data."""
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.ndim == 0 or tensor.stride(-1) != 1:
        raise ValueError(f"{name} must have unit stride in its last dimension")
    if tensor.data_ptr() % alignment:
        raise ValueError(f"{name} must have a {alignment}-byte aligned data pointer")
    for axis, (size, stride) in enumerate(zip(tensor.shape[:-1], tensor.stride()[:-1])):
        if size > 1 and stride <= 0:
            raise ValueError(f"{name} axis {axis} must have positive stride")
        if size > 1 and stride * tensor.element_size() % alignment:
            raise ValueError(
                f"{name} axis {axis} byte stride must be a multiple of {alignment}"
            )
