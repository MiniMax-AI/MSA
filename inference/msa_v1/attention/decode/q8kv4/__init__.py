"""High-performance SM100 Q8KV4 paged sparse decode attention."""

from .interface import BatchDecodeWithPagedKVCacheWrapper

__all__ = ["BatchDecodeWithPagedKVCacheWrapper"]

try:
    import ctypes

    import torch
    import tvm_ffi

    _FP8_DTYPE_MAP = {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
    }

    def _dlpack_capsule_set_dtype_int8(capsule):
        get_ptr = ctypes.pythonapi.PyCapsule_GetPointer
        get_ptr.restype = ctypes.c_void_p
        get_ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]
        dl_ptr = get_ptr(capsule, b"dltensor")
        dtype_addr = dl_ptr + 20
        ctypes.c_uint8.from_address(dtype_addr).value = 0
        ctypes.c_uint8.from_address(dtype_addr + 1).value = 8
        ctypes.c_uint16.from_address(dtype_addr + 2).value = 1

    def _tvm_to_torch(x):
        if type(x).__module__ == "tvm_ffi.core" and type(x).__name__ == "Tensor":
            fp8_dtype = _FP8_DTYPE_MAP.get(str(x.dtype))
            if fp8_dtype is not None:
                capsule = x._to_dlpack()
                _dlpack_capsule_set_dtype_int8(capsule)
                return torch.from_dlpack(capsule).view(fp8_dtype)
            return torch.from_dlpack(x)
        if type(x).__module__ == "tvm_ffi.container" and type(x).__name__ == "Map":
            return {str(k): _tvm_to_torch(x[k]) for k in x}
        return x

    from .jit import (
        get_fmha_fwd_variant,
        get_plan_fn,
        get_reduction_module,
    )

    def _jit_get_fmha_fwd_sparse_variant(
        topk,
        split_kv=False,
        device=None,
        gqa_ratio=16,
        output_mode="bf16",
    ):
        return get_fmha_fwd_variant(
            topk=int(topk),
            split_kv=bool(split_kv),
            gqa_ratio=int(gqa_ratio),
            output_mode=str(output_mode),
            device=None if device is None else int(device),
        )._fn

    def _jit_get_plan(device=None):
        return get_plan_fn(None if device is None else int(device)).plan

    def _jit_get_reduction(device=None):
        return get_reduction_module(None if device is None else int(device)).reduction

    tvm_ffi.register_global_func(
        "msa_v1.attention.decode.q8kv4.jit_get_fmha_fwd_sparse_variant",
        _jit_get_fmha_fwd_sparse_variant,
        override=True,
    )
    tvm_ffi.register_global_func(
        "msa_v1.attention.decode.q8kv4.jit_get_plan",
        _jit_get_plan,
        override=True,
    )
    tvm_ffi.register_global_func(
        "msa_v1.attention.decode.q8kv4.jit_get_reduction",
        _jit_get_reduction,
        override=True,
    )

except ImportError:
    pass
