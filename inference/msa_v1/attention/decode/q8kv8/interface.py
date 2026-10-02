"""FlashInfer-backed Q8KV8 paged sparse decode attention."""

import torch

from inference.msa_v1.attention.decode._flashinfer_interface import (
    _BatchDecodeWithPagedKVCacheWrapperBase,
)


class BatchDecodeWithPagedKVCacheWrapper(_BatchDecodeWithPagedKVCacheWrapperBase):
    """Run sparse decode with independent pages for each query and KV head."""

    storage_dtype = torch.float8_e4m3fn


__all__ = ["BatchDecodeWithPagedKVCacheWrapper"]
