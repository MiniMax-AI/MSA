"""FlashInfer-backed BF16 paged sparse decode attention."""

import torch

from inference.msa_v1.attention.decode._flashinfer_interface import (
    _BatchDecodeWithPagedKVCacheWrapperBase,
)


class BatchDecodeWithPagedKVCacheWrapper(_BatchDecodeWithPagedKVCacheWrapperBase):
    """Run sparse decode with independent pages for each query and KV head."""

    storage_dtype = torch.bfloat16


__all__ = ["BatchDecodeWithPagedKVCacheWrapper"]
