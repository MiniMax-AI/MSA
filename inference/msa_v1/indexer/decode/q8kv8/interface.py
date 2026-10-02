"""Q8KV8 paged decode indexer public interface."""

import torch

from inference.msa_v1.indexer.decode._interface import (
    _BatchDecodeIndexerWrapperBase,
)


class BatchDecodeIndexerWithPagedKVCacheWrapper(_BatchDecodeIndexerWrapperBase):
    """Select historical pages independently for each local index head."""

    storage_dtype = torch.float8_e4m3fn


__all__ = ["BatchDecodeIndexerWithPagedKVCacheWrapper"]
