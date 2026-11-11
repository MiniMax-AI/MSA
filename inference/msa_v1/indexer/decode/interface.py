"""Public score-only decode entry for callers that select pooled blocks."""

from inference.msa_v1.indexer.decode._interface import _BatchDecodeProxyScoreWrapper


class BatchDecodeBlockScoreWrapper(_BatchDecodeProxyScoreWrapper):
    """Return FP32 per-head maxima over 128 original-token blocks.

    One stored key covers ``pooling_factor`` (1, 2, or 4) original tokens, so a
    block holds ``128 // pooling_factor`` keys. ``page_size`` is 64 or 128
    original tokens; each physical page stores ``page_size // pooling_factor``
    key rows, and a page-64 block combines two independently mapped pages.
    Historical-only mode is the default; ``include_partial_block=True`` also
    scores completed groups in the current block. All invalid blocks contain
    negative infinity in both modes.
    """

    _initialize_output = True


__all__ = ["BatchDecodeBlockScoreWrapper"]
