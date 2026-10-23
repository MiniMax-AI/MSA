"""PDL launch of the decode indexer must not change any score."""

from __future__ import annotations

import pytest
import torch

from inference.msa_v1.indexer.decode.q8kv4.interface import _BatchDecodeProxyScoreWrapper
from tests.inference.msa_v1.indexer.decode.q8kv4.reference import make_inputs


@pytest.mark.parametrize("use_cuda_graph", (False, True))
@pytest.mark.parametrize("batch", (1, 32, 129))
def test_pdl_launch_is_bit_identical(batch: int, use_cuda_graph: bool) -> None:
    device = torch.device("cuda")
    max_pages = 8
    # Mixed lengths so that some CTAs own history pages (loaded before the grid
    # dependency wait) and some own only the page that holds the query tokens.
    lengths = torch.arange(batch, dtype=torch.int32) * 97 % (max_pages * 128 - 7) + 8
    q, packed_k, k_scale, page_table, seq_lens = make_inputs(
        batch, max_pages, lengths, seed=11, device=device
    )

    def run(pdl: bool) -> torch.Tensor:
        output = torch.full(
            (1, batch * 8, max_pages), 123.0, dtype=torch.float32, device=device
        )
        kwargs = {}
        if use_cuda_graph:
            kwargs = dict(
                page_table_buffer=torch.empty_like(page_table),
                seq_lens_buffer=torch.empty_like(seq_lens),
            )
        wrapper = _BatchDecodeProxyScoreWrapper(use_cuda_graph=use_cuda_graph, **kwargs)
        wrapper.plan(page_table, seq_lens)
        if not use_cuda_graph:
            wrapper.run(q, packed_k, k_scale, out=output, pdl=pdl)
            torch.cuda.synchronize()
            return output
        wrapper.run(q, packed_k, k_scale, out=output, pdl=pdl)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper.run(q, packed_k, k_scale, out=output, pdl=pdl)
        output.fill_(123.0)
        graph.replay()
        torch.cuda.synchronize()
        return output

    torch.testing.assert_close(run(True), run(False), atol=0, rtol=0)
