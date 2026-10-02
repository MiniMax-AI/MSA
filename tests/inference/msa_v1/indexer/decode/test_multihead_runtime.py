"""Cross-head, Graph, cache and workspace isolation contracts."""

import gc
import importlib
import os
import threading
import weakref

import pytest
import torch


def _synchronize(stream=None):
    watchdog = threading.Timer(30.0, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        if stream is None:
            torch.cuda.synchronize()
        else:
            stream.synchronize()
    finally:
        watchdog.cancel()


@pytest.mark.parametrize("kind", ("q8kv8", "q8kv4", "bf16"))
@pytest.mark.parametrize("heads", (1, 2, 4))
def test_multihead_runtime(kind, heads):
    api = importlib.import_module(f"inference.msa_v1.indexer.decode.{kind}.interface")
    ref = importlib.import_module(
        f"tests.inference.msa_v1.indexer.decode.{kind}.reference"
    )
    device = torch.device("cuda")
    wrappers, inputs = [], []
    keys = None
    for batch, pages in ((3, 19), (5, 21), (129, 5), (7, 3)):
        lengths = (torch.arange(batch, dtype=torch.int32) * 17) % 100 + (
            pages - 1
        ) * 128
        if (batch, pages) == (7, 3):
            # Empty requests at both ends and in the middle exercise prefix ties.
            lengths = torch.tensor([8, 129, 128, 384, 8, 257, 8], dtype=torch.int32)
        values = ref.make_inputs(
            batch, pages, lengths, seed=1701, device=device, num_index_heads=heads
        )
        q, k, *tail = values
        scale, table, seq = tail if kind == "q8kv4" else (None, *tail)
        kwargs = {"k_scale": scale} if scale is not None else {}
        wrapper = api.BatchDecodeIndexerWithPagedKVCacheWrapper()
        wrapper.plan(table, seq, num_index_heads=heads)
        proxy = wrapper._proxy_score
        scheduler_before = proxy._workspace_buffer.clone()
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        schedule = (
            proxy._workspace_buffer[: (batch + sm_count + 2) * 4]
            .view(torch.int32)
            .cpu()
        )
        history_pages = (lengths - 1) // 128
        torch.testing.assert_close(
            schedule[: batch + 1],
            torch.cat(
                (
                    torch.zeros(1, dtype=torch.int32),
                    history_pages.cumsum(0).to(torch.int32),
                )
            ),
        )
        bounds = schedule[batch + 1 :]
        assert bounds[0] == 0 and bounds[-1] == history_pages.sum()
        counts = bounds.diff()
        assert counts.min() >= 0 and counts.max() - counts.min() <= 1
        start_fields = 2 if kind != "q8kv4" else 3
        start_offset = batch + sm_count + 2
        worker_start = (
            proxy._workspace_buffer[
                start_offset * 4 : (start_offset + start_fields * sm_count) * 4
            ]
            .view(torch.int32)
            .view(start_fields, sm_count)
            .cpu()
        )
        prefix = schedule[: batch + 1]
        first_request = torch.searchsorted(prefix[1:], bounds[:-1], right=True)
        torch.testing.assert_close(worker_start[0], first_request.to(torch.int32))
        torch.testing.assert_close(worker_start[1], bounds[:-1] - prefix[first_request])
        if kind == "q8kv4":
            request_end = prefix[(first_request + 1).clamp(max=batch)]
            first_count = (torch.minimum(bounds[1:], request_end) - bounds[:-1]).clamp(
                max=16
            )
            torch.testing.assert_close(worker_start[2], first_count)
        out = torch.empty(heads, batch * 8, 16, dtype=torch.int32, device=device)
        assert wrapper.run(q, k, out=out, **kwargs) is out
        _synchronize()
        assert torch.equal(proxy._workspace_buffer, scheduler_before)
        original = out.clone()
        expected = ref.indexer_gemm_reference(*values)
        local = (seq[:, None] - 8 + torch.arange(8, device=device)) // 128
        mask = torch.arange(pages, device=device).view(1, 1, -1) < local.reshape(
            1, -1, 1
        )
        torch.testing.assert_close(
            wrapper._proxy_scores.masked_select(mask),
            expected.masked_select(mask),
            atol=1e-4,
            rtol=1e-4,
        )
        if kind != "q8kv4":
            current = set(
                importlib.import_module(
                    "inference.msa_v1.indexer.decode._interface"
                )._COMPILE_CACHE
            )
        else:
            jit = importlib.import_module("inference.msa_v1.indexer.decode.q8kv4.jit")
            current = jit._load_extension_for_arch.cache_info().misses
        if keys is not None:
            assert keys == current
        keys = current
        unchanged_scores = wrapper._proxy_scores[1:].masked_select(mask).clone()
        changed = q.float()
        changed[:, :, 0] *= -1
        changed = changed.to(q.dtype)
        wrapper.run(changed, k, out=out, **kwargs)
        _synchronize()
        assert torch.equal(out[1:], original[1:])
        assert torch.equal(
            wrapper._proxy_scores[1:].masked_select(mask), unchanged_scores
        )
        with pytest.raises(ValueError):
            wrapper.run(q, k, out=out.view(-1, 16), **kwargs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper.run(q, k, out=out, **kwargs)
        for _ in range(3):
            wrapper._proxy_scores.fill_(float("nan"))
            graph.replay()
            _synchronize()
            assert torch.equal(out, original)
            torch.testing.assert_close(
                wrapper._proxy_scores.masked_select(mask),
                expected.masked_select(mask),
                atol=1e-4,
                rtol=1e-4,
            )
        previous_local = local.clone()
        previous_history_pages = ((seq - 1) // 128).clone()
        seq.sub_(128).clamp_(min=8)
        table.copy_(table.flip(1))
        wrapper.plan(table, seq, num_index_heads=heads)
        graph.replay()
        _synchronize()
        expected = ref.indexer_gemm_reference(*values)
        local = (seq[:, None] - 8 + torch.arange(8, device=device)) // 128
        assert torch.equal(local, (previous_local - 1).clamp(min=0))
        assert torch.equal((seq - 1) // 128, (previous_history_pages - 1).clamp(min=0))
        mask = torch.arange(pages, device=device).view(1, 1, -1) < local.reshape(
            1, -1, 1
        )
        torch.testing.assert_close(
            wrapper._proxy_scores.masked_select(mask),
            expected.masked_select(mask),
            atol=1e-4,
            rtol=1e-4,
        )
        from tests.inference.msa_v1.indexer._common.topk_select.reference import (
            assert_quantized_topk_contract,
        )

        assert_quantized_topk_contract(
            expected.reshape(-1, pages).cpu().numpy(),
            (local + 1).reshape(-1).repeat(heads).cpu().numpy(),
            out.reshape(-1, 16).cpu().numpy(),
        )
        original = out.clone()
        wrappers.append(wrapper)
        inputs.append((q, k, kwargs, out, original))
        del graph
    assert wrappers[0]._proxy_scores.data_ptr() != wrappers[1]._proxy_scores.data_ptr()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream, wrapper, (q, k, kwargs, out, _) in zip(
        streams, wrappers[:2], inputs[:2]
    ):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            wrapper.run(q, k, out=out, **kwargs)
    for stream in streams:
        _synchronize(stream)
    for *_, out, original in inputs:
        assert torch.equal(out, original)
    reference = weakref.ref(wrappers[0]._proxy_scores)
    wrappers.clear()
    gc.collect()
    assert reference() is None
