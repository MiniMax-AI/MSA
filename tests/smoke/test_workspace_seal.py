# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Unit tests for fmha_sm100.api's workspace prime/seal/tripwire contract.

Graph-captured kernels (the serving engine's decode/verify CUDA graphs and
the MSA prefill piecewise graphs) bake workspace POINTERS returned by
``_alloc_workspace_buf``; growing a tagged buffer relocates it and strands
those pointers. ``prime_workspace_cache`` + ``seal_workspace_cache`` make the
post-priming sizes immutable: any later growth must raise WorkspaceGrowthError
instead of silently reallocating. These tests exercise the cache in pure
allocation semantics — no kernels, no CUDA: ``torch.empty`` inside api.py is
patched to allocate on CPU while the device argument stays metadata.
"""

import unittest
from unittest import mock

import torch

import fmha_sm100.api as api

_DEV = torch.device("cuda", 0)  # metadata only: supplies the cache slot index
_REAL_EMPTY = torch.empty


def _cpu_empty(size, dtype, device=None):
    return _REAL_EMPTY(size, dtype=dtype, device="cpu")


def _alloc(tag, size, dtype=torch.int32):
    with mock.patch.object(api.torch, "empty", side_effect=_cpu_empty):
        return api._alloc_workspace_buf(tag, size, _DEV, dtype)


def _prime(tag, size, dtype=torch.int32):
    with mock.patch.object(api.torch, "empty", side_effect=_cpu_empty):
        api.prime_workspace_cache(tag, size, _DEV, dtype)


class TestWorkspaceSeal(unittest.TestCase):
    def setUp(self):
        self._saved_cache = api._workspace_cache
        self._saved_sealed = api._workspace_sealed
        api._workspace_cache = [[None] * api._BuffTag.Total for _ in range(16)]
        api._workspace_sealed = False

    def tearDown(self):
        api._workspace_cache = self._saved_cache
        api._workspace_sealed = self._saved_sealed

    def test_unsealed_growth_keeps_legacy_semantics(self):
        tag = api._BuffTag.sparse_topk_workspace
        a = _alloc(tag, 128)
        self.assertEqual(a.shape[0], 128)
        # smaller request: existing buffer returned
        self.assertIs(_alloc(tag, 64), a)
        # unsealed growth reallocates exactly as before (no raise)
        b = _alloc(tag, 256)
        self.assertIsNot(b, a)
        self.assertEqual(b.shape[0], 256)
        self.assertIs(api._get_workspace_buf(tag, _DEV), b)

    def test_prime_returns_largest_unsealed(self):
        tag = api._BuffTag.sparse_topk_workspace
        _prime(tag, 512)
        _prime(tag, 128)  # no-op: cached buffer already covers it
        buf = api._get_workspace_buf(tag, _DEV)
        self.assertEqual(buf.shape[0], 512)
        _prime(tag, 1024)  # unsealed prime may still grow forward
        self.assertEqual(api._get_workspace_buf(tag, _DEV).shape[0], 1024)

    def test_seal_refuses_growth_but_serves_reads(self):
        tag = api._BuffTag.sparse_topk_workspace
        _prime(tag, 512)
        api.seal_workspace_cache()
        self.assertTrue(api.workspace_cache_sealed())
        # reads at/under the primed size keep working, same object identity
        self.assertEqual(_alloc(tag, 512).shape[0], 512)
        self.assertEqual(_alloc(tag, 200).shape[0], 512)
        with self.assertRaises(api.WorkspaceGrowthError):
            _alloc(tag, 513)

    def test_seal_allows_first_touch_of_untouched_tag(self):
        api.seal_workspace_cache()
        tag = api._BuffTag.workspace_o
        buf = _alloc(tag, 64, dtype=torch.bfloat16)  # first touch: legal
        self.assertEqual(buf.shape[0], 64)
        # but the once-allocated tag can never grow again
        with self.assertRaises(api.WorkspaceGrowthError):
            _alloc(tag, 65, dtype=torch.bfloat16)

    def test_prime_after_seal(self):
        tag = api._BuffTag.workspace_o
        _prime(tag, 256, dtype=torch.bfloat16)
        api.seal_workspace_cache()
        _prime(tag, 200, dtype=torch.bfloat16)  # covered: silent no-op
        self.assertEqual(api._get_workspace_buf(tag, _DEV).shape[0], 256)
        with self.assertRaises(api.WorkspaceGrowthError):
            _prime(tag, 257, dtype=torch.bfloat16)

    def test_error_is_runtime_error(self):
        self.assertTrue(issubclass(api.WorkspaceGrowthError, RuntimeError))

    def test_per_device_slots_are_independent(self):
        tag = api._BuffTag.sparse_topk_workspace
        _prime(tag, 128)
        api.seal_workspace_cache()
        with self.assertRaises(api.WorkspaceGrowthError):
            _alloc(tag, 129)
        # a different device id still gets its first-touch allocation
        other = torch.device("cuda", 1)
        with mock.patch.object(api.torch, "empty", side_effect=_cpu_empty):
            buf = api._alloc_workspace_buf(tag, 8, other, torch.int32)
        self.assertEqual(buf.shape[0], 8)


if __name__ == "__main__":
    unittest.main()
