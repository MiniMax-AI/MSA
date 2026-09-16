"""Pinned real 192K/CP16 release manifest for MSA v1."""

from __future__ import annotations

from functools import cache

from datas.training.cases import MSA_V1_SPEC
from tests.training.cases import (
    FULL_CASE_COUNT,
    SMOKE_CASE_COUNT,
    MsaTestCase,
    build_manifest,
    manifest_digest,
    validate_manifest,
)

EXPECTED_MANIFEST_DIGEST = (
    "8e4dc65bc9a83fb26d99fe53d0a38c06331e3a973dfbd6e7f2b8ae9cf02f0e6d"
)


@cache
def full_manifest() -> tuple[MsaTestCase, ...]:
    return build_manifest("full_gpu", MSA_V1_SPEC)


@cache
def smoke_manifest() -> tuple[MsaTestCase, ...]:
    return build_manifest("smoke", MSA_V1_SPEC)


__all__ = [
    "EXPECTED_MANIFEST_DIGEST",
    "FULL_CASE_COUNT",
    "SMOKE_CASE_COUNT",
    "MsaTestCase",
    "full_manifest",
    "manifest_digest",
    "smoke_manifest",
    "validate_manifest",
]
