"""Shared 64-bit fingerprinting and host-side open-addressing helpers.

The project previously used 32-bit FNV-1a fingerprints. At multi-million
candidate scales the birthday-bound collision rate becomes material, so all
new ranking/CELF paths use FNV-1a-64. The hash table is collision-safe in the
usual hash-table sense: equal fingerprints are deduplicated, unequal
fingerprints are kept distinct through linear probing. As with any fixed-size
fingerprint, a mathematically adversarial 64-bit collision is still possible;
this module deliberately documents that limitation instead of claiming
cryptographic collision-proofness.
"""

from __future__ import annotations

import json
import mmap
import os
from typing import Optional, Tuple

import numpy as np

FNV1A64_OFFSET = 14695981039346656037
FNV1A64_PRIME = 1099511628211
UINT64_MASK = (1 << 64) - 1

FNV_CACHE_VERSION = 2
FNV_CACHE_ALGO = "fnv1a64"


def fast_fnv1a_hash_64(data: bytes) -> int:
    """Fast pure-Python FNV-1a-64 over a bytes-like object."""
    h = FNV1A64_OFFSET
    for byte in data:
        h ^= byte
        h = (h * FNV1A64_PRIME) & UINT64_MASK
    return h


def fast_fnv1a_hash_32(data: bytes) -> int:
    """Legacy compatibility helper; ranking/CELF no longer depend on it."""
    h = 2166136261
    for byte in data:
        h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
    return h


def table_size_for_count(count: int, load_factor: float = 0.50) -> int:
    """Return a power-of-two table size at or below *load_factor*."""
    if count < 1:
        return 2
    target = max(2, int(np.ceil(count / load_factor)))
    size = 1
    while size < target:
        size <<= 1
    return size


def _slot_for_key(keys: np.ndarray, mask: int) -> np.ndarray:
    keys_u = np.asarray(keys, dtype=np.uint64)
    # SplitMix64 finalizer gives a better slot distribution than using the
    # raw low bits of FNV-1a. All arithmetic is deliberately uint64 wraparound.
    z = keys_u + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z & np.uint64(mask)).astype(np.int64)


def build_open_addressing_table_uint64(keys, table_size: Optional[int] = None):
    """Build a uint64 linear-probing table without Python per-key loops.

    Returns ``(hash_table, states)`` where ``states`` is uint32 and uses:
    0 = empty, 2 = occupied/ready. A separate state array means key 0 is
    valid. Duplicate keys are collapsed before insertion.
    """
    arr = np.unique(np.asarray(keys, dtype=np.uint64))
    if table_size is None:
        table_size = table_size_for_count(int(arr.size))
    table_size = int(table_size)
    if table_size < table_size_for_count(int(arr.size)):
        raise ValueError("table_size is too small for the requested load factor")
    mask = table_size - 1

    table = np.zeros(table_size, dtype=np.uint64)
    states = np.zeros(table_size, dtype=np.uint32)
    if arr.size == 0:
        return table, states

    pending_keys = arr
    pending_slots = _slot_for_key(pending_keys, mask)
    while pending_keys.size:
        _, first = np.unique(pending_slots, return_index=True)
        winner = np.zeros(pending_keys.size, dtype=bool)
        winner[first] = True

        cand_slots = pending_slots[winner]
        cand_keys = pending_keys[winner]
        free = states[cand_slots] == 0
        if np.any(free):
            table[cand_slots[free]] = cand_keys[free]
            states[cand_slots[free]] = 2

        retry = ~winner
        winner_pos = np.flatnonzero(winner)
        blocked = ~free
        if np.any(blocked):
            retry[winner_pos[blocked]] = True

        pending_keys = pending_keys[retry]
        pending_slots = (pending_slots[retry] + 1) & mask

    return table, states


def cache_paths(source_path: str, max_len: int):
    base = f"{source_path}.{FNV_CACHE_ALGO}.max{int(max_len)}"
    return base + ".npy", base + ".json"


def load_cached_hashes(source_path: str, max_len: int):
    """Load a validated uint64 fingerprint cache or return ``None``."""
    try:
        source_stat = os.stat(source_path)
        cache_path, meta_path = cache_paths(source_path, max_len)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if (
            meta.get("version") != FNV_CACHE_VERSION
            or meta.get("algorithm") != FNV_CACHE_ALGO
            or meta.get("source_size") != source_stat.st_size
            or meta.get("source_mtime_ns") != source_stat.st_mtime_ns
            or meta.get("max_len") != int(max_len)
        ):
            return None
        cached = np.load(cache_path, allow_pickle=False)
        if cached.ndim != 1 or cached.dtype != np.dtype(np.uint64):
            return None
        if cached.size > 1 and not np.all(cached[1:] > cached[:-1]):
            return None
        return np.asarray(cached), int(meta.get("n_skipped", 0))
    except (OSError, EOFError, ValueError, TypeError, UnicodeError, json.JSONDecodeError):
        return None


def save_cached_hashes(source_path: str, max_len: int, hashes, n_skipped: int) -> bool:
    """Atomically write the uint64 fingerprint cache; failures are ignored."""
    tmp_npy = tmp_meta = None
    try:
        source_stat = os.stat(source_path)
        cache_path, meta_path = cache_paths(source_path, max_len)
        cache_dir = os.path.dirname(cache_path) or "."
        os.makedirs(cache_dir, exist_ok=True)
        pid = os.getpid()
        tmp_npy = f"{cache_path}.tmp.{pid}"
        tmp_meta = f"{meta_path}.tmp.{pid}"
        arr = np.asarray(hashes, dtype=np.uint64)
        with open(tmp_npy, "wb") as f:
            np.save(f, arr, allow_pickle=False)
        os.replace(tmp_npy, cache_path)
        meta = {
            "version": FNV_CACHE_VERSION,
            "algorithm": FNV_CACHE_ALGO,
            "source_size": int(source_stat.st_size),
            "source_mtime_ns": int(source_stat.st_mtime_ns),
            "max_len": int(max_len),
            "n_hashes": int(arr.size),
            "n_skipped": int(n_skipped),
        }
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, separators=(",", ":"))
        os.replace(tmp_meta, meta_path)
        return True
    except (OSError, ValueError, TypeError):
        for p in (tmp_npy, tmp_meta):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        return False
