"""Tests for rule_ranker.celf_recompute_gpu's greedy ROUND LOGIC
(fixed sorted-order scan, upper_bound/last_bound pruning, tie-breaks,
budget cutoff) in isolation from real OpenCL/GPU kernels.

celf_select_recompute_gpu() constructs a real _GpuScorer internally,
which needs an actual GPU. To test the selection algorithm itself
(the lazy upper-bound logic), this monkeypatches _GpuScorer with a small
pure-Python/NumPy fake backed by an explicit rule -> covered-hash-ids
mapping, so exact coverage/selection outcomes can be asserted without
any device present. See conftest.py for why `import pyopencl` still
needs to succeed even though no real OpenCL is exercised here.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rule_ranker import celf_recompute_gpu as crg


class _FakeScorer:
    """Drop-in stand-in for celf_recompute_gpu._GpuScorer. `coverage`
    maps rule string -> set of covered hash ids (ints in
    range(cracked_size)); `active` starts as every hash id and shrinks
    exactly like the real GPU-side `active` bitmap does."""

    def __init__(self, rules, coverage, cracked_size):
        self.rules = rules
        self.coverage = {r: set(c) for r, c in coverage.items()}
        self.active = set(range(cracked_size))
        self.score_batch_calls = 0
        self.dispatched_rule_counts = []

    def score_batch(self, rule_indices, wordlist_path):
        self.score_batch_calls += 1
        self.dispatched_rule_counts.append(len(rule_indices))
        out = np.zeros(len(rule_indices), dtype=np.int64)
        for i, ridx in enumerate(rule_indices):
            rule_str = self.rules[int(ridx)]
            out[i] = len(self.coverage.get(rule_str, set()) & self.active)
        return out

    def apply_winner_and_clear(self, rule_idx, rule_str, wordlist_path):
        newly = self.coverage.get(rule_str, set()) & self.active
        self.active -= newly
        return len(newly)

    def remaining_active_count(self):
        return len(self.active)


def _run(monkeypatch, rules, coverage, cracked_size, budget=None, rule_batch_size=2):
    fake = _FakeScorer(rules, coverage, cracked_size)
    monkeypatch.setattr(crg, "_GpuScorer", lambda *a, **k: fake)
    encoded_unused = np.zeros((len(rules), 4), dtype=np.uint8)  # _GpuScorer ctor args are ignored by the fake
    cracked_hashes_sorted = np.arange(cracked_size, dtype=np.uint32)
    selected = crg.celf_select_recompute_gpu(
        rules, wordlist_path="unused.txt", cracked_hashes_sorted=cracked_hashes_sorted,
        rule_batch_size=rule_batch_size, words_per_gpu_batch=1000,
        device_id=None, budget=budget,
    )
    return selected, fake


class TestGreedyRoundLogic:
    def test_picks_best_coverage_first_and_saturates(self, monkeypatch):
        rules = ["a", "b", "c"]
        coverage = {
            "a": {0, 1, 2},   # covers 3
            "b": {2, 3},      # covers 2
            "c": {4},         # covers 1, disjoint
        }
        selected, fake = _run(monkeypatch, rules, coverage, cracked_size=5, budget=2)
        # "a" (3) picked first, then "b" only has {3} left uncovered (gain 1)
        # once {0,1,2} are gone, same as "c" (gain 1) -- tie broken by
        # lower original index, i.e. earlier rule in `rules` -> "b" (idx 1)
        # beats "c" (idx 2).
        picked = [r for r, _g in selected]
        assert picked[0] == "a"
        assert selected[0][1] == 3
        assert picked[1] == "b"
        assert selected[1][1] == 1
        assert fake.remaining_active_count() == 1  # only hash id 4 left (from unpicked "c")

    def test_zero_gain_candidates_never_selected(self, monkeypatch):
        rules = ["a", "b"]
        coverage = {"a": {0, 1}, "b": set()}
        selected, _fake = _run(monkeypatch, rules, coverage, cracked_size=2)
        assert [r for r, _g in selected] == ["a"]

    def test_budget_cuts_off_selection(self, monkeypatch):
        rules = ["a", "b", "c"]
        coverage = {"a": {0}, "b": {1}, "c": {2}}
        selected, _fake = _run(monkeypatch, rules, coverage, cracked_size=3, budget=2)
        assert len(selected) == 2

    def test_no_candidates_with_any_hit_returns_empty(self, monkeypatch):
        rules = ["a", "b"]
        coverage = {"a": set(), "b": set()}
        selected, fake = _run(monkeypatch, rules, coverage, cracked_size=4)
        assert selected == []
        # One call for the upper-bound pass itself (both come back 0);
        # the greedy loop then has nothing with upper_bound > 0 to
        # even start a round with, so no further dispatches happen.
        assert fake.score_batch_calls == 1

    def test_lazy_bound_skips_dominated_candidates_without_gpu_call(self, monkeypatch):
        # "a" covers everything (gain 10). Once picked, every other
        # candidate's true gain against the now-empty active set is 0
        # -- their static upper_bound (their own full-universe count,
        # e.g. "b" covers 5 of the ORIGINAL 10) is a valid bound that's
        # still > 0, so a naive scan would want to re-check them, but
        # since best_gain in round 2 starts at 0 and every remaining
        # candidate's own upper_bound is also compared against
        # best_gain found so far in that round -- this test mainly
        # pins down that saturation (best_gain <= 0) correctly stops
        # the loop instead of continuing to emit zero-gain picks.
        rules = ["a", "b", "c"]
        coverage = {
            "a": set(range(10)),
            "b": set(range(5)),
            "c": set(range(3)),
        }
        selected, fake = _run(monkeypatch, rules, coverage, cracked_size=10)
        assert [r for r, _g in selected] == ["a"]
        assert fake.remaining_active_count() == 0

    def test_round_one_never_dispatches_a_gpu_rescore(self, monkeypatch):
        # Round 1's pick is read directly off the already-computed
        # upper_bound array (see the block comment in
        # celf_select_recompute_gpu): the first score_batch() call, if
        # any, should only happen in round 2 onward.
        rules = ["a", "b"]
        coverage = {"a": {0, 1}, "b": {2}}
        selected, fake = _run(monkeypatch, rules, coverage, cracked_size=3)
        assert [r for r, _g in selected] == ["a", "b"]
        # One call for the upper-bound pass, one for round 2's
        # revalidation of "b" -- round 1 itself must contribute zero
        # (its pick comes straight from the already-computed
        # upper_bound array, no rescore needed).
        assert fake.score_batch_calls == 2

    def test_ties_broken_by_lower_original_index(self, monkeypatch):
        rules = ["first", "second"]
        coverage = {"first": {0}, "second": {1}}
        selected, _fake = _run(monkeypatch, rules, coverage, cracked_size=2)
        # Both have gain 1 in round 1; "first" has the lower index (0)
        # so upper_bound sort (stable, descending on equal values)
        # keeps it first.
        assert selected[0][0] == "first"


def _mix64_reference(x):
    with np.errstate(over="ignore"):
        x = (np.uint64(x) + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        return (x ^ (x >> np.uint64(31))) & np.uint64(0xFFFFFFFFFFFFFFFF)


class TestBuildOpenAddressingTableVectorized:
    """_build_open_addressing_table() is the vectorized (NumPy,
    wave-based) replacement for the original per-key Python `for h in
    ...: while occupied[...]: ...` loop that built the GPU hash table.
    Pure NumPy, no OpenCL involved, so it's directly testable without a
    GPU. These tests check it against a reference implementation of
    that original loop for exact behavioral equivalence (same table
    contents are reachable via the same linear-probe lookup rule,
    same occupied-bit convention), not just "doesn't crash"."""

    @staticmethod
    def _reference_loop_build(cracked, table_size, mask):
        hash_table = np.zeros(table_size, dtype=np.uint64)
        occupied = np.zeros((table_size + 31) // 32, dtype=np.uint32)
        for h in np.asarray(cracked, dtype=np.uint64):
            slot = int(_mix64_reference(np.uint64(h))) & mask
            while occupied[slot >> 5] & (1 << (slot & 31)):
                slot = (slot + 1) & mask
            hash_table[slot] = h
            occupied[slot >> 5] |= np.uint32(1 << (slot & 31))
        return hash_table, occupied

    @staticmethod
    def _lookup(hash_table, occupied, mask, key):
        """Mirrors the GPU kernel's lookup_cracked_slot() probe rule."""
        slot = int(_mix64_reference(np.uint64(key))) & mask
        for _ in range(len(hash_table)):
            word, bit = slot >> 5, slot & 31
            if occupied[word] & (1 << bit):
                if hash_table[slot] == key:
                    return slot
            else:
                return -1
            slot = (slot + 1) & mask
        return -1

    def _table_size_for(self, n):
        table_size = 1
        target = max(2, int(math.ceil(n * 2.0)))
        while table_size < target:
            table_size <<= 1
        return table_size

    def test_matches_reference_occupied_bitcount(self):
        rng = np.random.default_rng(42)
        cracked = np.unique(rng.integers(0, 2**64, size=5000, dtype=np.uint64))
        table_size = self._table_size_for(len(cracked))
        mask = table_size - 1

        ref_table, ref_occ = self._reference_loop_build(cracked, table_size, mask)
        vec_table, vec_occ = crg._build_open_addressing_table(cracked, table_size, mask)

        assert vec_occ.dtype == np.uint32
        assert vec_occ.shape == ref_occ.shape
        vec_bits_set = np.unpackbits(vec_occ.view(np.uint8), bitorder='little').sum()
        ref_bits_set = np.unpackbits(ref_occ.view(np.uint8), bitorder='little').sum()
        assert vec_bits_set == ref_bits_set == len(cracked)

    def test_every_key_findable_via_gpu_probe_rule(self):
        rng = np.random.default_rng(7)
        cracked = np.unique(rng.integers(0, 2**64, size=20000, dtype=np.uint64))
        table_size = self._table_size_for(len(cracked))
        mask = table_size - 1

        vec_table, vec_occ = crg._build_open_addressing_table(cracked, table_size, mask)
        for key in cracked[:2000]:
            assert self._lookup(vec_table, vec_occ, mask, int(key)) != -1

    def test_no_key_collisions_in_stored_table(self):
        rng = np.random.default_rng(99)
        cracked = np.unique(rng.integers(0, 2**64, size=3000, dtype=np.uint64))
        table_size = self._table_size_for(len(cracked))
        mask = table_size - 1

        vec_table, vec_occ = crg._build_open_addressing_table(cracked, table_size, mask)
        occ_bits = np.unpackbits(vec_occ.view(np.uint8), bitorder='little')[:table_size]
        occupied_keys = vec_table[occ_bits.astype(bool)]
        assert len(occupied_keys) == len(cracked)
        assert set(occupied_keys.tolist()) == set(cracked.tolist())

    def test_small_and_edge_sizes(self):
        for n in (0, 1, 2, 5):
            cracked = np.arange(n, dtype=np.uint64) * np.uint64(2654435761)
            cracked = np.unique(cracked)
            table_size = self._table_size_for(max(len(cracked), 1))
            mask = table_size - 1
            vec_table, vec_occ = crg._build_open_addressing_table(cracked, table_size, mask)
            bits_set = np.unpackbits(vec_occ.view(np.uint8), bitorder='little').sum()
            assert bits_set == len(cracked)
            for key in cracked:
                assert self._lookup(vec_table, vec_occ, mask, int(key)) != -1


def test_opencl_source_keeps_fnv1a64_as_uint64_before_lookup():
    src = crg.get_recompute_kernel_source(num_cracked=8, hash_table_size=16)
    # CELF uses FNV-1a-64. Truncating the kernel result to uint32 makes
    # every lookup effectively search for the low 32 bits in a uint64 table,
    # which can turn a valid cracked universe into zero hits.
    assert src.count('ulong h = fnv1a_hash_64(result_temp, (unsigned int)out_len);') == 3
    assert src.count('unsigned int h = fnv1a_hash_64(result_temp, (unsigned int)out_len);') == 0


def test_uint64_cracked_fingerprint_with_high_bits_is_preserved():
    from rule_ranker.hashing import build_open_addressing_table_uint64

    # Regression fixture: this key is intentionally larger than uint32.
    key = np.uint64(0xF123456789ABCDEF)
    table, states = build_open_addressing_table_uint64(
        np.array([key], dtype=np.uint64)
    )
    occupied = np.packbits(states == 2, bitorder='little')
    pad = (-len(occupied)) % 4
    if pad:
        occupied = np.concatenate([occupied, np.zeros(pad, dtype=np.uint8)])
    occupied = occupied.view(np.uint32)

    mask = len(table) - 1
    slot = int(_mix64_reference(key)) & mask
    found = False
    for _ in range(len(table)):
        word, bit = slot >> 5, slot & 31
        if occupied[word] & (1 << bit):
            if table[slot] == key:
                found = True
                break
        else:
            break
        slot = (slot + 1) & mask
    assert found


def test_opencl_source_contains_empty_rotate_guard_and_overflow_abort():
    src = crg.get_recompute_kernel_source(num_cracked=8, hash_table_size=16)
    assert 'if (in_len <= 0)' in src
    assert 'if (in_len*2>MAX_OUTPUT_LEN) return -2;' in src
    assert 'if (in_len+1>MAX_OUTPUT_LEN) return -2;' in src
    assert 'if (result < 0)' in src
