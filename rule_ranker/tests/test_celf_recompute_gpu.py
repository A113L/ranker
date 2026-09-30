"""Tests for rule_ranker.celf_recompute_gpu's greedy ROUND LOGIC
(fixed sorted-order scan, upper_bound/last_bound pruning, tie-breaks,
budget cutoff) in isolation from real OpenCL/GPU kernels.

celf_select_recompute_gpu() constructs a real _GpuScorer internally,
which needs an actual GPU. To test the selection algorithm itself
(the part that was rewritten to fix the "terribly slow" over-fetching
bug -- see the block comment above celf_select_recompute_gpu() in
celf_recompute_gpu.py), this monkeypatches _GpuScorer with a small
pure-Python/NumPy fake backed by an explicit rule -> covered-hash-ids
mapping, so exact coverage/selection outcomes can be asserted without
any device present. See conftest.py for why `import pyopencl` still
needs to succeed even though no real OpenCL is exercised here.
"""
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

    def apply_winner_and_clear(self, rule_str, wordlist_path):
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
