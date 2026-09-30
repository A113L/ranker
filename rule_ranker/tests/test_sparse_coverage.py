"""Tests for rule_ranker.sparse_coverage -- the sparse-coverage-store
+ pure-CPU CELF strategy (--strategy sparse). GPU kernel source
generation / OpenCL device code paths are out of scope here, same as
the other strategy test files; see conftest.py for why `import
pyopencl` still needs to succeed.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rule_ranker import sparse_coverage as sc


class TestInMemorySparseCoverageStore:
    def test_put_many_and_getitem(self):
        store = sc.InMemorySparseCoverageStore()
        store.put_many([
            (0, np.array([1, 2, 3], dtype=np.int32)),
            (1, sc.EMPTY_HITS),
        ])
        assert list(store[0]) == [1, 2, 3]
        assert len(store[1]) == 0

    def test_iter_candidates_with_hits_skips_empty(self):
        store = sc.InMemorySparseCoverageStore()
        store.put_many([
            (0, np.array([1], dtype=np.int32)),
            (1, sc.EMPTY_HITS),
            (2, np.array([5, 6], dtype=np.int32)),
        ])
        got = dict(store.iter_candidates_with_hits())
        assert got == {0: 1, 2: 2}
        assert store.count_with_hits() == 2


class TestSQLiteSparseCoverageStore:
    def test_put_many_and_getitem_roundtrip(self, tmp_path):
        db_path = str(tmp_path / "cov.db")
        store = sc.SQLiteSparseCoverageStore(path=db_path)
        try:
            store.put_many([
                (0, np.array([10, 20, 30], dtype=np.int32)),
                (1, sc.EMPTY_HITS),
                (2, np.array([7], dtype=np.int32)),
            ])
            assert list(store[0]) == [10, 20, 30]
            assert len(store[1]) == 0
            assert list(store[2]) == [7]
            assert len(store) == 3
            assert store.count_with_hits() == 2
            got = dict(store.iter_candidates_with_hits())
            assert got == {0: 3, 2: 1}
        finally:
            store.close()

    def test_missing_key_raises(self, tmp_path):
        store = sc.SQLiteSparseCoverageStore(path=str(tmp_path / "cov2.db"))
        try:
            with pytest.raises(KeyError):
                _ = store[999]
        finally:
            store.close()

    def test_temp_file_cleaned_up_on_close(self):
        store = sc.SQLiteSparseCoverageStore()
        path = store._path
        store.put_many([(0, np.array([1], dtype=np.int32))])
        assert os.path.exists(path)
        store.close()
        assert not os.path.exists(path)

    def test_getitem_cache_returns_consistent_values(self, tmp_path):
        # Exercises the LRU cache path (repeated lookups of the same
        # hot rule, as CELF's lazy revalidation does).
        store = sc.SQLiteSparseCoverageStore(path=str(tmp_path / "cov3.db"))
        try:
            store.put_many([(0, np.array([1, 2], dtype=np.int32))])
            for _ in range(5):
                assert list(store[0]) == [1, 2]
        finally:
            store.close()


class TestCelfSelectSparse:
    def test_picks_best_coverage_first_and_saturates(self):
        rules = ["a", "b", "c"]
        store = sc.InMemorySparseCoverageStore()
        store.put_many([
            (0, np.array([0, 1, 2], dtype=np.int32)),   # a: covers 3
            (1, np.array([2, 3], dtype=np.int32)),       # b: covers 2, overlaps a
            (2, np.array([4], dtype=np.int32)),          # c: covers 1, disjoint
        ])
        selected = sc.celf_select_sparse(rules, store, cracked_size=5)
        picked = [r for r, _g in selected]
        assert picked[0] == "a"
        assert selected[0][1] == 3
        # After "a", "b" only has {3} left (gain 1), same as "c" (gain 1);
        # tie broken by lower original index -> "b" (idx 1) beats "c" (idx 2).
        assert picked[1] == "b"
        assert selected[1][1] == 1
        assert picked[2] == "c"
        assert selected[2][1] == 1

    def test_budget_cuts_off_selection(self):
        rules = ["a", "b", "c"]
        store = sc.InMemorySparseCoverageStore()
        store.put_many([
            (0, np.array([0], dtype=np.int32)),
            (1, np.array([1], dtype=np.int32)),
            (2, np.array([2], dtype=np.int32)),
        ])
        selected = sc.celf_select_sparse(rules, store, cracked_size=3, budget=2)
        assert len(selected) == 2

    def test_zero_coverage_rules_never_selected(self):
        rules = ["a", "b"]
        store = sc.InMemorySparseCoverageStore()
        store.put_many([
            (0, np.array([0, 1], dtype=np.int32)),
            (1, sc.EMPTY_HITS),
        ])
        selected = sc.celf_select_sparse(rules, store, cracked_size=2)
        assert [r for r, _g in selected] == ["a"]

    def test_works_against_a_plain_dict_not_just_a_store_type(self):
        # celf_select_sparse() must also accept a plain dict (no
        # iter_candidates_with_hits/count_with_hits), exactly like
        # celf_select_recompute_gpu()'s equivalent fallback.
        rules = ["a", "b"]
        plain = {
            0: np.array([0, 1], dtype=np.int32),
            1: np.array([1], dtype=np.int32),
        }
        selected = sc.celf_select_sparse(rules, plain, cracked_size=2)
        assert [r for r, _g in selected] == ["a"]

    def test_no_candidates_with_hits_returns_empty(self):
        rules = ["a", "b"]
        store = sc.InMemorySparseCoverageStore()
        store.put_many([(0, sc.EMPTY_HITS), (1, sc.EMPTY_HITS)])
        selected = sc.celf_select_sparse(rules, store, cracked_size=4)
        assert selected == []

    def test_sqlite_backed_store_gives_same_result_as_in_memory(self, tmp_path):
        rules = ["a", "b", "c"]
        rows = [
            (0, np.array([0, 1, 2], dtype=np.int32)),
            (1, np.array([2, 3], dtype=np.int32)),
            (2, np.array([4], dtype=np.int32)),
        ]
        mem_store = sc.InMemorySparseCoverageStore()
        mem_store.put_many(rows)
        mem_selected = sc.celf_select_sparse(rules, mem_store, cracked_size=5)

        db_store = sc.SQLiteSparseCoverageStore(path=str(tmp_path / "cov4.db"))
        try:
            db_store.put_many(rows)
            db_selected = sc.celf_select_sparse(rules, db_store, cracked_size=5)
        finally:
            db_store.close()

        assert mem_selected == db_selected
