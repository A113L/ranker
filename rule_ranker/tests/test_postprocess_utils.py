"""Tests for rule_ranker.ranker_postprocess -- popcount helpers, budget
parsing, hash function, and the CPU CELF greedy-select algorithm.
GPU kernel source generation / OpenCL device code paths are out of
scope here (see conftest.py for why pyopencl still needs to import)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rule_ranker import ranker_postprocess as rp


class TestPopcount:
    def test_popcount_row_zero(self):
        row = np.zeros(4, dtype=np.uint32)
        assert rp.popcount_row(row) == 0

    def test_popcount_row_all_ones(self):
        row = np.full(2, 0xFFFFFFFF, dtype=np.uint32)
        assert rp.popcount_row(row) == 64

    def test_popcount_row_known_pattern(self):
        # 0b1011 = 3 bits set
        row = np.array([0b1011], dtype=np.uint32)
        assert rp.popcount_row(row) == 3

    def test_popcount_rows_matches_popcount_row(self):
        mat = np.array([
            [0b1011, 0],
            [0xFFFFFFFF, 0xFFFFFFFF],
            [0, 0],
        ], dtype=np.uint32)
        rows = rp.popcount_rows(mat)
        expected = [rp.popcount_row(mat[i]) for i in range(mat.shape[0])]
        assert list(rows) == expected


class TestFnv1a:
    def test_deterministic(self):
        h1 = rp.fast_fnv1a_hash_32(b"password123")
        h2 = rp.fast_fnv1a_hash_32(b"password123")
        assert h1 == h2

    def test_different_inputs_differ(self):
        h1 = rp.fast_fnv1a_hash_32(b"password123")
        h2 = rp.fast_fnv1a_hash_32(b"password124")
        assert h1 != h2

    def test_within_32_bits(self):
        h = rp.fast_fnv1a_hash_32(b"x" * 100)
        assert 0 <= h <= 0xFFFFFFFF


class TestParseBudgets:
    def test_basic(self):
        assert rp.parse_budgets("64,250,5000") == [64, 250, 5000]

    def test_dedupes_and_sorts(self):
        assert rp.parse_budgets("500,10,500,10,20") == [10, 20, 500]

    def test_handles_whitespace(self):
        assert rp.parse_budgets(" 64 , 250 ,5000 ") == [64, 250, 5000]

    def test_empty_string_returns_empty(self):
        assert rp.parse_budgets("") == []

    def test_none_returns_empty(self):
        assert rp.parse_budgets(None) == []

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            rp.parse_budgets("64,-1,5000")

    def test_rejects_zero(self):
        with pytest.raises(ValueError):
            rp.parse_budgets("0,5")


class TestLoadCrackedUniverse:
    def test_dedupes_and_sorts_hashes(self, tmp_path):
        cracked_path = tmp_path / "cracked.txt"
        cracked_path.write_text("password1\npassword2\npassword1\n", encoding="utf-8")
        arr = rp.load_cracked_universe(str(cracked_path), max_len=256)
        assert len(arr) == 2  # duplicate collapsed
        assert list(arr) == sorted(arr.tolist())  # sorted ascending
        assert arr.dtype == np.uint32

    def test_respects_max_len(self, tmp_path):
        cracked_path = tmp_path / "cracked.txt"
        cracked_path.write_text("short\n" + ("x" * 300) + "\n", encoding="utf-8")
        arr = rp.load_cracked_universe(str(cracked_path), max_len=256)
        assert len(arr) == 1  # the 300-char line is dropped


class TestCelfSelect:
    """Exercises the CPU lazy-greedy CELF loop directly against small,
    hand-built coverage bitmaps -- no GPU/OpenCL involved."""

    def _bitmap_from_bools(self, rows_of_bools):
        """rows_of_bools: list[list[bool]] -> (n, W) uint32 array."""
        n = len(rows_of_bools)
        n_bits = len(rows_of_bools[0])
        W = max(1, (n_bits + 31) // 32)
        mat = np.zeros((n, W), dtype=np.uint32)
        for i, bits in enumerate(rows_of_bools):
            for b, val in enumerate(bits):
                if val:
                    mat[i, b >> 5] |= np.uint32(1 << (b & 31))
        return mat

    def test_picks_best_covering_rule_first(self):
        # rule 0 covers bits {0,1,2}; rule 1 covers only {0}
        rules = ["rule_a", "rule_b"]
        bitmap = self._bitmap_from_bools([
            [1, 1, 1, 0],
            [1, 0, 0, 0],
        ])
        selected = rp.celf_select(rules, bitmap, budget=1)
        assert len(selected) == 1
        assert selected[0][0] == "rule_a"
        assert selected[0][1] == 3  # gain = popcount of its row

    def test_greedy_covers_universe_with_disjoint_rules(self):
        # rule 0 covers {0,1}; rule 1 covers {2,3} -- together full coverage
        rules = ["rule_a", "rule_b"]
        bitmap = self._bitmap_from_bools([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ])
        selected = rp.celf_select(rules, bitmap, budget=None)
        assert {r for r, _ in selected} == {"rule_a", "rule_b"}
        total_gain = sum(g for _, g in selected)
        assert total_gain == 4

    def test_lazy_revalidation_picks_second_best_after_overlap(self):
        # rule_a covers {0,1,2}; rule_b covers {2,3}; after picking
        # rule_a (gain 3), rule_b's TRUE marginal gain is only 1 (bit 3),
        # not its stale initial gain of 2 -- CELF's lazy re-validation
        # must catch this rather than trusting the stale heap value.
        rules = ["rule_a", "rule_b"]
        bitmap = self._bitmap_from_bools([
            [1, 1, 1, 0],
            [0, 0, 1, 1],
        ])
        selected = rp.celf_select(rules, bitmap, budget=None)
        assert selected[0][0] == "rule_a"
        assert selected[0][1] == 3
        assert selected[1][0] == "rule_b"
        assert selected[1][1] == 1  # re-validated marginal gain, not stale 2

    def test_zero_gain_rules_excluded(self):
        rules = ["rule_a", "rule_none"]
        bitmap = self._bitmap_from_bools([
            [1, 1, 0, 0],
            [0, 0, 0, 0],
        ])
        selected = rp.celf_select(rules, bitmap, budget=None)
        assert len(selected) == 1
        assert selected[0][0] == "rule_a"

    def test_budget_caps_selection_size(self):
        rules = ["a", "b", "c"]
        bitmap = self._bitmap_from_bools([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        selected = rp.celf_select(rules, bitmap, budget=2)
        assert len(selected) == 2

    def test_accepts_precomputed_initial_gains(self):
        rules = ["a", "b"]
        bitmap = self._bitmap_from_bools([
            [1, 1, 0],
            [0, 0, 1],
        ])
        gains = rp.popcount_rows(bitmap)
        selected = rp.celf_select(rules, bitmap, initial_gains=gains, budget=None)
        assert len(selected) == 2

    def test_works_on_memmap_backed_array(self, tmp_path):
        rules = ["a", "b"]
        bitmap = self._bitmap_from_bools([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ])
        mm_path = tmp_path / "cov.dat"
        mm = np.memmap(str(mm_path), dtype=np.uint32, mode="w+", shape=bitmap.shape)
        mm[:] = bitmap
        mm.flush()
        selected = rp.celf_select(rules, mm, budget=None)
        assert {r for r, _ in selected} == {"a", "b"}


class TestSaveOutput:
    def test_writes_rule_and_csv_files(self, tmp_path):
        out_path = tmp_path / "selected.rule"
        selected = [("l", 100), ("u", 80), ("[c]", 10)]
        rp.save_output(selected, str(out_path))

        assert out_path.exists()
        rule_lines = out_path.read_text(encoding="utf-8").splitlines()
        assert rule_lines[0] == ":"
        assert "l" in rule_lines
        assert "u" in rule_lines

        csv_path = tmp_path / "selected_celf.csv"
        assert csv_path.exists()
        csv_text = csv_path.read_text(encoding="utf-8")
        assert "Rank" in csv_text and "Marginal_Gain" in csv_text


class TestSaveOutputMulti:
    def test_creates_one_file_pair_per_budget(self, tmp_path):
        out_path = tmp_path / "selected.rule"
        selected = [("a", 10), ("b", 8), ("c", 5), ("d", 1)]
        rp.save_output_multi(selected, str(out_path), [2, 4])

        top2 = tmp_path / "selected_top2.rule"
        top4 = tmp_path / "selected_top4.rule"
        assert top2.exists()
        assert top4.exists()
        top2_rules = [
            ln for ln in top2.read_text(encoding="utf-8").splitlines() if ln != ":"
        ]
        assert top2_rules == ["a", "b"]

    def test_budget_exceeding_selection_writes_all(self, tmp_path, capsys):
        out_path = tmp_path / "selected.rule"
        selected = [("a", 10), ("b", 5)]
        rp.save_output_multi(selected, str(out_path), [100])
        top100 = tmp_path / "selected_top100.rule"
        assert top100.exists()
        rules = [ln for ln in top100.read_text(encoding="utf-8").splitlines() if ln != ":"]
        assert rules == ["a", "b"]


class TestPostprocessCLI:
    def test_requires_ranking_or_rules_source(self):
        try:
            rp.main(["-w", "w.txt", "-k", "c.txt", "-o", "out.rule"])
        except SystemExit as e:
            assert e.code != 0
        else:
            raise AssertionError("expected SystemExit for missing -r/-f")

    def test_help_exits_zero(self):
        try:
            rp.main(["--help"])
        except SystemExit as e:
            assert e.code == 0
        else:
            raise AssertionError("expected SystemExit(0) from --help")

    def test_aborts_on_empty_cracked_list(self, tmp_path, capsys):
        rules_path = tmp_path / "rules.rule"
        rules_path.write_text(":\nl\nu\n", encoding="utf-8")
        wordlist_path = tmp_path / "words.txt"
        wordlist_path.write_text("password\n", encoding="utf-8")
        cracked_path = tmp_path / "cracked.txt"
        # Not a truly empty file (load_cracked_universe mmaps it, which
        # requires a non-zero size) -- a single line longer than max_len
        # gets filtered out by load_cracked_universe, which is the
        # actual way to reach "zero valid cracked hashes" without also
        # hitting the unrelated empty-file mmap edge case.
        cracked_path.write_text("x" * 300 + "\n", encoding="utf-8")
        out_path = tmp_path / "out.rule"

        with pytest.raises(SystemExit) as excinfo:
            rp.main([
                "-f", str(rules_path),
                "-w", str(wordlist_path),
                "-k", str(cracked_path),
                "-o", str(out_path),
            ])
        assert excinfo.value.code == 1
