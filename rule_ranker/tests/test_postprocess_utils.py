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
        arr, n_skipped = rp.load_cracked_universe(str(cracked_path), max_len=256)
        assert len(arr) == 2  # duplicate collapsed
        assert list(arr) == sorted(arr.tolist())  # sorted ascending
        assert arr.dtype == np.uint32
        assert n_skipped == 0

    def test_respects_max_len(self, tmp_path):
        cracked_path = tmp_path / "cracked.txt"
        cracked_path.write_text("short\n" + ("x" * 300) + "\n", encoding="utf-8")
        arr, n_skipped = rp.load_cracked_universe(str(cracked_path), max_len=256)
        assert len(arr) == 1  # the 300-char line is dropped
        assert n_skipped == 1  # ...and counted as skipped, not silently lost


class TestEstimateOutputLen:
    """CPU-only static length estimator used by --auto-max-output-len /
    --print-output-len-estimate, mirroring the GPU kernel's length
    transformations (not byte content)."""

    @pytest.mark.parametrize("rule,in_len,expected", [
        (":", 10, 10),
        ("l", 10, 10),
        ("d", 10, 20),
        ("f", 10, 20),
        ("q", 10, 20),
        ("dd", 10, 40),
        ("p2", 10, 30),   # duplicate_word: new_len = in_len * (n+1)
        ("z5", 10, 15),
        ("Z3", 10, 13),
        ("y2", 10, 12),
        ("Y3", 10, 13),
        ("^a", 10, 11),
        ("$a$b", 10, 12),
        ("i0a", 10, 11),
        ("D0", 10, 9),
        ("[", 10, 9),
        ("[", 1, 1),      # guarded: in_len>1 required, no-op at len 1
        ("x02", 10, 2),
        ("O02", 10, 8),
    ])
    def test_matches_expected_length(self, rule, in_len, expected):
        assert rp.estimate_output_len(rule, in_len) == expected

    def test_worst_case_picks_largest_and_identifies_rule(self):
        rules = [":", "l", "d", "p3"]
        worst_len, worst_rule = rp.estimate_worst_case_output_len(rules, max_word_len=8)
        assert worst_len == 8 * 4  # p3 -> in_len * (3+1)
        assert worst_rule == "p3"

    def test_worst_case_falls_back_to_input_len_when_nothing_grows(self):
        rules = [":", "l", "u", "D0"]
        worst_len, worst_rule = rp.estimate_worst_case_output_len(rules, max_word_len=8)
        assert worst_len == 8
        assert worst_rule is None


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

    def test_honors_non_rule_extension_exactly(self, tmp_path):
        # Regression test: save_output() used to silently rewrite any
        # extension other than '.rule' to '.rule', so "-o rules.txt"
        # was actually written to "rules.rule" and the path the user
        # asked for never existed. It must now write to exactly the
        # path given.
        out_path = tmp_path / "selected.txt"
        selected = [("l", 100), ("u", 80)]
        rp.save_output(selected, str(out_path))

        assert out_path.exists()
        wrong_path = tmp_path / "selected.rule"
        assert not wrong_path.exists()
        csv_path = tmp_path / "selected_celf.csv"
        assert csv_path.exists()

    def test_appends_rule_extension_only_when_none_given(self, tmp_path):
        out_path = tmp_path / "selected"
        selected = [("l", 100)]
        rp.save_output(selected, str(out_path))
        assert (tmp_path / "selected.rule").exists()


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

    def test_also_writes_exact_output_path(self, tmp_path):
        # Regression test: passing --budgets used to mean the literal
        # -o/--output path was never written, only the _topN files --
        # the other half of the "-o doesn't save to the given path"
        # bug. The exact path must always end up on disk too.
        out_path = tmp_path / "selected.rule"
        selected = [("a", 10), ("b", 8), ("c", 5), ("d", 1)]
        rp.save_output_multi(selected, str(out_path), [2, 4])

        assert out_path.exists()
        all_rules = [
            ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln != ":"
        ]
        assert all_rules == ["a", "b", "c", "d"]


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

    def test_strategy_accepts_only_recompute_gpu(self, tmp_path):
        rules_path = tmp_path / "rules.rule"
        rules_path.write_text(":\nl\n", encoding="utf-8")
        wordlist_path = tmp_path / "words.txt"
        wordlist_path.write_text("password\n", encoding="utf-8")
        cracked_path = tmp_path / "cracked.txt"
        cracked_path.write_text("x" * 300 + "\n", encoding="utf-8")
        out_path = tmp_path / "out.rule"

        with pytest.raises(SystemExit) as excinfo:
            rp.main([
                "-f", str(rules_path), "-w", str(wordlist_path), "-k", str(cracked_path),
                "-o", str(out_path), "--strategy", "recompute-gpu",
            ])
        assert excinfo.value.code == 1  # parsed successfully, then empty universe aborts before GPU import

    def test_removed_strategy_choices_are_rejected(self):
        for removed in ("bitmap", "sparse"):
            with pytest.raises(SystemExit) as excinfo:
                rp.main(["-w", "w.txt", "-k", "c.txt", "-o", "out.rule",
                         "--strategy", removed])
            assert excinfo.value.code == 2

    def test_strategy_rejects_unknown_choice(self):
        with pytest.raises(SystemExit) as excinfo:
            rp.main(["-w", "w.txt", "-k", "c.txt", "-o", "out.rule",
                     "--strategy", "not-a-real-strategy"])
        assert excinfo.value.code == 2  # argparse's own choices= rejection

    @pytest.mark.parametrize("argv", [
        ["--bitmap-path", "x"],
        ["--keep-bitmap"],
        ["--no-hybrid"],
        ["--in-ram"],
        ["--no-parallel-celf"],
        ["--celf-workers", "2"],
        ["--celf-io-threads", "2"],
        ["--celf-batch-multiplier", "2"],
        ["--sparse-disk-threshold", "2"],
        ["--sparse-store-path", "x"],
        ["--sparse-combined-budget-mb", "1"],
        ["--gpu-celf"],
        ["--gpu-celf-batch", "2"],
        ["--gpu-celf-hit-budget", "10"],
        ["--gpu-celf-vram-fraction", "0.7"],
        ["--gpu-celf-mode", "auto"],
    ])
    def test_removed_coverage_flags_are_rejected(self, argv):
        with pytest.raises(SystemExit) as excinfo:
            rp.main(["-w", "w.txt", "-k", "c.txt", "-o", "out.rule", *argv])
        assert excinfo.value.code == 2

    def test_recompute_gpu_bare_flag_is_not_a_separate_option(self):
        # The only strategy selector is --strategy recompute-gpu; there is
        # intentionally no separate --recompute-gpu flag.
        with pytest.raises(SystemExit) as excinfo:
            rp.main(["-w", "w.txt", "-k", "c.txt", "-o", "out.rule",
                     "--recompute-gpu"])
        assert excinfo.value.code == 2


def test_postprocess_rule_length_matches_ranker():
    from rule_ranker import ranker
    assert rp.MAX_RULE_LEN == ranker.MAX_RULE_LEN == 255

