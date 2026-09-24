"""Tests for rule_ranker.ranker_handler -- rule cleaning, CSV parsing,
and analysis helpers. No GPU dependency in this module at all."""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rule_ranker import ranker_handler as rh


class TestCleanRuleCached:
    def test_empty_string(self):
        assert rh.clean_rule_cached("") == ""

    def test_bracketed_single_rule(self):
        assert rh.clean_rule_cached("[l]") == "l"

    def test_plain_rule_unchanged(self):
        assert rh.clean_rule_cached("l") == "l"

    def test_multi_rule_sequence_with_space_separated_brackets(self):
        # NOTE: clean_rule_cached only strips the OUTERMOST bracket pair
        # when the whole string starts with '[' and ends with ']', and
        # bails out of that fast path as soon as the (now unwrapped)
        # remainder still contains a space -- it does not re-strip
        # brackets around each individual space-separated part in that
        # case. This pins down that actual (slightly surprising)
        # behavior rather than an idealized one.
        assert rh.clean_rule_cached("[l] [u]") == "l] [u"

    def test_mixed_bracketed_and_plain_sequence(self):
        # Doesn't start with '[' (starts with 'x'), so it takes the
        # space-split path directly, where each space-separated part
        # *does* get its own brackets stripped.
        assert rh.clean_rule_cached("x [l] u") == "x l u"

    def test_whitespace_trimmed(self):
        assert rh.clean_rule_cached("  l  ") == "l"


class TestFmtNum:
    def test_thousands_separator(self):
        assert rh.fmt_num(1234567) == "1,234,567"

    def test_small_number(self):
        assert rh.fmt_num(42) == "42"


def _write_legacy_csv(path, rows):
    """rows: list of (rank, combined, effectiveness, uniqueness, rule)"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Rank", "Combined_Score", "Effectiveness_Score", "Uniqueness_Score", "Rule_Data"])
        for row in rows:
            w.writerow(row)


class TestParseRankingFileFast:
    def test_parses_legacy_csv(self, tmp_path):
        csv_path = tmp_path / "ranking.csv"
        _write_legacy_csv(csv_path, [
            (1, 500, 300, 200, "l"),
            (2, 400, 250, 150, "u"),
            (3, 100, 50, 50, "[c]"),
        ])
        data = rh.parse_ranking_file_fast(str(csv_path), show_progress=False)
        assert len(data) == 3
        assert data[0]["rank"] == 1
        assert data[0]["combined_score"] == 500
        assert data[0]["rule_data"] == "l"
        assert data[2]["rule_data"] == "[c]"  # cleaning happens later, in analyze

    def test_skips_malformed_rows(self, tmp_path):
        csv_path = tmp_path / "ranking.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Rank", "Combined_Score", "Effectiveness_Score", "Uniqueness_Score", "Rule_Data"])
            w.writerow(["not_a_number", "1", "1", "1", "l"])  # malformed rank -> header re-detect edge case
            w.writerow([1, 10, 5, 5, "u"])
        data = rh.parse_ranking_file_fast(str(csv_path), show_progress=False)
        # at least the well-formed row must survive
        assert any(d["rule_data"] == "u" for d in data)

    def test_missing_file_returns_empty_list(self):
        data = rh.parse_ranking_file_fast("/nonexistent/path/does_not_exist.csv", show_progress=False)
        assert data == []


class TestAnalyzeRulesFast:
    def test_aggregates_across_files_and_sorts_by_combined_score(self, tmp_path):
        csv1 = tmp_path / "a.csv"
        csv2 = tmp_path / "b.csv"
        _write_legacy_csv(csv1, [(1, 100, 60, 40, "l")])
        _write_legacy_csv(csv2, [(1, 900, 500, 400, "u")])

        data1 = rh.parse_ranking_file_fast(str(csv1), show_progress=False)
        data2 = rh.parse_ranking_file_fast(str(csv2), show_progress=False)

        results = rh.analyze_rules_fast([data1, data2], top_n=None, show_progress=False)
        assert results["total_rules"] == 2
        # top rule (by combined_score) should be 'u' (900 > 100)
        top_rules = results["top_rules"]
        assert top_rules[0]["cleaned_rule"] == "u"
        assert top_rules[0]["combined_score"] == 900


class TestHandlerCLI:
    def test_help_exits_zero(self, capsys):
        try:
            rh.main(["--help"])
        except SystemExit as e:
            assert e.code == 0
        else:
            raise AssertionError("expected SystemExit(0) from --help")

    def test_missing_required_input_exits_nonzero(self):
        try:
            rh.main([])
        except SystemExit as e:
            assert e.code != 0
        else:
            raise AssertionError("expected SystemExit for missing -i/--input")

    def test_end_to_end_summary_and_rules_output(self, tmp_path):
        csv_path = tmp_path / "ranking.csv"
        _write_legacy_csv(csv_path, [
            (1, 900, 500, 400, "u"),
            (2, 100, 60, 40, "l"),
        ])
        summary_out = tmp_path / "summary.txt"
        rules_out = tmp_path / "clean.rule"

        rh.main([
            "-i", str(csv_path),
            "-o", str(summary_out),
            "-r", str(rules_out),
            "--no-console",
            "--no-progress",
        ])

        assert summary_out.exists()
        assert rules_out.exists()
        rules_text = rules_out.read_text(encoding="utf-8")
        assert "u" in rules_text.splitlines()
