"""Tests for run_ranker.py -- the top-level subcommand dispatcher.
Verifies routing/argv-passthrough without actually running a GPU job."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import run_ranker


class TestDispatcher:
    def test_no_args_prints_usage_and_returns_1(self, capsys):
        rc = run_ranker.main([])
        assert rc == 1
        out = capsys.readouterr().out
        assert "rank" in out and "handler" in out and "postprocess" in out

    def test_top_level_help_returns_0(self, capsys):
        rc = run_ranker.main(["--help"])
        assert rc == 0

    def test_unknown_command_returns_1(self, capsys):
        rc = run_ranker.main(["bogus-command"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "unknown command" in err

    def test_handler_dispatch_routes_argv_through(self, tmp_path, capsys):
        # `handler` has no GPU dependency, so this exercises a real
        # end-to-end dispatch: run_ranker.py -> rule_ranker.ranker_handler.main()
        import csv
        csv_path = tmp_path / "ranking.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Rank", "Combined_Score", "Effectiveness_Score", "Uniqueness_Score", "Rule_Data"])
            w.writerow([1, 100, 60, 40, "l"])
        summary_out = tmp_path / "summary.txt"

        rc = run_ranker.main([
            "handler", "-i", str(csv_path), "-o", str(summary_out),
            "--no-console", "--no-progress",
        ])
        assert rc == 0
        assert summary_out.exists()

    def test_handler_help_routes_through_and_exits_zero(self, capsys):
        try:
            run_ranker.main(["handler", "--help"])
        except SystemExit as e:
            assert e.code == 0
        else:
            raise AssertionError("expected SystemExit(0) from 'handler --help'")

    def test_postprocess_help_routes_through_and_exits_zero(self, capsys):
        try:
            run_ranker.main(["postprocess", "--help"])
        except SystemExit as e:
            assert e.code == 0
        else:
            raise AssertionError("expected SystemExit(0) from 'postprocess --help'")

    def test_rank_help_routes_through_and_exits_zero(self, capsys):
        # exercises importing rule_ranker.ranker itself, which needs
        # pyopencl importable -- covered by tests/conftest.py's stub
        # when the real pyopencl isn't installed.
        try:
            run_ranker.main(["rank", "--help"])
        except SystemExit as e:
            assert e.code == 0
        else:
            raise AssertionError("expected SystemExit(0) from 'rank --help'")
