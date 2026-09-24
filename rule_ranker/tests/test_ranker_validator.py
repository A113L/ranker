"""Tests for rule_ranker.ranker -- the pure-Python rule validator and
CLI argument parsing. Does not touch the GPU/OpenCL code paths (see
conftest.py for why pyopencl needs to be importable regardless)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rule_ranker import ranker


# --------------------------------------------------------------
# should_exclude_rule / HashcatRuleValidator -- covers the bug fixes
# called out in the module's changelog (v5.1/v5.2)
# --------------------------------------------------------------
class TestShouldExcludeRule:
    def test_bare_bang_excluded(self):
        # BUG FIX #3: bare '!' must be excluded even without an argument
        assert ranker.should_exclude_rule("!") is True

    def test_single_char_reject_ops_excluded(self):
        for op in ("_", "M", "4", "6", "Q", "!"):
            assert ranker.should_exclude_rule(op) is True

    def test_ordinary_single_char_rule_not_excluded(self):
        for op in ("l", "u", "c", "t", "r", "d"):
            assert ranker.should_exclude_rule(op) is False

    def test_two_char_reject_ops_excluded(self):
        for op in ("!a", "/b", "(c", ")d", "<e", ">f", "_g"):
            assert ranker.should_exclude_rule(op) is True

    def test_three_char_reject_ops_excluded(self):
        for op in ("?0a", "=0a", "v0a"):
            assert ranker.should_exclude_rule(op) is True

    def test_empty_rule_not_excluded(self):
        assert ranker.should_exclude_rule("") is False


class TestHashcatRuleValidatorPositions:
    def test_is_pos_accepts_digits_and_uppercase(self):
        # BUG FIX: positions >9 encoded as A-Z (A=10..Z=35)
        for c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            assert ranker.HashcatRuleValidator.is_pos(c) is True

    def test_is_pos_rejects_lowercase_and_symbols(self):
        for c in "a-z!@#":
            if c in "abcdefghijklmnopqrstuvwxyz":
                assert ranker.HashcatRuleValidator.is_pos(c) is False

    def test_high_position_rule_T_accepted(self):
        # 'TA' = toggle case at position 10 -- previously rejected by a
        # digit-only position check
        assert ranker.HashcatRuleValidator.validate_rule_for_gpu("TA") is True

    def test_p_with_high_arg_accepted(self):
        # BUG FIX #8: 'pV' (duplicate word 31 times) -- bare 'p' must
        # consume its mandatory position-encoded argument
        assert ranker.HashcatRuleValidator.validate_rule_for_gpu("pV") is True

    def test_bare_p_without_arg_rejected(self):
        assert ranker.HashcatRuleValidator.validate_rule_for_gpu("p") is False

    def test_zN_with_high_arg_accepted(self):
        assert ranker.HashcatRuleValidator.validate_rule_for_gpu("zA") is True

    def test_simple_valid_rules_accepted(self):
        for rule in (":", "l", "u", "c", "r", "$1", "^a", "so0"):
            assert ranker.HashcatRuleValidator.validate_rule_for_gpu(rule) is True

    def test_banned_operator_rules_rejected(self):
        for rule in ("M", "4", "6", "Q", "!"):
            assert ranker.HashcatRuleValidator.validate_rule_for_gpu(rule) is False


# --------------------------------------------------------------
# CLI wiring -- argparse setup, --list-devices short-circuit
# --------------------------------------------------------------
class TestArgParser:
    def test_required_args_enforced(self):
        parser = ranker.build_arg_parser()
        with __import__("pytest").raises(SystemExit):
            parser.parse_args([])

    def test_minimal_valid_args_parse(self):
        parser = ranker.build_arg_parser()
        args = parser.parse_args([
            "-w", "words.txt", "-r", "rules.rule", "-c", "cracked.txt",
        ])
        assert args.wordlist == "words.txt"
        assert args.rules == "rules.rule"
        assert args.cracked == "cracked.txt"
        assert args.output == "ranker_output.csv"  # default
        assert args.legacy is False

    def test_legacy_flag(self):
        parser = ranker.build_arg_parser()
        args = parser.parse_args([
            "-w", "w.txt", "-r", "r.rule", "-c", "c.txt", "--legacy",
        ])
        assert args.legacy is True

    def test_list_devices_exits_zero(self, monkeypatch):
        # -w/-r/-c are `required=True` on the original parser, so they
        # must still be supplied even for --list-devices (unchanged
        # behavior from the original script: parse_args() runs before
        # the list_devices check). main() should then call
        # list_platforms_and_devices() and sys.exit(0) without going
        # on to run any ranking.
        monkeypatch.setattr(ranker, "list_platforms_and_devices", lambda: None)
        try:
            ranker.main([
                "-w", "w.txt", "-r", "r.rule", "-c", "c.txt", "--list-devices",
            ])
        except SystemExit as e:
            assert e.code == 0
        else:
            raise AssertionError("expected SystemExit(0) from --list-devices")
