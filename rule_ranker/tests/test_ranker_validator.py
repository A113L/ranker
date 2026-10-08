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


def test_load_rules_skips_overlong_rules(tmp_path):
    rules_path = tmp_path / "rules.rule"
    rules_path.write_text(":" + "\n" + ("l" * 256) + "\n", encoding="latin-1")
    loaded = ranker.load_rules(str(rules_path))
    assert [row["rule_data"] for row in loaded] == [":"]


def test_ranker_uses_64_bit_independent_fingerprint_kernel():
    src = ranker.get_kernel_source(rule_hash_table_bits=8, cracked_hash_table_bits=8)
    assert 'fnv1a_hash_64' in src
    assert '__global ulong* rule_hash_tables' in src
    assert '__global uint* rule_hash_states' in src
    assert 'table_base = rule_idx * (rule_hash_table_mask + 1U)' in src
    assert 'insert_rule_fingerprint' in src
    assert 'lookup_cracked_fingerprint' in src
    assert 'GLOBAL_HASH_MAP_MASK' not in src


def test_sample_reader_is_bounded_and_non_empty(tmp_path):
    path = tmp_path / 'words.txt'
    path.write_bytes(b'\n'.join(f'word{i}'.encode() for i in range(1000)) + b'\n')
    arr, count = ranker._read_stratified_word_sample(str(path), max_len=32, sample_words=64, seed=123)
    assert count > 0
    assert count <= 64
    assert arr.shape == (count, 32)
    assert arr.dtype == __import__('numpy').uint8


def test_mab_top_rules_handles_request_equal_to_population():
    rules = [{'rule_id': i, 'rule_data': ':'} for i in range(2)]
    bandit = ranker.MultiPassMAB(rules, final_trials=1, screening_trials=1)
    bandit.trials[:] = 1
    bandit.successes[:] = [4.0, 2.0]
    bandit.failures[:] = [2.0, 4.0]
    assert [r['rule_id'] for r in bandit.get_top_rules(2)] == [0, 1]


def test_mab_eta_does_not_underflow_uint32_trials():
    import numpy as np
    rules = [{'rule_id': i, 'rule_data': ':'} for i in range(3)]
    bandit = ranker.MultiPassMAB(rules, final_trials=50, screening_trials=5)
    # One rule is beyond the budget, one exactly at the budget, one needs one trial.
    bandit.trials[:] = np.array([60, 50, 49], dtype=np.uint32)
    assert ranker._estimate_mab_remaining_iterations(bandit) == 1


def test_mab_does_not_resample_rules_past_final_trials():
    import numpy as np
    rules = [{'rule_id': i, 'rule_data': ':'} for i in range(3)]
    bandit = ranker.MultiPassMAB(rules, final_trials=2, screening_trials=1)
    bandit.trials[:] = np.array([2, 1, 2], dtype=np.uint32)
    bandit.successes[:] = 2.0
    bandit.failures[:] = 2.0
    selected = bandit.select_rules(batch_size=3, iteration=0)
    assert selected == [1]
