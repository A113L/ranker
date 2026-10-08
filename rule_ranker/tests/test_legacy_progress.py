import numpy as np

import rule_ranker.ranker as ranker


class _FakePbar:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.total = kwargs.get("total")
        self.n = 0
        self.postfixes = []
        self.refresh_count = 0
        self.closed = False
        type(self).instances.append(self)

    def set_postfix_str(self, text, refresh=False):
        self.postfixes.append((text, refresh))

    def update(self, value, refresh=True):
        self.n += value

    def refresh(self, force=False):
        self.refresh_count += 1

    def set_total(self, value, refresh=True):
        self.total = value

    def close(self):
        self.closed = True


class _FakeScorer:
    def __init__(self, *args, **kwargs):
        self.max_rules = 2
        self.rule_table_size = 16
        self.device = type("D", (), {"name": "fake-gpu"})()
        self.platform = type("P", (), {"name": "fake-platform"})()

    def score(self, words_np, rules_encoded, progress_callback=None):
        assert progress_callback is not None
        # Simulate two completed GPU dispatches. This mirrors the callback
        # contract used by the real scorer when the rule set spans chunks.
        progress_callback(0, 2, 3)
        progress_callback(2, 3, 3)
        return (
            np.ones(len(rules_encoded), dtype=np.uint64),
            np.ones(len(rules_encoded), dtype=np.uint64),
        )


def test_legacy_progress_updates_during_gpu_dispatches(monkeypatch, tmp_path):
    _FakePbar.instances.clear()
    scorer = _FakeScorer()

    monkeypatch.setattr(ranker, "_LegacyProgress", _FakePbar)
    monkeypatch.setattr(ranker, "IndependentRuleGpuScorer", lambda **kwargs: scorer)
    monkeypatch.setattr(ranker, "estimate_word_count", lambda path: 4)
    monkeypatch.setattr(ranker, "load_rules", lambda path: [
        {"rule_id": 0, "rule_data": "l"},
        {"rule_id": 1, "rule_data": "u"},
        {"rule_id": 2, "rule_data": "c"},
    ])
    monkeypatch.setattr(ranker, "load_cracked_hashes", lambda path, max_len: np.array([], dtype=np.uint64))
    monkeypatch.setattr(
        ranker,
        "_encode_rules_matrix",
        lambda rules: np.zeros((len(rules), ranker.MAX_RULE_LEN), dtype=np.uint8),
    )
    monkeypatch.setattr(ranker, "setup_interrupt_handler", lambda *args: None)
    monkeypatch.setattr(ranker, "save_ranking_data", lambda *args, **kwargs: str(tmp_path / "out.csv"))
    monkeypatch.setattr(ranker, "save_top_k_rules", lambda *args, **kwargs: None)
    monkeypatch.setattr(ranker, "update_progress_stats", lambda *args: None)
    monkeypatch.setattr(ranker, "time", lambda: 0.0)
    monkeypatch.setattr(ranker, "interrupted", False)
    monkeypatch.setattr(ranker, "optimized_wordlist_iterator", lambda *args: iter([
        (np.zeros((4, ranker.MAX_WORD_LEN), dtype=np.uint8), np.zeros(4, dtype=np.uint64), 4),
    ]))

    ranker.rank_rules_exhaustive(
        wordlist_path="words.txt",
        rules_path="rules.rule",
        cracked_list_path="cracked.txt",
        ranking_output_path=str(tmp_path / "out.csv"),
        top_k=0,
        words_per_gpu_batch=4,
    )

    assert len(_FakePbar.instances) == 1
    pbar = _FakePbar.instances[0]
    assert pbar.kwargs["stream"] is ranker.sys.stdout
    assert pbar.kwargs["desc"] == "LEGACY GPU"
    assert pbar.n == 12  # 4 words × 3 rules, reached via two callbacks
    assert pbar.total == 12  # normal completion replaces the word-count estimate
    assert pbar.closed is True
    assert any("GPU dispatch" in text for text, _ in pbar.postfixes)
    assert any("complete" in text for text, _ in pbar.postfixes)


def test_legacy_progress_bar_not_created_during_gpu_init(monkeypatch, tmp_path):
    """The bar must not exist (and so cannot render anything) while the GPU
    scorer is still initializing - only once init succeeds and the batch
    loop is about to start."""
    _FakePbar.instances.clear()
    creation_order = []

    def _tracking_scorer(**kwargs):
        creation_order.append("scorer")
        return _FakeScorer()

    class _TrackingPbar(_FakePbar):
        def __init__(self, *args, **kwargs):
            creation_order.append("pbar")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(ranker, "_LegacyProgress", _TrackingPbar)
    monkeypatch.setattr(ranker, "IndependentRuleGpuScorer", _tracking_scorer)
    monkeypatch.setattr(ranker, "estimate_word_count", lambda path: 4)
    monkeypatch.setattr(ranker, "load_rules", lambda path: [
        {"rule_id": 0, "rule_data": "l"},
        {"rule_id": 1, "rule_data": "u"},
        {"rule_id": 2, "rule_data": "c"},
    ])
    monkeypatch.setattr(ranker, "load_cracked_hashes", lambda path, max_len: np.array([], dtype=np.uint64))
    monkeypatch.setattr(
        ranker,
        "_encode_rules_matrix",
        lambda rules: np.zeros((len(rules), ranker.MAX_RULE_LEN), dtype=np.uint8),
    )
    monkeypatch.setattr(ranker, "setup_interrupt_handler", lambda *args: None)
    monkeypatch.setattr(ranker, "save_ranking_data", lambda *args, **kwargs: str(tmp_path / "out.csv"))
    monkeypatch.setattr(ranker, "save_top_k_rules", lambda *args, **kwargs: None)
    monkeypatch.setattr(ranker, "update_progress_stats", lambda *args: None)
    monkeypatch.setattr(ranker, "time", lambda: 0.0)
    monkeypatch.setattr(ranker, "interrupted", False)
    monkeypatch.setattr(ranker, "optimized_wordlist_iterator", lambda *args: iter([
        (np.zeros((4, ranker.MAX_WORD_LEN), dtype=np.uint8), np.zeros(4, dtype=np.uint64), 4),
    ]))

    ranker.rank_rules_exhaustive(
        wordlist_path="words.txt",
        rules_path="rules.rule",
        cracked_list_path="cracked.txt",
        ranking_output_path=str(tmp_path / "out.csv"),
        top_k=0,
        words_per_gpu_batch=4,
    )

    # The scorer (GPU init) must be created strictly before the bar.
    assert creation_order == ["scorer", "pbar"]
    pbar = _TrackingPbar.instances[0]
    # No postfix mentions "init" - the bar only ever shows post-init state.
    assert not any("init" in text.lower() for text, _ in pbar.postfixes)


def test_legacy_progress_words_climb_within_a_single_batch(monkeypatch, tmp_path):
    """`words=` must move during a batch, not sit at 0 until the whole batch
    (which can be the entire wordlist) finishes."""
    _FakePbar.instances.clear()
    scorer = _FakeScorer()

    monkeypatch.setattr(ranker, "_LegacyProgress", _FakePbar)
    monkeypatch.setattr(ranker, "IndependentRuleGpuScorer", lambda **kwargs: scorer)
    monkeypatch.setattr(ranker, "estimate_word_count", lambda path: 4)
    monkeypatch.setattr(ranker, "load_rules", lambda path: [
        {"rule_id": 0, "rule_data": "l"},
        {"rule_id": 1, "rule_data": "u"},
        {"rule_id": 2, "rule_data": "c"},
    ])
    monkeypatch.setattr(ranker, "load_cracked_hashes", lambda path, max_len: np.array([], dtype=np.uint64))
    monkeypatch.setattr(
        ranker,
        "_encode_rules_matrix",
        lambda rules: np.zeros((len(rules), ranker.MAX_RULE_LEN), dtype=np.uint8),
    )
    monkeypatch.setattr(ranker, "setup_interrupt_handler", lambda *args: None)
    monkeypatch.setattr(ranker, "save_ranking_data", lambda *args, **kwargs: str(tmp_path / "out.csv"))
    monkeypatch.setattr(ranker, "save_top_k_rules", lambda *args, **kwargs: None)
    monkeypatch.setattr(ranker, "update_progress_stats", lambda *args: None)
    monkeypatch.setattr(ranker, "time", lambda: 0.0)
    monkeypatch.setattr(ranker, "interrupted", False)
    # A single batch covering the entire (estimated) wordlist, mirroring the
    # real-world case that looked "stuck" - words_per_gpu_batch >= total words.
    monkeypatch.setattr(ranker, "optimized_wordlist_iterator", lambda *args: iter([
        (np.zeros((4, ranker.MAX_WORD_LEN), dtype=np.uint8), np.zeros(4, dtype=np.uint64), 4),
    ]))

    ranker.rank_rules_exhaustive(
        wordlist_path="words.txt",
        rules_path="rules.rule",
        cracked_list_path="cracked.txt",
        ranking_output_path=str(tmp_path / "out.csv"),
        top_k=0,
        words_per_gpu_batch=4,
    )

    pbar = _FakePbar.instances[0]
    dispatch_postfixes = [text for text, _ in pbar.postfixes if text.startswith("GPU dispatch ")]
    # Two dispatch callbacks fire (see _FakeScorer): progress_callback(0, 2, 3)
    # then progress_callback(2, 3, 3). words= must already show nonzero
    # movement on the first one (2/3 of the rules done, not 0), not wait for
    # the batch to fully finish.
    assert len(dispatch_postfixes) == 2
    assert "words=0/" not in dispatch_postfixes[0]
    assert "words=2/" in dispatch_postfixes[0]  # 2 of 3 rules done -> 2/3 of this batch's 4 words
    assert "words=4/" in dispatch_postfixes[1]  # batch's rules fully dispatched -> all 4 words


def test_legacy_progress_writes_immediately_without_tty(monkeypatch):
    writes = []

    class _Stdout:
        def isatty(self):
            return False

        def fileno(self):
            return 123

        def write(self, text):
            raise AssertionError("buffered fallback should not be used")

        def flush(self):
            raise AssertionError("buffered fallback should not be used")

    monkeypatch.setattr(ranker.sys, "stdout", _Stdout())
    monkeypatch.setattr(ranker.os, "write", lambda fd, data: writes.append((fd, data)) or len(data))
    monkeypatch.setattr(ranker, "time", lambda: 10.0)

    pbar = ranker._LegacyProgress(total=100, desc="LEGACY GPU", unit="rule·word", interval=0.5)
    try:
        pbar.set_postfix_str("GPU dispatch 1/4", refresh=True)
        pbar.update(25)
    finally:
        pbar.close()
        pbar._heartbeat.set()

    output = b"".join(data for _, data in writes).decode("utf-8")
    assert writes
    assert all(fd == 123 for fd, _ in writes)
    assert "LEGACY GPU" in output
    assert "25.0%" in output
    assert "GPU dispatch 1/4" in output
    assert output.endswith("\n")


def test_legacy_progress_heartbeat_keeps_output_alive(monkeypatch):
    writes = []
    monkeypatch.setattr(ranker, "_write_progress_bytes", lambda text: writes.append(text))

    pbar = ranker._LegacyProgress(total=1000, desc="LEGACY GPU", unit="rule·word", interval=0.25)
    try:
        pbar.set_postfix_str("GPU dispatch running", refresh=True)
        import time as _time
        _time.sleep(0.65)
    finally:
        pbar.close()

    # More than the initial frame means the heartbeat produced visible output
    # even though no GPU dispatch callback advanced the counter.
    assert len(writes) >= 3
    assert all("LEGACY GPU" in text for text in writes)
    assert any("GPU dispatch running" in text for text in writes)
