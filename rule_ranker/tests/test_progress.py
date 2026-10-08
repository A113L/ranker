import io

from rule_ranker.progress import ProgressBar


def test_progress_uses_single_carriage_returned_line_and_throttles_updates(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr("rule_ranker.progress.time.monotonic", lambda: 1.0)

    pbar = ProgressBar(
        total=100,
        desc="TEST",
        unit="item",
        stream=stream,
        min_interval=0.5,
        heartbeat=False,
    )
    pbar.update(1)
    pbar.update(1)
    pbar.set_postfix_str("phase=work")
    pbar.close()

    output = stream.getvalue()
    assert "TEST" in output
    assert "phase=work" in output
    assert "2.0%" in output
    assert "\r" in output
    # A progress update must not create a newline for every item update.
    assert output.count("\n") == 1


def test_progress_context_manager_finishes_on_actual_count(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr("rule_ranker.progress.time.monotonic", lambda: 10.0)

    with ProgressBar(4, "WORK", "unit", stream=stream, heartbeat=False) as pbar:
        pbar.update(3)

    output = stream.getvalue()
    assert "75.0%" in output
    assert "3/4" in output
    assert output.endswith("\n")
