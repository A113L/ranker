from pathlib import Path

import numpy as np

from rule_ranker.ranker import optimized_wordlist_iterator


def test_optimized_wordlist_iterator_returns_2d_batches(tmp_path: Path):
    wordlist = tmp_path / "words.txt"
    wordlist.write_bytes(b"alpha\nb\ngamma\ndelta\n")

    batches = list(optimized_wordlist_iterator(str(wordlist), max_len=8, batch_size=2))

    assert len(batches) == 2

    words0, hashes0, count0 = batches[0]
    assert count0 == 2
    assert words0.shape == (2, 8)
    assert words0.dtype == np.uint8
    assert bytes(words0[0, :5]) == b"alpha"
    assert bytes(words0[1, :1]) == b"b"
    assert len(hashes0) == 2

    words1, hashes1, count1 = batches[1]
    assert count1 == 2
    assert words1.shape == (2, 8)
    assert bytes(words1[0, :5]) == b"gamma"
    assert bytes(words1[1, :5]) == b"delta"
    assert len(hashes1) == 2
