"""Real OpenCL integration tests for the GPU rule scorer.

The test is intentionally separate from the CPU-only suite.  It runs the
actual generated OpenCL kernels against a tiny deterministic workload whenever
an OpenCL runtime/device is available (GPU or CPU).  In environments without
an ICD/device, pytest skips it explicitly; the skip must not be mistaken for a
passing GPU validation.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pyopencl as cl

from rule_ranker import celf_recompute_gpu as crg
from rule_ranker import ranker_postprocess as rp


@pytest.fixture
def real_opencl_device():
    if getattr(cl, "_RULE_RANKER_STUB", False):
        pytest.skip("No real pyopencl/OpenCL runtime is available in this environment")
    try:
        platforms = cl.get_platforms()
    except Exception as exc:
        pytest.skip(f"OpenCL platform discovery unavailable: {exc}")
    for platform in platforms:
        try:
            devices = platform.get_devices()
        except Exception:
            continue
        if devices:
            # Prefer a GPU when the machine has one; otherwise use the first
            # CPU/other device as the portable reference device.
            gpus = [d for d in devices if d.type == cl.device_type.GPU]
            return (platform, gpus[0] if gpus else devices[0])
    pytest.skip("No OpenCL device/ICD found")


def test_celf_gpu_reference_device_executes_kernels(monkeypatch, tmp_path, real_opencl_device):
    platform, device = real_opencl_device
    monkeypatch.setattr(crg, "select_device", lambda _device_id=None: (platform, device))

    # Two base words. `l` should crack the lowercase forms exactly once each.
    # The second rule deliberately overflows mid-chain: 20 -> 640 with `pV`, which exceeds the current
    # MAX_OUTPUT_LEN=512, then `$X` must NOT run on an empty intermediate.
    # Therefore it must produce no hit for cracked hash of `X`.
    wordlist = tmp_path / "words.txt"
    wordlist.write_bytes(b"HELLO\nWORLD\n" + b"a" * 20 + b"\n")
    cracked = np.array(
        sorted({
            rp.fast_fnv1a_hash_32(b"hello"),
            rp.fast_fnv1a_hash_32(b"world"),
            rp.fast_fnv1a_hash_32(b"X"),
        }),
        dtype=np.uint32,
    )

    rules = np.zeros((2, crg.MAX_RULE_LEN), dtype=np.uint8)
    rule_bytes = [b"l", b"pV$X"]
    rule_lens = np.zeros(2, dtype=np.uint32)
    for i, rb in enumerate(rule_bytes):
        rules[i, :len(rb)] = np.frombuffer(rb, dtype=np.uint8)
        rule_lens[i] = len(rb)

    scorer = crg._GpuScorer(
        rules,
        len(cracked),
        cracked,
        rule_batch_size=2,
        words_per_gpu_batch=16,
        device_id=None,
        wordlist_path=str(wordlist),
        rule_lens=rule_lens,
    )

    gains = scorer.score_batch(np.array([0, 1], dtype=np.int64), str(wordlist))
    assert gains.tolist() == [2, 0]

    cleared = scorer.apply_winner_and_clear(0, "l", str(wordlist))
    assert cleared == 2
    assert scorer.remaining_active_count() == 1
