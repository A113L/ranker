# Hashcat Rule Ranker v5.0

> **GPU-Accelerated Hashcat Rule Ranking using Multi-Armed Bandit (MAB) with Early Elimination**

`ranker.py` evaluates and ranks a Hashcat ruleset against a wordlist and a list of known-cracked passwords. It applies each rule on the GPU via OpenCL, scores rules by how many unique and real-world cracked passwords they produce, and outputs a ranked CSV plus an optimized `.rule` file containing only the top performers.

---

## Features

- **GPU-accelerated** rule application via [PyOpenCL](https://documen.tician.de/pyopencl/) — supports NVIDIA, AMD, and Intel devices
- **Multi-Pass MAB mode** (default) — Thompson Sampling bandit with a screening phase and early elimination of low-performing rules, dramatically reducing compute time on large rulesets
- **Legacy exhaustive mode** — tests every rule against every word (v3.2 behaviour)
- **Full Hashcat rule support** — all GPU-compatible rules up to 255 characters
- **Adaptive VRAM management** — auto-tunes batch size and hash map dimensions to available GPU memory; supports `low_memory / medium_memory / high_memory / recommend` presets
- **Memory-mapped file I/O** — handles wordlists of any size with minimal RAM overhead
- **Graceful interrupt handling** — `Ctrl+C` saves intermediate results so a run can be inspected
- **Dual output** — ranked CSV with full statistics and a ready-to-use `.rule` file of top-K rules

---

## Requirements

| Dependency | Notes |
|---|---|
| Python 3.8+ | Minimum Python 3.8 reqired to run script |
| `pyopencl` | Requires a working OpenCL runtime (GPU driver or CPU fallback) |
| `numpy` | Numerical operations & data preparation |
| `tqdm` | Progress bars |

Install dependencies:

```bash
pip install pyopencl numpy tqdm
```

> **OpenCL runtime** — you also need a platform-specific runtime installed:
> - NVIDIA: CUDA Toolkit (includes OpenCL)
> - AMD: ROCm or AMDGPU-PRO drivers
> - Intel: Intel OpenCL Runtime

---

## Quick Start

```bash
# Rank rules using MAB mode (recommended)
python ranker.py \
  -w rockyou.txt \
  -r best64.rule \
  -c cracked_passwords.txt \
  -o ranked_output.csv \
  -k 500

# List available OpenCL devices
python ranker.py --list-devices
```

---

## Arguments

### Required

| Argument | Description |
|---|---|
| `-w`, `--wordlist` | Path to the base wordlist |
| `-r`, `--rules` | Path to the Hashcat `.rule` file |
| `-c`, `--cracked` | Path to the list of known-cracked passwords (used for effectiveness scoring) |

### Output

| Argument | Default | Description |
|---|---|---|
| `-o`, `--output` | `ranker_output.csv` | Output CSV file path |
| `-k`, `--topk` | `1000` | Number of top-ranked rules to write to the optimized `.rule` file |

### Performance Tuning

| Argument | Default | Description |
|---|---|---|
| `--batch-size` | auto | Words per GPU batch (overrides auto-detection) |
| `--global-bits` | `35` | Bit width of the global uniqueness hash map |
| `--cracked-bits` | `33` | Bit width of the cracked-password hash map |
| `--preset` | — | Memory preset: `low_memory`, `medium_memory`, `high_memory`, or `recommend` |

### MAB Options

| Argument | Default | Description |
|---|---|---|
| `--mab-exploration` | `2.0` | UCB / Thompson exploration factor |
| `--mab-final-trials` | `50` | Minimum trials required before a surviving rule is finalised |
| `--mab-screening-trials` | `5` | Trials before a rule is eligible for early elimination |
| `--mab-no-zero-eliminate` | — | Flag — disables automatic elimination of rules with zero successes |

### Mode & Device

| Argument | Description |
|---|---|
| `--legacy` | Run in exhaustive mode (v3.2) — tests all rules against all words |
| `--device` | OpenCL device ID to use (see `--list-devices`) |
| `--list-devices` | Print all available OpenCL platforms and devices, then exit |

---

## How It Works

### MAB Mode (default)

1. **Screening phase** — every rule receives a minimum number of trials (`--mab-screening-trials`). Rules that produce zero successes are eliminated early.
2. **Deep-testing phase** — surviving rules are selected by a Thompson Sampling bandit. Rules that consistently underperform are eliminated; high-performing rules receive more trials until all survivors reach `--mab-final-trials`.
3. **Scoring** — each rule accumulates an *effectiveness score* (transforms that match a cracked password) and a *uniqueness score* (transforms that produce any new candidate). The combined score is `effectiveness × 10 + uniqueness + mab_success_probability × 1000`.

### Legacy Mode (`--legacy`)

Every rule is applied to every word in the wordlist in a single exhaustive pass. Use this when you need reproducible, fully-deterministic rankings or when the ruleset is small enough that MAB overhead is not worthwhile.

---

## Output Files

| File | Description |
|---|---|
| `<output>.csv` | Full ranked list of all rules with scores and MAB statistics |
| `<output>_optimized.rule` | Top-K rules in Hashcat `.rule` format, ready to use |
| `<output>_INTERRUPTED.csv` | Saved automatically on `Ctrl+C` |
| `<output>_INTERRUPTED.rule` | Saved automatically on `Ctrl+C` |

### CSV Columns (MAB mode)

| Column | Description |
|---|---|
| `Rank` | Final rank (1 = best) |
| `Combined_Score` | Weighted composite score |
| `Effectiveness_Score` | Transforms matching known-cracked passwords |
| `Uniqueness_Score` | Unique candidate words generated |
| `MAB_Success_Prob` | Thompson Sampling success probability |
| `Times_Tested` | Number of batches this rule was tested in |
| `MAB_Trials` | Total MAB trial count |
| `Selections` | Times selected by the bandit |
| `Total_Successes` | Cumulative successes across all trials |
| `Total_Trials` | Cumulative trials across all batches |
| `Eliminated` | Whether the rule was eliminated early |
| `Eliminate_Reason` | `zero_success` or `low_success_rate` |
| `Rule_Data` | Original Hashcat rule string |

---

## Examples

```bash
# Use a specific GPU (device 1)
python ranker.py -w words.txt -r rules.rule -c cracked.txt --device 1

# Low-VRAM machine
python ranker.py -w words.txt -r rules.rule -c cracked.txt --preset low_memory

# Aggressive exploration with more final trials
python ranker.py -w words.txt -r rules.rule -c cracked.txt \
  --mab-exploration 3.0 --mab-final-trials 100

# Legacy exhaustive mode, save top 2000 rules
python ranker.py -w words.txt -r rules.rule -c cracked.txt --legacy -k 2000

# Interrupt safely — progress is written to *_INTERRUPTED files
# Press Ctrl+C at any time during a run
```

---

## Notes

- The cracked passwords file is used **only** for scoring (effectiveness); it is not required to be the original hash file — a plaintext dump of previously cracked passwords works perfectly.
- Comments (`#`) and blank lines in the rules file are ignored automatically.
- Words longer than 256 bytes are silently skipped.
- On systems with multiple OpenCL platforms (e.g., both an NVIDIA GPU and an Intel integrated GPU), the tool auto-selects in preference order: NVIDIA → AMD → Intel → first available. Use `--device` to override.

---


## 📄 License

See `LICENSE` for details.


🙏 **Credits**

- Hashcat community for rule sets and inspiration
- PyOpenCL developers for GPU bindings
- Cybersecurity researchers worldwide
- 0xVavaldi for inspiration - https://github.com/0xVavaldi
