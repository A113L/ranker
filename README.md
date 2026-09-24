# Hashcat Rule Ranker

> **GPU-Accelerated Hashcat Rule Ranking using Multi-Armed Bandit (MAB) with Early Elimination, plus a CELF greedy max-coverage post-stage**

`ranker.py` evaluates and ranks a Hashcat ruleset against a wordlist and a list of known-cracked passwords. It applies each rule on the GPU via OpenCL, scores rules by how many unique and real-world cracked passwords they produce, and outputs a ranked CSV plus an optimized `.rule` file containing only the top performers.

`ranker_postprocess.py` is an optional second stage that takes `ranker.py`'s output and runs **CELF** (Cost-Effective Lazy Forward selection) — a lazy-greedy max-coverage algorithm — to pick the smallest/most efficient subset of rules that collectively crack the most passwords, rather than just taking the top-K individually-scored rules. See [CELF Post-Processing](#celf-post-processing) below.

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
- **CELF post-processing stage** (`ranker_postprocess.py`) — lazy-greedy max-coverage rule selection on top of `ranker.py`'s output, with a GPU coverage-bitmap pass and a CPU (or GPU-assisted) CELF greedy select, so the final ruleset is chosen for combined coverage rather than individual score alone

---

## Requirements

| Dependency | Notes |
|---|---|
| Python 3.8+ | Minimum Python 3.8 required to run script |
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

## CELF Post-Processing

`ranker.py`'s Combined_Score ranks rules **individually**. Two top-10 rules might mostly crack the *same* passwords, which makes for a redundant top-K `.rule` file. `ranker_postprocess.py` fixes this: it takes the top-scored candidates from a `ranker.py` run and runs **CELF** (Cost-Effective Lazy Forward selection), a lazy-greedy algorithm for the max-coverage problem, to pick the subset of rules that covers the most unique cracked passwords for a given rule budget — with a provably near-optimal guarantee relative to brute-force greedy, at a fraction of the cost.

### How it works

1. **Coverage-bitmap pass (GPU)** — for each candidate rule, the tool applies it across the wordlist and records, as a bitmap over the cracked-password universe, which cracked passwords it produces. This runs in rule batches on the GPU (`--rule-batch-size` / `--words-batch-size`).
2. **CELF greedy select (CPU, or multi-core parallel by default)** — starting from an empty set, CELF repeatedly picks the rule with the highest *marginal* gain in newly-covered passwords, using a lazy priority queue so most rules never need to be re-evaluated on every round (this is what makes it fast compared to naive greedy). Selection stops when the rule budget is hit or coverage saturates.
3. **Output** — a `.rule` file containing the selected rules, ordered best-first by marginal gain, plus a `_celf.csv` with each selected rule's incremental gain.

### Memory modes

The coverage-bitmap matrix (`candidates × ceil(cracked_universe / 32)` `uint32` words) can be large — e.g. ~34 GB for 20,000 candidates against 14.3M unique cracked hashes. By default it's streamed to a disk-backed `np.memmap` (`--bitmap-path`, deleted after a successful run unless `--keep-bitmap` is passed) instead of held fully in RAM. Pass `--in-ram` to use a plain in-RAM array instead — faster (no disk I/O during the GPU write pass or CELF's lazy re-validation reads) but requires the full estimated size (printed at startup) as free RAM.

### Usage

```bash
python ranker_postprocess.py \
  --ranking-csv ranker_output.csv \
  --wordlist rockyou.txt \
  --cracked cracked_passwords.txt \
  --candidates 20000 \
  --budget 5000 \
  --output celf_selected.rule

# Or feed it a plain .rule file instead of a ranking CSV:
python ranker_postprocess.py --rules-file top_optimized.rule \
  --wordlist rockyou.txt --cracked cracked_passwords.txt \
  --budget 5000 --output celf_selected.rule

# Export several budget cutoffs from a single CELF run:
python ranker_postprocess.py --ranking-csv ranker_output.csv \
  --wordlist rockyou.txt --cracked cracked_passwords.txt \
  --budgets 64,250,5000 --output celf_selected.rule

# Keep everything in RAM (faster, needs enough free RAM):
python ranker_postprocess.py --ranking-csv ranker_output.csv \
  --wordlist rockyou.txt --cracked cracked_passwords.txt \
  --budget 5000 --output celf_selected.rule --in-ram
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `-r`, `--ranking-csv` | — | `ranker.py` output CSV to source candidates from (mutually exclusive with `--rules-file`) |
| `-f`, `--rules-file` | — | Plain `.rule` file of already-ranked/optimized rules to use instead of a ranking CSV |
| `-w`, `--wordlist` | required | Base wordlist |
| `-k`, `--cracked` | required | Known-cracked passwords list |
| `-o`, `--output` | required | Output `.rule` path |
| `-c`, `--candidates` | `20000` | How many top-scored rules to feed into CELF (not the final selection size — see `--budget`) |
| `-b`, `--budget` | none (run to saturation) | Max rules in the final selection; ignored if `--budgets` is given |
| `-B`, `--budgets` | — | Comma-separated budget cutoffs exported as separate files from one CELF run, e.g. `64,250,5000` |
| `-R`, `--rule-batch-size` | `1024` | Candidate rules evaluated per GPU dispatch batch; also bounds peak host RAM in memmap mode |
| `-W`, `--words-batch-size` | `150000` | Words per GPU batch for the coverage pass |
| `-d`, `--device` | — | OpenCL device ID |
| `--bitmap-path` | `<output_base>.bitmap.dat` | Where to stream the on-disk coverage-bitmap matrix; ignored if `--in-ram` is set |
| `--keep-bitmap` | — | Flag — don't delete the on-disk bitmap file after a successful run |
| `--in-ram` | — | Flag — build the coverage matrix fully in RAM instead of a disk-backed memmap |
| `--no-parallel-celf` | — | Flag — disable multi-core parallel CELF select and use the single-threaded version |
| `--celf-workers` | `os.cpu_count()` | Worker processes for parallel CELF select |
| `--celf-io-threads` | `4` | Concurrent `os.pread()` calls per worker in parallel CELF select (raise on fast NVMe) |
| `--celf-batch-multiplier` | `8` | How many candidates parallel CELF revalidates per round, as a multiple of `workers × io_threads` |

By default (disk-backed bitmap, i.e. no `--in-ram`), CELF's greedy-select phase runs across all CPU cores via multiprocessing, with each worker issuing several concurrent `os.pread()` calls against the on-disk bitmap file to keep read queue depth up — this is what keeps rules/s high during lazy re-validation on fast storage. Use `--no-parallel-celf` to fall back to the plain single-threaded selector.

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

# Full pipeline: rank, then CELF-select a compact high-coverage ruleset
python ranker.py -w rockyou.txt -r hashcat_rules.rule -c cracked.txt -o ranking.csv
python ranker_postprocess.py --ranking-csv ranking.csv -w rockyou.txt -k cracked.txt \
  --candidates 20000 --budget 5000 -o celf_final.rule
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
- 0xVavaldi for inspiration - https://github.com/0xVavaldi
