# Hashcat Rule Ranker

> **GPU-Accelerated Hashcat Rule Ranking using Multi-Armed Bandit (MAB) with Early Elimination, a CELF greedy max-coverage post-stage, and a fast CSV/rule analysis tool — all under one `rule_ranker` package**

The project is three tools sharing one package, reachable through a single dispatcher (`run_ranker.py`):

| Subcommand | Module | What it does |
|---|---|---|
| `rank` | `rule_ranker.ranker` | Applies each rule to a wordlist on the GPU via OpenCL, scores rules by how many unique and real-world cracked passwords they produce, and writes a ranked CSV plus an optimized `.rule` file of the top performers. |
| `postprocess` | `rule_ranker.ranker_postprocess` | Optional second stage. Takes `rank`'s output and runs **CELF** (Cost-Effective Lazy Forward selection) — a lazy-greedy max-coverage algorithm — to pick the smallest/most efficient subset of rules that collectively crack the most passwords, rather than just taking the top-K individually-scored rules. See [CELF Post-Processing](#celf-post-processing). |
| `handler` | `rule_ranker.ranker_handler` | Optional analysis utility. Parses one or more `rank` output CSVs (or plain rule files), de-duplicates and aggregates rules across files, and writes a plain-text summary plus a clean top-N `.rule` file — without touching the GPU. See [Handler (CSV/Rule Analysis)](#handler-csvrule-analysis). |

Each submodule is a thin wrapper around the original standalone script — nothing about the ranking or CELF logic was rewritten, they were just packaged with a callable `main(argv=None)` so they can be run through `run_ranker.py`, as `python3 -m rule_ranker.<module>`, or (once installed) as the `run-ranker` console script.

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
- **CELF post-processing stage** (`postprocess`) — lazy-greedy max-coverage rule selection on top of `rank`'s output, with a GPU coverage-bitmap pass and a CPU (or GPU-assisted) CELF greedy select, so the final ruleset is chosen for combined coverage rather than individual score alone
- **Fast analysis stage** (`handler`) — merges and de-duplicates rules across multiple ranking runs, reports occurrence/consistency across files, and extracts a clean top-N `.rule` file, all CPU-side and without any GPU/OpenCL dependency

---

## Requirements

| Dependency | Notes |
|---|---|
| Python 3.8+ | Minimum Python 3.8 required to run the package |
| `pyopencl` | Requires a working OpenCL runtime (GPU driver or CPU fallback). Only needed for `rank` and `postprocess` — `handler` has no GPU dependency |
| `numpy` | Numerical operations & data preparation |
| `tqdm` | Progress bars |

Install dependencies:

```bash
pip install -r requirements.txt
```

> **OpenCL runtime** — for `rank` and `postprocess` you also need a platform-specific runtime installed:
> - NVIDIA: CUDA Toolkit (includes OpenCL)
> - AMD: ROCm or AMDGPU-PRO drivers
> - Intel: Intel OpenCL Runtime
>
> `handler` is pure CPU/Python and does not need OpenCL at all — see the note under [Mode & Device](#mode--device) on lazy dependency loading.

---

## Quick Start

```bash
# Rank rules using MAB mode (recommended)
python run_ranker.py rank \
  -w rockyou.txt \
  -r best64.rule \
  -c cracked_passwords.txt \
  -o ranked_output.csv \
  -k 500

# List available OpenCL devices
python run_ranker.py rank --list-devices

# Any subcommand also works invoked directly as a module, e.g.:
python -m rule_ranker.ranker -w rockyou.txt -r best64.rule -c cracked_passwords.txt -o ranked_output.csv
```

Top-level usage:

```bash
python run_ranker.py --help
python run_ranker.py <rank|handler|postprocess> --help   # each subcommand's own argument list
```

---

## Arguments (`rank`)

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

Note: `run_ranker.py` imports each subcommand's module lazily, only once you actually invoke it. So `python run_ranker.py handler ...` works even on a machine with no OpenCL runtime or `pyopencl` installed — only `rank` and `postprocess` need that dependency present.

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

`rank`'s Combined_Score ranks rules **individually**. Two top-10 rules might mostly crack the *same* passwords, which makes for a redundant top-K `.rule` file. `postprocess` fixes this: it takes the top-scored candidates from a `rank` run and runs **CELF** (Cost-Effective Lazy Forward selection), a lazy-greedy algorithm for the max-coverage problem, to pick the subset of rules that covers the most unique cracked passwords for a given rule budget — with a provably near-optimal guarantee relative to brute-force greedy, at a fraction of the cost.

### How it works

1. **Coverage-bitmap pass (GPU)** — for each candidate rule, the tool applies it across the wordlist and records, as a bitmap over the cracked-password universe, which cracked passwords it produces. This runs in rule batches on the GPU (`--rule-batch-size` / `--words-batch-size`).
2. **CELF greedy select (CPU, or multi-core parallel by default)** — starting from an empty set, CELF repeatedly picks the rule with the highest *marginal* gain in newly-covered passwords, using a lazy priority queue so most rules never need to be re-evaluated on every round (this is what makes it fast compared to naive greedy). Selection stops when the rule budget is hit or coverage saturates.
3. **Output** — a `.rule` file containing the selected rules, ordered best-first by marginal gain, plus a `_celf.csv` with each selected rule's incremental gain.

### Memory modes

The coverage-bitmap matrix (`candidates × ceil(cracked_universe / 32)` `uint32` words) can be large — e.g. ~34 GB for 20,000 candidates against 14.3M unique cracked hashes. By default it's streamed to a disk-backed `np.memmap` (`--bitmap-path`, deleted after a successful run unless `--keep-bitmap` is passed) instead of held fully in RAM. Pass `--in-ram` to use a plain in-RAM array instead — faster (no disk I/O during the GPU write pass or CELF's lazy re-validation reads) but requires the full estimated size (printed at startup) as free RAM.

### Usage

```bash
python run_ranker.py postprocess \
  --ranking-csv ranker_output.csv \
  --wordlist rockyou.txt \
  --cracked cracked_passwords.txt \
  --candidates 20000 \
  --budget 5000 \
  --output celf_selected.rule

# Or feed it a plain .rule file instead of a ranking CSV:
python run_ranker.py postprocess --rules-file top_optimized.rule \
  --wordlist rockyou.txt --cracked cracked_passwords.txt \
  --budget 5000 --output celf_selected.rule

# Export several budget cutoffs from a single CELF run:
python run_ranker.py postprocess --ranking-csv ranker_output.csv \
  --wordlist rockyou.txt --cracked cracked_passwords.txt \
  --budgets 64,250,5000 --output celf_selected.rule

# Keep everything in RAM (faster, needs enough free RAM):
python run_ranker.py postprocess --ranking-csv ranker_output.csv \
  --wordlist rockyou.txt --cracked cracked_passwords.txt \
  --budget 5000 --output celf_selected.rule --in-ram
```

### Arguments

| Argument                    | Default                  | Description                                                                                                                                                                             |
| --------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| -r, --ranking-csv           | —                        | ranker output CSV to source candidates from (mutually exclusive with --rules-file)                                                                                                      |
| -f, --rules-file            | —                        | Plain .rule file of already-ranked/optimized rules (mutually exclusive with --ranking-csv)                                                                                              |
| -w, --wordlist              | required                 | Base wordlist                                                                                                                                                                           |
| -k, --cracked               | required                 | Known-cracked passwords list                                                                                                                                                            |
| -o, --output                | required                 | Output .rule path                                                                                                                                                                       |
| -c, --candidates            | 20000                    | How many top-scored rules to feed into CELF (not the final selection size — see --budget)                                                                                               |
| -b, --budget                | none (run to saturation) | Max rules in the final selection; ignored if --budgets is given                                                                                                                         |
| -B, --budgets               | —                        | Comma-separated budget cutoffs exported as separate files from one CELF run, e.g. 64,250,5000                                                                                           |
| -R, --rule-batch-size       | 1024                     | Candidate rules evaluated per GPU dispatch batch; also bounds peak host RAM in memmap mode                                                                                              |
| -W, --words-batch-size      | 150000                   | Words per GPU batch for the coverage pass                                                                                                                                               |
| --max-word-len              | 32                       | Words/cracked entries longer than this are skipped (not truncated); also sizes the GPU kernel's per-thread word buffer                                                                  |
| --max-rule-len              | 32                       | Max characters per hashcat rule considered                                                                                                                                              |
| --max-output-len            | 64                       | Max length of a rule's output word the GPU kernel will produce                                                                                                                          |
| --auto-max-output-len       | off                      | Before the GPU pass, run a fast CPU-only static estimate over every candidate rule and raise --max-output-len if needed                                                                 |
| --print-output-len-estimate | off                      | Run the same static estimate, print the recommended --max-output-len (and the rule responsible for the worst case), then exit without touching the GPU                                  |
| -d, --device                | —                        | OpenCL device ID                                                                                                                                                                        |
| --bitmap-path               | <output_base>.bitmap.dat | Where to stream the on-disk coverage-bitmap matrix; ignored if --in-ram is set                                                                                                          |
| --keep-bitmap                | off                      | Don't delete the on-disk bitmap file after a successful run                                                                                                                             |
| --no-hybrid                  | off                      | Use the old dense, fixed-stride on-disk bitmap format instead of the default hybrid dense/sparse format                                                                                 |
| --in-ram                    | off                      | Build the coverage matrix fully in RAM instead of a disk-backed memmap                                                                                                                  |
| --no-parallel-celf          | off                      | Disable multi-core parallel CELF select and use the single-threaded version                                                                                                             |
| --celf-workers              | os.cpu_count()           | Worker processes for parallel CELF select                                                                                                                                               |
| --celf-io-threads           | 4                        | Concurrent os.pread() calls per worker in parallel CELF select (raise on fast NVMe)                                                                                                     |
| --celf-batch-multiplier     | 8                        | How many candidates parallel CELF revalidates per round, as a multiple of workers × io_threads                                                                                          |
| --strategy                  | bitmap                   | Coverage + selection strategy: bitmap (default dense/hybrid matrix), recompute-gpu (no matrix; re-score survivors on GPU each CELF round), or sparse (sparse coverage store + CPU CELF) |
| --sparse-disk-threshold     | 100000                   | --strategy sparse only: switch coverage store from in-memory dict to SQLite above this many candidates                                                                                  |
| --sparse-store-path         | —                        | --strategy sparse only: persist the SQLite coverage store at this path instead of a temp file                                                                                           |
| --gpu-celf                  | off                      | --strategy sparse only: run the CELF greedy-select loop on the GPU instead of CPU                                                                                                       |
| --gpu-celf-batch            | (module default)         | --strategy sparse --gpu-celf only: max stale heap entries revalidated per GPU dispatch                                                                                                  |

By default (disk-backed bitmap, i.e. no `--in-ram`), CELF's greedy-select phase runs across all CPU cores via multiprocessing, with each worker issuing several concurrent `os.pread()` calls against the on-disk bitmap file to keep read queue depth up — this is what keeps rules/s high during lazy re-validation on fast storage. Use `--no-parallel-celf` to fall back to the plain single-threaded selector.

---

## Handler (CSV/Rule Analysis)

`handler` is a lightweight, GPU-free companion tool for working with `rank`'s output after the fact. It's useful when you have several ranking runs (different wordlists, different rule sets, re-runs over time) and want one clean, de-duplicated view across all of them, without re-running any GPU work.

It accepts one or more ranking CSVs (or plain rule/text files) produced by `rank`, supports both the legacy 5-column and MAB 13-column CSV formats, and will:

- Parse and clean rule strings (with caching, so repeated rules across files are cheap to re-process)
- Aggregate scores per unique (cleaned) rule and flag rules that appear in more than one input file, along with their average score across files
- Extract the top-N rules by `Combined_Score` using a heap (avoids a full sort for large rulesets)
- Optionally process multiple input files in parallel, across CPU cores
- Write a human-readable text summary, an optional clean `.rule` file of the top rules, and an optional rules-with-scores file for debugging
- Print a colorized console summary, including a rules/sec throughput figure for the analysis pass itself

### Usage

```bash
# Summarize a single ranking run, with a top-1000 clean rule file
python run_ranker.py handler -i ranking.csv -t 1000 -r top_1000_rules.rule

# Merge and de-duplicate across several ranking runs
python run_ranker.py handler -i run1.csv run2.csv run3.csv -o combined_summary.txt --parallel

# Just want the clean rule file, skip the text summary
python run_ranker.py handler -i ranking.csv -r hashcat_rules.rule -t 10000 --rules-only
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `-i`, `--input` | required | One or more input ranking file(s), in CSV or TXT format |
| `-o`, `--output` | `rule_analysis_summary.txt` | Output file for the analysis summary |
| `-r`, `--rules-output` | — | Output `.rule` file of cleaned, de-duplicated Hashcat rules |
| `-s`, `--scores-output` | — | Output file of rules with scores (for debugging) |
| `-t`, `--top` | all rules | Number of top rules to extract |
| `--console-top` | `20` | Number of top rules to show in the console summary |
| `--no-console` | off | Don't print results to the console |
| `--parallel` | off | Process multiple input files in parallel (one process per file, up to CPU count) |
| `--chunk-size` | `10000` | Chunk size used while streaming/parsing each input file |
| `--no-progress` | off | Disable progress bars |
| `--rules-only` | off | Only write the clean `.rule` file; skip the text analysis summary |
| `--no-stats` | off | Skip statistics: no per-rule occurrence/score tracking and no analysis summary file (-o) |
---

## Output Files

### `rank`

| File | Description |
|---|---|
| `<output>.csv` | Full ranked list of all rules with scores and MAB statistics |
| `<output>_optimized.rule` | Top-K rules in Hashcat `.rule` format, ready to use |
| `<output>_INTERRUPTED.csv` | Saved automatically on `Ctrl+C` |
| `<output>_INTERRUPTED.rule` | Saved automatically on `Ctrl+C` |

### `postprocess`

| File | Description |
|---|---|
| `<output>` | Selected rules in Hashcat `.rule` format, ordered best-first by marginal gain (one file per budget if `--budgets` is used) |
| `<output>_celf.csv` | Each selected rule's incremental coverage gain |

### `handler`

| File | Description |
|---|---|
| `<output>` (`--output`) | Text analysis summary: totals/averages, top-N rules, and rules that recur across multiple input files |
| `<rules-output>` (`--rules-output`) | Clean, de-duplicated top-N `.rule` file |
| `<scores-output>` (`--scores-output`) | Same as above but with scores included, for debugging |

### CSV Columns (MAB mode, `rank`)

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

`handler` accepts this format as well as the older legacy 5-column CSV.

---

## Examples

```bash
# Use a specific GPU (device 1)
python run_ranker.py rank -w words.txt -r rules.rule -c cracked.txt --device 1

# Low-VRAM machine
python run_ranker.py rank -w words.txt -r rules.rule -c cracked.txt --preset low_memory

# Aggressive exploration with more final trials
python run_ranker.py rank -w words.txt -r rules.rule -c cracked.txt \
  --mab-exploration 3.0 --mab-final-trials 100

# Legacy exhaustive mode, save top 2000 rules
python run_ranker.py rank -w words.txt -r rules.rule -c cracked.txt --legacy -k 2000

# Interrupt safely — progress is written to *_INTERRUPTED files
# Press Ctrl+C at any time during a run

# Full pipeline: rank, CELF-select a compact high-coverage ruleset, then summarize it
python run_ranker.py rank -w rockyou.txt -r hashcat_rules.rule -c cracked.txt -o ranking.csv
python run_ranker.py postprocess --ranking-csv ranking.csv -w rockyou.txt -k cracked.txt \
  --candidates 20000 --budget 5000 -o celf_final.rule
python run_ranker.py handler -i ranking.csv -t 5000 -r top_5000_rules.rule
```

---

## Project Layout

```
rule_ranker/
├── run_ranker.py          # single CLI entry point / subcommand dispatcher
├── setup.py                # pip-installable; adds the `run-ranker` console script
├── requirements.txt
├── rule_ranker/
│   ├── __init__.py
│   ├── ranker.py            # `rank` — GPU MAB/legacy rule ranking
│   ├── ranker_postprocess.py# `postprocess` — CELF max-coverage selection
│   ├── ranker_handler.py    # `handler` — fast CSV/rule analysis (CPU-only)
│   ├── sparse_coverage.py   # sparse coverage store used by postprocess --strategy sparse
│   └── celf_recompute_gpu.py# GPU-assisted CELF path used by postprocess --strategy recompute-gpu / --gpu-celf
└── tests/                   # pytest suite covering the above
```

## Notes

- The cracked passwords file is used **only** for scoring (effectiveness); it is not required to be the original hash file — a plaintext dump of previously cracked passwords works perfectly.
- Comments (`#`) and blank lines in the rules file are ignored automatically.
- Words longer than 256 bytes are silently skipped.
- On systems with multiple OpenCL platforms (e.g., both an NVIDIA GPU and an Intel integrated GPU), the tool auto-selects in preference order: NVIDIA → AMD → Intel → first available. Use `--device` to override.
- `handler` never touches the GPU or requires `pyopencl`, so it's safe to run on a machine that only has the ranking CSVs copied over from wherever `rank`/`postprocess` actually ran.

---

## 📄 License

See `LICENSE` for details.


🙏 **Credits**

- Hashcat community for rule sets and inspiration
- 0xVavaldi for inspiration - https://github.com/0xVavaldi
