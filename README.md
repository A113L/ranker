# Hashcat Rule Ranker

> **GPU-Accelerated Hashcat Rule Ranking with sample-based MAB, independent per-rule 64-bit fingerprints, CELF greedy max-coverage post-processing, and fast CSV/rule analysis — all under one `rule_ranker` package**

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
- **Sample-Based MAB mode** (default) — Thompson Sampling uses fresh stratified wordlist samples per trial instead of rescanning the full wordlist for every bandit update
- **Legacy exhaustive mode** — tests every rule against every word (full-pass reference mode)
- **Validated Hashcat-compatible subset** — explicitly filtered GPU-supported rule syntax up to 255 characters; no silent truncation
- **Adaptive VRAM management** — sizes independent per-rule fingerprint tables from the requested word batch/sample and available VRAM
- **Memory-mapped file I/O** — handles wordlists of any size with minimal RAM overhead
- **Graceful interrupt handling** — `Ctrl+C` saves intermediate results so a run can be inspected
- **Dual output** — ranked CSV with full statistics and a ready-to-use `.rule` file of top-K rules
- **CELF post-processing stage** (`postprocess`) — GPU recompute + lazy-greedy max-coverage selection on top of `rank`'s output, so the final ruleset is chosen for combined coverage rather than individual score alone
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
python run_ranker.py rank -w words.txt -r rules.rule -c cracked.txt --list-devices

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
| `--global-bits` | legacy | Deprecated compatibility option; independent 64-bit scoring no longer uses a shared global bitmap |
| `--cracked-bits` | legacy | Deprecated compatibility option; cracked membership now uses a 64-bit open-addressing table |
| `--preset` | — | Legacy compatibility option; MAB sizing is derived from sample size + VRAM |

### MAB Options

| Argument | Default | Description |
|---|---|---|
| `--mab-exploration` | `2.0` | UCB / Thompson exploration factor |
| `--mab-final-trials` | `50` | Minimum trials required before a surviving rule is finalised |
| `--mab-screening-trials` | `5` | Sample trials before a rule is eligible for early elimination |
| `--mab-sample-words` | `8192` | Words sampled per MAB trial; samples are stratified across the wordlist |
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

1. **Fresh stratified sample** — each MAB iteration reads only `--mab-sample-words` valid words, drawing windows from byte ranges spread across the wordlist. The full wordlist is not rescanned for every trial.
2. **Screening phase** — every rule receives `--mab-screening-trials` sample trials. Rules with no observed cracked outputs can be eliminated early.
3. **Deep testing** — surviving rules are selected by Thompson Sampling until they reach `--mab-final-trials`.
4. **Independent scoring** — every evaluated rule owns a disjoint 64-bit fingerprint table. A duplicate generated by one rule therefore cannot suppress the uniqueness or effectiveness score of another rule in the same dispatch.

### Fingerprinting

Ranking and CELF use FNV-1a-64 fingerprints stored in linear-probing tables. Table collisions are resolved by probing and never by sharing a bitmap bit between rules. A fixed 64-bit fingerprint is collision-resistant at practical scales, but it is not cryptographically collision-proof; the implementation does not pretend otherwise.

### Legacy Mode (`--legacy`)

Every rule is applied to every word in the wordlist. The implementation still uses independent per-rule fingerprint tables, but uniqueness is accumulated per streamed word batch so VRAM remains bounded. Use this mode as a deterministic/full-pass reference rather than as the fast path for large rule sets.

---

## CELF Post-Processing

`rank`'s Combined_Score ranks rules **individually**. Two top-10 rules might mostly crack the *same* passwords, which makes for a redundant top-K `.rule` file. `postprocess` fixes this: it takes the top-scored candidates from a `rank` run and runs a GPU recompute + lazy-greedy max-coverage selector to pick the subset of rules that covers the most unique cracked passwords for a given rule budget.

### How it works

1. **Upper-bound pass (GPU)** — every candidate is scored once against the full cracked-password universe. These full-target hit counts become monotone upper bounds for later rounds.
2. **Lazy-greedy selection (GPU)** — a single `active` bitset tracks cracked targets that are still uncovered. In each round, only candidates whose upper bound can still beat the current best are re-scored on the GPU. Once a winner is chosen, its newly covered targets are cleared from `active`.
3. **Output** — a `.rule` file containing the selected rules, ordered best-first by marginal gain, plus a `_celf.csv` with each selected rule's incremental gain.

### Heap usage in CELF

The lazy-greedy selector (`celf_recompute_gpu.py`) keeps candidates in a max-heap by upper bound, stored as a min-heap of `(-upper_bound, rule_index)` pairs via `heapq`. Each round pops the current best bound and re-scores it exactly on the GPU; if no remaining bound can beat that exact gain, the round ends immediately without touching the rest of the heap — that's the "lazy" part. Otherwise a small batch of next-best bounds is popped and re-scored together, losers are pushed back with their now-exact gain as a tighter bound, and zero-gain candidates are dropped for good. Ties go to the smaller rule index. This replaces an older flat-array approach that needed more full rescans per round as the uncovered-target set shrank — the heap always surfaces just the most promising unresolved candidate instead.

A smaller heap is also used when loading candidates from a ranking CSV (`ranker_postprocess.py`): with `--candidates` set, rows are streamed and only the current top-C by `Combined_Score` are kept in a bounded heap, so the full CSV never has to sit in memory.

### Memory model

The removed bitmap and sparse coverage stores are **not** part of the post-processing pipeline anymore. `postprocess` never allocates or writes a per-candidate coverage matrix, either as an in-memory array or a disk-backed file. Peak selection state is O(candidates) plus the uncovered-target bitset and the GPU/host wordlist buffers. The trade-off is repeated GPU rule application during lazy re-validation instead of random reads from stored coverage rows.

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

# Export several budget cutoffs from a single run:
python run_ranker.py postprocess --ranking-csv ranker_output.csv \
  --wordlist rockyou.txt --cracked cracked_passwords.txt \
  --budgets 64,250,5000 --output celf_selected.rule

# Explicitly select the only supported strategy (also the default):
python run_ranker.py postprocess --ranking-csv ranker_output.csv \
  --wordlist rockyou.txt --cracked cracked_passwords.txt \
  --budget 5000 --output celf_selected.rule --strategy recompute-gpu
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| -r, --ranking-csv | — | ranker output CSV to source candidates from (mutually exclusive with --rules-file) |
| -f, --rules-file | — | Plain .rule file of already-ranked/optimized rules (mutually exclusive with --ranking-csv) |
| -w, --wordlist | required | Base wordlist |
| -k, --cracked | required | Known-cracked passwords list |
| -o, --output | required | Output .rule path |
| -c, --candidates | 20000 | How many top-scored rules to feed into greedy selection (not the final selection size — see --budget) |
| -b, --budget | none (run to saturation) | Max rules in the final selection; ignored if --budgets is given |
| -B, --budgets | — | Comma-separated budget cutoffs exported as separate files from one run, e.g. 64,250,5000 |
| -R, --rule-batch-size | 1024 | Candidate rules evaluated per GPU dispatch batch and used as the upper-bound scan chunk size |
| -W, --words-batch-size | 150000 | Words per GPU batch when the scorer streams the wordlist |
| --max-word-len | 32 | Words/cracked entries longer than this are skipped (not truncated) |
| --max-rule-len | 255 | Max characters per hashcat rule considered (matches `rank`) |
| --max-output-len | 512 | Max length of a rule's output word the GPU kernel will produce |
| --auto-max-output-len | off | Estimate the worst-case rule output length before the GPU pass and use that value (+ margin) |
| --print-output-len-estimate | off | Print the recommended --max-output-len and exit without touching the GPU |
| -d, --device | — | OpenCL device ID |
| --strategy | recompute-gpu | The only supported coverage/selection strategy; no per-candidate coverage matrix is materialized |

There are intentionally no bitmap-store, sparse-store, parallel-file-I/O, or GPU-CELF tuning flags in this version. Those belonged to the removed `bitmap` and `sparse` strategies.

---


### GPU safety semantics

The GPU rule engine treats `MAX_OUTPUT_LEN` overflow as a **rule rejection for the current word**, not as an empty intermediate word. If any command would exceed the configured output buffer, the current rule chain stops for that word and produces no candidate hash. This prevents a later command in the same chain from accidentally operating on a zero-length intermediate.

The `rank` and `postprocess` stages both support rules up to **255 characters** by default. Post-processing skips rules longer than the configured `--max-rule-len` instead of silently truncating their bytes.

### OpenCL integration testing

The test suite contains a real OpenCL integration test for the CELF GPU scorer and CPU-side regression tests for the 64-bit fingerprint table, sample-based MAB wiring, and top-K boundary handling. It executes the generated OpenCL kernels against a small reference workload and also covers the empty-input rotate guard and output-overflow rejection. The test is run automatically when a working OpenCL Python binding plus at least one OpenCL CPU/GPU device is available; on machines without an OpenCL ICD/device it is explicitly skipped rather than counted as a passing GPU test.

For CI, use a portable CPU OpenCL implementation such as **PoCL** together with `pyopencl` to make this integration test reproducible on machines without a discrete GPU.


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
│   ├── celf_recompute_gpu.py# GPU recompute + lazy-greedy CELF implementation used by postprocess
│   ├── hashing.py           # shared hashing helpers (cracked-password/global hash maps)
│   └── progress.py          # shared progress-bar/console output helpers
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
