# rule_ranker

A modular package wrapping three previously-standalone scripts into one
installable Python package with a single dispatcher CLI:

| Original script          | Package module                    | What it does                                              |
|---------------------------|------------------------------------|-------------------------------------------------------------|
| `ranker.py`                | `rule_ranker.ranker`               | GPU-accelerated Hashcat rule ranking (MAB or legacy exhaustive) |
| `ranker_handler.py`        | `rule_ranker.ranker_handler`       | Fast analysis/summarization of ranker output CSVs           |
| `ranker_postprocess.py`    | `rule_ranker.ranker_postprocess`   | GPU/CPU CELF greedy max-coverage rule selection post-stage   |

None of the original logic was rewritten -- each file was relocated
into the package as-is, with only its `if __name__ == '__main__':`
block turned into a `main(argv=None)` function so it's callable both as
a script and programmatically (e.g. from tests or the dispatcher).

## Install

```bash
pip install -r requirements.txt
# or, editable install with console script:
pip install -e .
```

`numpy` and `tqdm` are required by all three tools. `pyopencl` (with a
working OpenCL ICD / GPU driver) is required by `rank` and by the GPU
coverage pass of `postprocess`. `handler` has no GPU dependency at all.
`postprocess`'s CELF greedy-select phase runs entirely on CPU (see
`rule_ranker/README_ranker_postprocess.md` for how it parallelizes
across cores).

## Usage

All three tools are reachable through one dispatcher:

```bash
python3 run_ranker.py rank -w wordlist.txt -r rules.rule -c cracked.txt -o out.csv
python3 run_ranker.py rank --legacy -w wordlist.txt -r rules.rule -c cracked.txt -o out.csv
python3 run_ranker.py rank --list-devices

python3 run_ranker.py handler -i out.csv -o summary.txt -r clean.rule -t 1000

python3 run_ranker.py postprocess -r out.csv -w wordlist.txt -k cracked.txt \
    -o celf_selected.rule --candidates 20000 --budget 5000
python3 run_ranker.py postprocess -r out.csv -w wordlist.txt -k cracked.txt \
    -o celf_selected.rule --budgets 64,250,5000 --in-ram
```

Every subcommand's own `--help` shows that tool's full (unchanged)
argument list:

```bash
python3 run_ranker.py rank --help
python3 run_ranker.py handler --help
python3 run_ranker.py postprocess --help
```

Each tool also remains runnable directly as a module, without the
dispatcher:

```bash
python3 -m rule_ranker.ranker ...
python3 -m rule_ranker.ranker_handler ...
python3 -m rule_ranker.ranker_postprocess ...
```

If installed with `pip install -e .`, a `run-ranker` console script is
also available and behaves identically to `python3 run_ranker.py`.

## Typical pipeline

```bash
# 1. Rank candidate rules against a wordlist + cracked list (GPU)
python3 run_ranker.py rank -w rockyou.txt -r hashcat_rules.rule -c cracked.txt -o ranking.csv

# 2. (Optional) Summarize/clean the ranking output
python3 run_ranker.py handler -i ranking.csv -o summary.txt -r top_rules.rule -t 5000

# 3. Run CELF greedy max-coverage selection on the top candidates (GPU coverage pass + CPU/GPU CELF)
python3 run_ranker.py postprocess -r ranking.csv -w rockyou.txt -k cracked.txt \
    -o celf_final.rule --candidates 20000 --budget 5000
```

## Tests

```bash
pip install -r requirements.txt   # or at least: numpy, tqdm, pytest
python3 -m pytest tests/ -v
```

Tests cover the pure-Python/NumPy logic in each module -- rule
validation (including the documented bug fixes in `ranker.py`'s
changelog), CSV/rule-file parsing and analysis in `ranker_handler.py`,
and popcount/budget-parsing/CELF-greedy-selection logic in
`ranker_postprocess.py` -- plus the `run_ranker.py` dispatcher's
routing. GPU/OpenCL kernel code paths themselves are out of scope for
unit tests (they need a real GPU); `tests/conftest.py` installs a
minimal stub for `pyopencl` so the GPU-dependent modules can still be
*imported* (needed to reach their pure-Python code and their
`--help`/argument-parsing paths) in an environment without a real
OpenCL/GPU stack. If `pyopencl` is genuinely installed, the stub is
skipped and the real module is used.

## Package layout

```
rule_ranker_pkg/
├── run_ranker.py              # top-level dispatcher CLI
├── rule_ranker/
│   ├── __init__.py
│   ├── ranker.py               # (was ranker.py)
│   ├── ranker_handler.py       # (was ranker_handler.py)
│   └── ranker_postprocess.py   # (was ranker_postprocess.py)
├── tests/
│   ├── conftest.py
│   ├── test_ranker_validator.py
│   ├── test_handler_parsing.py
│   ├── test_postprocess_utils.py
│   └── test_run_ranker_dispatcher.py
├── requirements.txt
├── setup.py
└── README.md
```
