# ranker_handler.py

> **Fast, optimized rule scoring analysis tool for Hashcat ranking CSV files.**  
> Processes large rule sets 5–10× faster than naive approaches via vectorized operations, heap-based top-N extraction, and optional parallel file processing.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Input File Format](#input-file-format)
- [Usage](#usage)
  - [Basic Examples](#basic-examples)
  - [All Arguments](#all-arguments)
- [Output Files](#output-files)
- [Performance Notes](#performance-notes)
- [How It Works](#how-it-works)
- [License](#license)

---

## Overview

`ranker_handler.py` reads one or more ranking CSV files produced by a rule scoring pipeline and generates:

- A human-readable **analysis summary** with aggregate statistics.
- A clean **Hashcat `.rule` file** containing deduplicated top-ranked rules.
- An optional **scored rules file** (useful for debugging and further filtering).

It supports both **legacy 5-column** and **MAB 13-column** ranking CSV formats automatically.

---

## Features

| Feature | Detail |
|---|---|
| Multi-format support | Auto-detects legacy (5-col) and MAB (13-col) CSV schemas |
| Heap-based top-N | `O(n log k)` extraction — no full sort required for large sets |
| LRU rule cache | Deduplicates bracket-cleaning work across repeated rule patterns |
| Parallel processing | `ProcessPoolExecutor` across multiple input files (`--parallel`) |
| Buffered I/O | Chunked writes avoid memory pressure on large output files |
| Progress bars | `tqdm` bars for file reading, analysis, sorting, and saving |
| Coloured console | ANSI-coloured terminal output for at-a-glance status |
| No heavy dependencies | Pure stdlib + `tqdm` only (no NumPy/pandas required) |

---

## Requirements

- Python **3.8+**
- [`tqdm`](https://github.com/tqdm/tqdm)

---

## Installation

```bash
# Clone or copy the script, then install the single dependency:
pip install tqdm

# Make executable (optional, Linux/macOS)
chmod +x ranker_handler.py
```

---

## Input File Format

The tool accepts **CSV** (or `.txt`) ranking files with the following column layout.  
A header row is detected automatically and skipped.

### Legacy format (5 columns)

```
rank, combined_score, effectiveness_score, uniqueness_score, rule_data
```

### MAB format (13 columns)

```
rank, combined_score, effectiveness_score, uniqueness_score, <9 extra columns…>, rule_data
```

> The rule is **always taken from the last column**, so additional middle columns are ignored automatically.

**Score columns must be integers.** Rows that cannot be parsed are silently skipped.

---

## Usage

### Basic Examples

```bash
# Analyse a single file and write a summary
python ranker_handler.py -i ranking.csv

# Analyse two files and save summary + clean rules
python ranker_handler.py -i ranking1.csv ranking2.csv \
    -o analysis_summary.txt \
    -r top_rules.rule

# Extract the top 1 000 rules into a Hashcat rule file
python ranker_handler.py -i ranking.csv -t 1000 -r top_1000.rule

# Full pipeline: summary + rules + scored debug file, top 5 000
python ranker_handler.py \
    -i results1.csv results2.csv \
    -o full_analysis.txt \
    -r best_rules.rule \
    -s scored_debug.txt \
    -t 5000

# Rules-only output (skip analysis summary), with parallel loading
python ranker_handler.py -i ranking.csv -r hashcat_rules.rule \
    -t 10000 --rules-only --parallel
```

### All Arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `--input FILE [FILE …]` | `-i` | *(required)* | One or more input ranking CSV/TXT files |
| `--output FILE` | `-o` | `rule_analysis_summary.txt` | Destination for the text analysis summary |
| `--rules-output FILE` | `-r` | — | Write a clean Hashcat `.rule` file |
| `--scores-output FILE` | `-s` | — | Write rules with scores (debug format) |
| `--top N` | `-t` | all | Extract only the top *N* rules by combined score |
| `--console-top N` | — | `20` | How many top rules to print to the terminal |
| `--no-console` | — | off | Suppress all console result output |
| `--parallel` | — | off | Load multiple input files in parallel |
| `--chunk-size N` | — | `10000` | Internal read chunk size |
| `--no-progress` | — | off | Disable `tqdm` progress bars |
| `--rules-only` | — | off | Skip writing the analysis summary file |

---

## Output Files

### Analysis Summary (`-o`)

Plain-text report containing:

- **Overall statistics** — total rules, unique rules, rules appearing in multiple files, processing time.
- **Aggregate scores** — total and average combined / effectiveness / uniqueness scores.
- **Cross-file duplicates** — rules that appear in more than one input file, sorted by occurrence count.
- **Top rules table** — rank, combined score, and rule text for the top-N rules.
- **Detailed section** — per-rule breakdown (up to 100 rules) with original rule text, source file, and all three scores.

### Clean Hashcat Rules (`-r`)

One rule per line, ready for use with `hashcat -r`.

- Begins with a `:` (empty/identity rule) as per Hashcat convention.
- Fully deduplicated — each unique cleaned rule appears exactly once.
- Bracket notation (e.g. `[rule]`) is stripped automatically.

### Scored Rules (`-s`)

Debug/reference format:

```
# Format: Combined_Score:Effectiveness_Score:Uniqueness_Score:Rule
0:0:0::
1234:800:434:rule_string
…
```

---

## Performance Notes

| Technique | Benefit |
|---|---|
| `heapq` top-N extraction | Avoids a full O(n log n) sort when only the top *k* rules are needed |
| `@lru_cache` on `clean_rule_cached` | Eliminates redundant bracket-stripping for repeated rule strings (cache size: 100 000 entries) |
| Buffered writes (10 000-line chunks) | Reduces syscall overhead on large output files |
| `ProcessPoolExecutor` (`--parallel`) | Saturates multiple CPU cores when loading many input files simultaneously |
| List-based storage | Appending to parallel lists is faster than building a list of dicts for millions of entries |

Typical throughput is in the range of **hundreds of thousands to millions of rules per second** depending on hardware and rule complexity.

---

## How It Works

```
Input CSV files
      │
      ▼
parse_ranking_file_fast()   ← detects format, skips header, maps columns
      │
      ▼
analyze_rules_fast()        ← cleans rules (LRU cached), tracks cross-file
      │                        occurrences, extracts top-N via heap
      ▼
 ┌────┴──────────────────────────────────┐
 │                                       │
save_summary_fast()          save_clean_rules_fast()
(-o)                         (-r)
                             save_clean_rules_with_scores_fast()
                             (-s)
      │
      ▼
print_summary_to_console_fast()   ← ANSI-coloured terminal output
```

---

## License

MIT
