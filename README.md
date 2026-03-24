# 🔐 Hashcat Rule Ranker — GPU-Accelerated Ranking Tool

> GPU-accelerated Hashcat rule ranking using OpenCL. Scores rules by **uniqueness** (generates unseen words) and **effectiveness** (matches known cracked passwords). Built for large rule sets and large wordlists.

---

## 📁 Versions

| File | Version | Algorithm |
|------|---------|-----------|
| `ranker_v3.2.py` | **v3.2** — Optimized Large File Loading | Exhaustive brute-force |
| `ranker_v4_0.py` | **v4.0** — Multi-Pass MAB with Early Rules Elimination | Thompson Sampling + 2-phase MAB |

---

## 🆕 What Changed: v3.2 → v4.0

### Core Algorithm

| | v3.2 | v4.0 |
|---|---|---|
| **Strategy** | Tests **every rule** against **every word batch** | Multi-Armed Bandit — selects rules probabilistically |
| **Rule elimination** | None — all rules run to completion | Eliminates 80–90% of low-performing rules early |
| **Phases** | Single pass | Phase 1: Screening → Phase 2: Deep Testing |
| **Designed for** | General use | 100K – 2M+ rule sets |

### New Class: `MultiPassMAB`

v4.0 introduces a full `MultiPassMAB` class implementing Thompson Sampling with early elimination:

```
MultiPassMAB
├── Thompson Sampling  — Beta distribution per rule (α successes, β failures)
├── Screening Phase    — Eliminate rules after N trials (default: 5)
├── Deep Testing       — Test survivors to final_trials (default: 50)
└── 4 elimination strategies:
    ├── Zero-success after screening_trials
    ├── Phase-based success rate thresholds (0.00001% → 0.01%)
    ├── 1000× worse than top-100 average
    └── 3 consecutive zero-success batches
```

### OpenCL Kernel Differences

| | v3.2 | v4.0 |
|---|---|---|
| **Kernel name** | `bfs_kernel` | `ranker_kernel` |
| **Rule encoding** | Packed into 2× `uint32` (max ~8 chars usable) | Full `uint8[255]` array — all rule lengths supported |
| **`MAX_RULE_LEN`** | `16` | `255` |
| **Global hash map** | Read-only during kernel | **Read-write** — atomic `OR` writes uniqueness data back |
| **Constants in kernel** | String-interpolated masks | `#define` macros via f-string |

### Output Files

| File | v3.2 | v4.0 |
|---|---|---|
| `*_output.csv` | Rank, Combined/Effectiveness/Uniqueness Score, Rule_Data | + MAB fields: `MAB_Success_Prob`, `Times_Tested`, `MAB_Trials`, `Selections`, `Total_Successes`, `Total_Trials`, `Eliminated`, `Eliminate_Reason` |
| `*_optimized.rule` | Top K rules by combined score | Top K **active (non-eliminated)** rules only |
| `*_elimination_stats.csv` | ❌ | ✅ Per-iteration elimination tracking |

### CLI Arguments

**Shared arguments:**

```
-w / --wordlist       Path to base wordlist
-r / --rules          Path to Hashcat rules file
-c / --cracked        Path to cracked passwords list
-o / --output         Output CSV path (default: ranker_output.csv)
-k / --topk           Top K rules to save (default: 1000)
--batch-size          Words per GPU batch (auto if omitted)
--global-bits         Global hash map bits
--cracked-bits        Cracked hash map bits
--preset              low_memory | medium_memory | high_memory | recommend
```

**v3.2 only:**

```
--list-platforms      List all OpenCL platforms and devices
--platform INT        Select OpenCL platform index
--device INT          Select device index within platform
```

**v4.0 only:**

```
--device INT                    Single global device ID (flat index)
--list-devices                  List all OpenCL devices (flat list)
--mab-exploration FLOAT         Exploration factor (default: 2.0)
--mab-final-trials INT          Trials for deep testing phase (default: 50)
--mab-screening-trials INT      Trials before elimination decision (default: 5)
--mab-no-zero-eliminate         Disable zero-success rule elimination
```

### Device Selection

| | v3.2 | v4.0 |
|---|---|---|
| **Selection model** | Platform index + device index within platform | Flat device ID across all platforms |
| **Auto-select GPU** | Prefers NVIDIA → AMD → Intel → first | Largest VRAM GPU wins |
| **List command** | `--list-platforms` | `--list-devices` |

---

## 🚀 Quick Start

### Requirements

```bash
pip install pyopencl numpy tqdm
```

OpenCL runtime required for your GPU (CUDA Toolkit / ROCm / Intel OpenCL).

### v3.2 — Exhaustive Ranking

Best for small-to-medium rule sets where you want every rule scored.

```bash
# Basic run
python3 ranker.py \
  -w wordlist.txt \
  -r rules.rule \
  -c cracked.txt \
  -o ranked_output.csv \
  -k 1000

# List available OpenCL platforms/devices
python3 ranker.py --list-platforms

# Select specific GPU (platform 0, device 1)
python3 ranker.py -w wordlist.txt -r rules.rule -c cracked.txt \
  --platform 0 --device 1

# Use memory preset
python3 ranker.py -w wordlist.txt -r rules.rule -c cracked.txt \
  --preset high_memory
```

### v4.0 — MAB with Early Elimination

Best for very large rule sets (100K+). Eliminates weak rules early, concentrates compute on strong ones.

```bash
# Basic run
python3 ranker_v4_0.py \
  -w wordlist.txt \
  -r rules.rule \
  -c cracked.txt \
  -o ranked_output.csv \
  -k 1000

# List available OpenCL devices (flat index)
python3 ranker_v4_0.py --list-devices

# Select GPU by device ID
python3 ranker_v4_0.py -w wordlist.txt -r rules.rule -c cracked.txt \
  --device 0

# Tune MAB parameters
python3 ranker_v4_0.py -w wordlist.txt -r rules.rule -c cracked.txt \
  --mab-screening-trials 10 \
  --mab-final-trials 100 \
  --mab-exploration 3.0

# Disable zero-success elimination (not recommended for large sets)
python3 ranker_v4_0.py -w wordlist.txt -r rules.rule -c cracked.txt \
  --mab-no-zero-eliminate
```

---

## ⚙️ Performance Tuning

### Memory Presets

| Preset | Batch Size | Global Map | Cracked Map | Target GPU |
|--------|-----------|------------|-------------|------------|
| `low_memory` | 25,000 | 30 bits | 28 bits | < 4 GB VRAM |
| `medium_memory` | 75,000 | 33 bits | 31 bits | 4–8 GB VRAM |
| `high_memory` | 150,000 | 35 bits | 33 bits | > 8 GB VRAM |
| `recommend` | auto | auto | auto | auto-detected |

```bash
python3 ranker.py      --preset recommend ...
python3 ranker_v4_0.py --preset recommend ...
```

### Manual Tuning

```bash
python3 ranker.py \
  --batch-size 100000 \
  --global-bits 35 \
  --cracked-bits 33 \
  ...
```

---

## 🔢 Scoring

### v3.2 Scoring Formula

```
combined_score = effectiveness_score × 10 + uniqueness_score
```

- **uniqueness_score** — how many transformed words are NOT in the base wordlist
- **effectiveness_score** — how many transformed words ARE in the cracked list

### v4.0 Scoring Formula

```
combined_score = effectiveness_score × 10 + uniqueness_score + mab_success_prob × 1000
```

Additional MAB-derived fields per rule in the output CSV:

| Field | Description |
|-------|-------------|
| `MAB_Success_Prob` | `total_successes / total_trials` — empirical hit rate |
| `Times_Tested` | Number of batches this rule was selected for |
| `MAB_Trials` | Number of MAB selection rounds |
| `Selections` | Total times selected by the bandit |
| `Total_Successes` | Cumulative cracked-hash matches |
| `Total_Trials` | Cumulative words tested against this rule |
| `Eliminated` | `True` if discarded during screening |
| `Eliminate_Reason` | `zero_success` / `low_success_rate` / `below_threshold` |

---

## 🏗️ Architecture

### Shared Components (both versions)

```
┌─────────────────────────────────────────────────────────┐
│                     RANKER PIPELINE                      │
├──────────────────┬──────────────────────────────────────┤
│  File Loading    │  Memory-mapped I/O (mmap)             │
│                  │  FNV-1a hash pre-computation          │
│                  │  Fast word-count estimation           │
├──────────────────┼──────────────────────────────────────┤
│  Hash Maps       │  Global bitmap  — base wordlist       │
│                  │  Cracked bitmap — cracked passwords   │
│                  │  Bloom-filter style (bit-level OR)    │
├──────────────────┼──────────────────────────────────────┤
│  OpenCL Kernel   │  Hashcat rule application (GPU)      │
│                  │  FNV-1a hash per output word         │
│                  │  Dual uniqueness/effectiveness count  │
├──────────────────┼──────────────────────────────────────┤
│  Interrupt       │  SIGINT handler saves progress       │
│  Recovery        │  Writes *_INTERRUPTED.csv / .rule   │
└──────────────────┴──────────────────────────────────────┘
```

### v3.2 Processing Loop

```
for each word_batch:
    upload to GPU
    update global hash map
    for each rule_batch (1024 rules at a time):
        run bfs_kernel
        accumulate scores
```

Time complexity: `O(words × rules)`

### v4.0 Processing Loop

```
Phase 1 — SCREENING:
    repeat until all rules have ≥ screening_trials selections:
        select_rules() via Thompson Sampling
        for each word_batch:
            run ranker_kernel
            update MAB (successes/failures)
            eliminate_low_performers()

Phase 2 — DEEP TESTING:
    repeat until survivors have ≥ final_trials selections:
        same loop, only active (non-eliminated) rules selected
```

Time complexity: `O(words × active_rules × final_trials)` — active_rules << total_rules

---

## 📊 Supported Hashcat Rule Operations

Both versions implement the same rule set:

| Category | Rules |
|----------|-------|
| Case | `l` `u` `c` `C` `t` `Tn` `E` |
| Reverse / Rotate | `r` `{` `}` |
| Duplicate | `d` `f` `p` `q` `z` `Z` |
| Delete | `Dn` `Ln` `Rn` `[n` `]n` `@X` |
| Insert / Overwrite | `^X` `$X` `in X` `on X` |
| Substitute | `sXY` |
| Extract / Swap | `xn m` `*n m` `kK` |
| ASCII modify | `+n` `-n` |
| Reject | `!X` `/X` |

---

## 📂 Output Files

### v3.2

| File | Contents |
|------|----------|
| `<output>.csv` | All rules ranked by combined score |
| `<output>_optimized.rule` | Top K rules, ready for Hashcat |
| `<output>_INTERRUPTED.csv` | Saved on Ctrl+C |
| `<output>_INTERRUPTED.rule` | Saved on Ctrl+C |

### v4.0

| File | Contents |
|------|----------|
| `<output>.csv` | All rules with MAB metadata + eliminated flag |
| `<output>_optimized.rule` | Top K **active** rules only |
| `<output>_elimination_stats.csv` | Per-iteration elimination history |
| `<output>_INTERRUPTED.csv` | Saved on Ctrl+C |
| `<output>_INTERRUPTED.rule` | Saved on Ctrl+C |

---

## 🛠️ When to Use Which Version

| Scenario | Recommended |
|----------|-------------|
| Rule file < 50K rules | **v3.2** — exhaustive is fast enough, all rules scored equally |
| Rule file 50K–500K rules | **v4.0** — MAB avoids testing irrelevant rules |
| Rule file 500K–2M+ rules | **v4.0** — essential; exhaustive would take impractical time |
| Need every rule scored | **v3.2** — v4.0 skips eliminated rules |
| Cracked list is small / absent | **v3.2** — effectiveness scoring less meaningful for MAB |
| Rapid iteration / prototyping | **v4.0** with low `--mab-screening-trials` |

---

## 🐛 Common Issues

**`MEM_OBJECT_ALLOCATION_FAILURE`** — GPU OOM.
Use `--preset low_memory` or manually lower `--batch-size` and `--global-bits`.

**`No OpenCL platforms found`** — Runtime not installed.
Install CUDA Toolkit (NVIDIA), ROCm (AMD), or Intel OpenCL runtime.

**Cracked list not found** — Effectiveness scores will all be zero.
Both versions continue and score uniqueness only.

**v4.0: "No rules with sufficient trials"** — Run longer or lower `--mab-screening-trials`.
This appears when no rule has reached `screening_trials` yet.

---

## 📦 Dependencies

```
pyopencl    >= 2023.1
numpy       >= 1.24
tqdm        >= 4.65
```

---

## 📄 License

See `LICENSE` for details.


🙏 **Credits**

- Hashcat community for rule sets and inspiration
- PyOpenCL developers for GPU bindings
- Cybersecurity researchers worldwide
- 0xVavaldi for inspiration - https://github.com/0xVavaldi
