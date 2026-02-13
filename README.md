**RANKER v4.0**

# GPU-Accelerated Hashcat Rule Ranker

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCL](https://img.shields.io/badge/OpenCL-GPU%20Accelerated-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A high-performance GPU-accelerated tool for ranking and optimizing Hashcat rules based on uniqueness and effectiveness scores.

## ✨ Features

- **GPU Acceleration**: Uses OpenCL for massively parallel rule processing
- **Dual Scoring**: Calculates both uniqueness and effectiveness scores for rules
- **Smart Presets**: Auto-configures based on GPU memory and dataset size

[![mab.jpg](https://i.postimg.cc/TwWGZ2ZR/mab.jpg)](https://postimg.cc/BLsdF3Sy)

## 📋 Requirements and Usage

```bash
pip install pyopencl numpy tqdm

📝 Usage
bash
python3 ranker.py -h
usage: ranket.py [-h] [-w WORDLIST] [-r RULES] [-c CRACKED] [-o OUTPUT] [-k TOPK] [--batch-size BATCH_SIZE] [--global-bits GLOBAL_BITS]
              [--cracked-bits CRACKED_BITS] [--mab-exploration MAB_EXPLORATION] [--mab-final-trials MAB_FINAL_TRIALS]
              [--mab-screening-trials MAB_SCREENING_TRIALS] [--mab-no-zero-eliminate] [--preset PRESET] [--device DEVICE] [--list-devices]

GPU-Accelerated Hashcat Rule Ranking Tool with Multi-Pass MAB and Early Elimination

optional arguments:
  -h, --help            show this help message and exit
  -w WORDLIST, --wordlist WORDLIST
                        Path to the base wordlist file
  -r RULES, --rules RULES
                        Path to the Hashcat rules file to rank
  -c CRACKED, --cracked CRACKED
                        Path to a list of cracked passwords for effectiveness scoring
  -o OUTPUT, --output OUTPUT
                        Path to save the ranking CSV
  -k TOPK, --topk TOPK  Number of top rules to save (0 to skip)
  --batch-size BATCH_SIZE
                        Words per GPU batch (auto-calculated if not specified)
  --global-bits GLOBAL_BITS
                        Bits for global hash map (auto-calculated)
  --cracked-bits CRACKED_BITS
                        Bits for cracked hash map (auto-calculated)
  --mab-exploration MAB_EXPLORATION
                        Multi-Armed Bandit exploration factor (default: 2.0)
  --mab-final-trials MAB_FINAL_TRIALS
                        MAB final trials for deep testing (default: 50)
  --mab-screening-trials MAB_SCREENING_TRIALS
                        MAB screening trials - eliminate low performers after N trials (default: 5)
  --mab-no-zero-eliminate
                        DISABLE zero-success elimination (not recommended)
  --preset PRESET       Use preset configuration: "low_memory", "medium_memory", "high_memory", "recommend"
  --device DEVICE       OpenCL device ID
  --list-devices        List all available OpenCL devices and exit



```
**Core Architecture**

A GPU-accelerated Multi-Armed Bandit (MAB) system that intelligently ranks Hashcat rules using Thompson Sampling. Built for massive scale - handles 100,000+ rules efficiently.

**Two-Phase Processing**

- *Screening Phase*: Every rule receives exactly 5 trials (configurable) (750,000+ words each). Zero-success rules are immediately eliminated. Typically removes 80-90% of useless rules.

- *Deep Testing Phase*: Survivors receive 45 additional trials (50 total) (configurable) . Statistical elimination based on success rate thresholds (0.00001% → 0.001% → 0.01%).

**Key Technical Features**

- OpenCL kernel executes 150,000 words/sec per rule batch (tested on RTX 3060Ti GB)
- FNV-1a hash tables for global uniqueness & cracked hash detection
- Memory-mapped I/O for multi-MB wordlists
- Real-time progress tracking with interrupt
- Comprehensive CSV output with ranking, success probabilities, and elimination reasons

**Performance**

Cuts ranking time from weeks to hours by focusing compute power on promising rules while ruthlessly pruning the rest.

📄 **Licence**

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 **Credits**

- Hashcat community for rule sets and inspiration
- PyOpenCL developers for GPU bindings
- Cybersecurity researchers worldwide
- 0xVavaldi for inspiration - https://github.com/0xVavaldi
