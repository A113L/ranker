**RANKER v4.0**

# GPU-Accelerated Hashcat Rule Ranker

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCL](https://img.shields.io/badge/OpenCL-GPU%20Accelerated-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A high-performance GPU-accelerated tool for ranking and optimizing Hashcat rules based on uniqueness and effectiveness scores.

## ✨ Features

- **🚀 GPU Acceleration**: Uses OpenCL for massively parallel rule processing
- **📊 Dual Scoring**: Calculates both uniqueness and effectiveness scores for rules
- **🎯 Smart Presets**: Auto-configures based on GPU memory and dataset size

[![mab.jpg](https://i.postimg.cc/TwWGZ2ZR/mab.jpg)](https://postimg.cc/BLsdF3Sy)

## 📋 Requirements and Usage

```bash
pip install pyopencl numpy tqdm

📝 Usage
bash
python3 ranker.py --help
usage: ranker.py [-h] [-w WORDLIST] [-r RULES] [-c CRACKED] [-o OUTPUT] [-k TOPK] [--batch-size BATCH_SIZE] [--global-bits GLOBAL_BITS]
                     [--cracked-bits CRACKED_BITS] [--mab-exploration MAB_EXPLORATION] [--mab-min-trials MAB_MIN_TRIALS] [--preset PRESET]
                     [--device DEVICE] [--list-devices]

GPU-Accelerated Hashcat Rule Ranking Tool with Multi-Armed Bandits

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
                        Multi-Armed Bandit exploration factor (default: 1.0)
  --mab-min-trials MAB_MIN_TRIALS
                        MAB minimum trials before pruning (default: 100)
  --preset PRESET       Use preset configuration: "low_memory", "medium_memory", "high_memory", "recommend" (auto-selects best)
  --device DEVICE       OpenCL device ID
  --list-devices        List all available OpenCL devices and exit

```
The Combined Score is a weighted heuristic used to rank the strategic value of Hashcat rules.

*Effectiveness:* The total number of passwords a rule successfully cracks.

*Uniqueness:* The number of passwords cracked only by that specific rule.

By applying a 10x multiplier to Effectiveness, the formula prioritizes "high-yield" rules that deliver bulk results. The addition of Uniqueness ensures that "specialist" rules, which uncover rare patterns others miss, still gain a competitive rank. This balance creates an optimized rule set that maximizes total cracks while retaining the ability to hit complex, non-standard passwords.

**Key Mechanisms**

*Statistical Selection:* Each rule is modeled using a Beta Distribution ($\alpha$ for hits, $\beta$ for misses). Before each GPU batch, the script draws a random sample for every rule. High-performing rules yield higher samples, naturally prioritizing them for the next processing cycle.

*Hybrid Exploration:* To prevent "cold start" neglect, a UCB-inspired exploration bonus is applied to less-tested rules. This ensures every transformation is evaluated fairly. An Exploration Decay mechanism gradually reduces this bonus, shifting the strategy from discovery to pure exploitation of known "winners."

*Probabilistic Pruning:* The script implements "survival of the fittest" through automated pruning. Rules consistently performing below a 0.01% effectiveness threshold face a 10% chance of removal per batch. This purges "junk" rules, maximizing VRAM and GPU throughput for high-yield candidates.

Scoring:

- Uniqueness: Counts how many unique new words each rule generates
- Effectiveness: Counts how many cracked passwords each rule produces
- Ranking: Combines scores and ranks rules by effectiveness
- Output: Saves ranked list and optionally top-K optimized rules

📊 **Output Files**

- results.csv: Full ranking data with scores for all rules
- results_optimized.rule: Optimized rule file (if -k specified)
- results_INTERRUPTED.csv: Intermediate results if processing is interrupted

📄 **Licence**

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 **Credits**

- Hashcat community for rule sets and inspiration
- PyOpenCL developers for GPU bindings
- Cybersecurity researchers worldwide
- 0xVavaldi for inspiration - https://github.com/0xVavaldi

**Website**

https://hcrt.pages.dev/ranker.static_workflow
