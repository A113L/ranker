**RANKER v3.2**

# GPU-Accelerated Hashcat Rule Ranker

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCL](https://img.shields.io/badge/OpenCL-GPU%20Accelerated-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A high-performance GPU-accelerated tool for ranking and optimizing Hashcat rules based on uniqueness and effectiveness scores.

## ✨ Features

- **🚀 GPU Acceleration**: Uses OpenCL for massively parallel rule processing
- **📊 Dual Scoring**: Calculates both uniqueness and effectiveness scores for rules
- **🎯 Smart Presets**: Auto-configures based on GPU memory and dataset size

[![mermaid-20251204-d36b4d.png](https://i.postimg.cc/Z53wqDm5/mermaid-20251204-d36b4d.png)](https://postimg.cc/jDxHgcCp)

**Combine Score = Effectiveness*10+Uniqueness**

## 📋 Requirements

```bash
pip install pyopencl numpy tqdm

📝 Usage
bash
python3 rankerv3.py -h
usage: rankerv3.py [-h] -w WORDLIST -r RULES -c CRACKED [-o OUTPUT] [-k TOPK] [--batch-size BATCH_SIZE] [--global-bits GLOBAL_BITS] [--cracked-bits CRACKED_BITS]
                   [--preset {low_memory,medium_memory,high_memory,aggressive,balanced}] [--target-device TARGET_DEVICE] [--force-nvidia] [--force-amd] [--skip-rule-encoding]
                   [--disable-double-buffering] [--disable-progress-bars] [--benchmark-mode] [--list-presets]

Ultra-optimized GPU rule ranking

options:
  -h, --help            show this help message and exit
  -w WORDLIST, --wordlist WORDLIST
                        Path to the base wordlist file
  -r RULES, --rules RULES
                        Path to the Hashcat rules file to rank
  -c CRACKED, --cracked CRACKED
                        Path to a list of cracked passwords for effectiveness scoring
  -o OUTPUT, --output OUTPUT
                        Path to save the final ranking CSV (default: ranker_output.csv)
  -k TOPK, --topk TOPK  Number of top rules to save to optimized .rule file (default: 1000)
  --batch-size BATCH_SIZE
                        Number of words to process in each GPU batch (default: auto-calculate)
  --global-bits GLOBAL_BITS
                        Bits for global hash map size (default: auto-calculate)
  --cracked-bits CRACKED_BITS
                        Bits for cracked hash map size (default: auto-calculate)
  --preset {low_memory,medium_memory,high_memory,aggressive,balanced}
                        Use preset configuration for easy setup
  --target-device TARGET_DEVICE
                        Target specific GPU device (e.g., "RTX 3060 Ti")
  --force-nvidia        Force use of NVIDIA platform
  --force-amd           Force use of AMD platform
  --skip-rule-encoding  Skip parallel rule encoding optimization
  --disable-double-buffering
                        Disable double buffering for word loading
  --disable-progress-bars
                        Disable all progress bars for cleaner output
  --benchmark-mode      Enable detailed performance benchmarking
  --list-presets        List available performance presets and exit

Examples:
  rankerv3.py -w wordlist.txt -r rules.rule -c cracked.txt -o output.csv
  rankerv3.py -w wordlist.txt -r rules.rule -c cracked.txt -o output.csv -k 5000
  rankerv3.py -w wordlist.txt -r rules.rule -c cracked.txt -o output.csv --preset aggressive
  rankerv3.py -w wordlist.txt -r rules.rule -c cracked.txt -o output.csv --batch-size 200000 --global-bits 34

```
🎯 **How It Works**

- Processing: GPU processes each rule against each word

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
