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

[![mermaid-20251213-642c5b.png](https://i.postimg.cc/qBssNdW3/mermaid-20251213-642c5b.png)](https://postimg.cc/w3Byrnqq)

**Combine Score = Effectiveness*10+Uniqueness**

## 📋 Requirements

```bash
pip install pyopencl numpy tqdm

📝 Usage
bash
python3 ranker.py --help
usage: ranker.py [-h] [-w WORDLIST] [-r RULES] [-c CRACKED] [-o OUTPUT] [-k TOPK] [--batch-size BATCH_SIZE] [--global-bits GLOBAL_BITS]
                 [--cracked-bits CRACKED_BITS] [--preset PRESET] [--list-platforms] [--platform PLATFORM] [--device DEVICE]

GPU-Accelerated Hashcat Rule Ranking Tool (Ranker v3.2 - Optimized Large File Loading)

options:
  -h, --help            show this help message and exit

Main Arguments:
  -w WORDLIST, --wordlist WORDLIST
                        Path to the base wordlist file.
  -r RULES, --rules RULES
                        Path to the Hashcat rules file to rank.
  -c CRACKED, --cracked CRACKED
                        Path to a list of cracked passwords for effectiveness scoring.
  -o OUTPUT, --output OUTPUT
                        Path to save the final ranking CSV.
  -k TOPK, --topk TOPK  Number of top rules to save to an optimized .rule file. Set to 0 to skip.

Performance Tuning:
  --batch-size BATCH_SIZE
                        Number of words to process in each GPU batch (default: auto-calculate based on VRAM)
  --global-bits GLOBAL_BITS
                        Bits for global hash map size (default: auto-calculate based on VRAM)
  --cracked-bits CRACKED_BITS
                        Bits for cracked hash map size (default: auto-calculate based on VRAM)
  --preset PRESET       Use preset configuration: "low_memory", "medium_memory", "high_memory", "recommend" (auto-selects best)

Platform/Device Selection:
  --list-platforms      List all available OpenCL platforms and devices, then exit.
  --platform PLATFORM   Select specific OpenCL platform index (use --list-platforms to see available options).
  --device DEVICE       Select specific device index within platform (use --list-platforms to see available options).

Examples:
  List available platforms and devices:
    python3 ranker.py --list-platforms
  
  Run with auto-selected platform/device:
    python3 ranker.py -w wordlist.txt -r rules.txt -c cracked.txt -o output.csv -k 1000
  
  Run with specific platform and device:
    python3 ranker.py -w wordlist.txt -r rules.txt -c cracked.txt --platform 0 --device 0
  
  Use recommended preset:
    python3 ranker.py -w wordlist.txt -r rules.txt -c cracked.txt --preset recommend
  
  Use low memory preset:
    python3 ranker.py -w wordlist.txt -r rules.txt -c cracked.txt --preset low_memory


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
