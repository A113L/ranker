**RANKER v3.2**

# GPU-Accelerated Hashcat Rule Ranker

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCL](https://img.shields.io/badge/OpenCL-GPU%20Accelerated-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A high-performance GPU-accelerated tool for ranking and optimizing Hashcat rules based on uniqueness and effectiveness scores.

## ✨ Features

- **🚀 GPU Acceleration**: Uses OpenCL for massively parallel rule processing
- **📊 Dual Scoring**: Calculates both uniqueness and effectiveness scores for rules
- **💾 Memory Efficient**: Optimized memory-mapped file loading for large datasets
- **⏱️ Continuous Processing**: Processes entire wordlists without interruption
- **🛡️ Interrupt Protection**: Saves progress on Ctrl+C for resumable processing
- **🎯 Smart Presets**: Auto-configures based on GPU memory and dataset size

[![mermaid-20251204-d36b4d.png](https://i.postimg.cc/Z53wqDm5/mermaid-20251204-d36b4d.png)](https://postimg.cc/jDxHgcCp)

**Combine Score = Effectiveness*10+Uniqueness**

## 📋 Requirements

```bash
pip install pyopencl numpy tqdm
🚀 Quick Start
bash
# Basic usage
python ranker_v3.2.py -w wordlist.txt -r rules.txt -c cracked.txt -o results.csv

# Save top 1000 optimized rules
python ranker_v3.2.py -w wordlist.txt -r rules.txt -c cracked.txt -o results.csv -k 1000

# Use high memory preset for large datasets
python ranker_v3.2.py -w wordlist.txt -r rules.txt -c cracked.txt --preset high_memory
📝 Usage
bash
python ranker_v3.2.py [-h] -w WORDLIST -r RULES -c CRACKED 
                     [-o OUTPUT] [-k TOPK] [--batch-size BATCH_SIZE]
                     [--global-bits GLOBAL_BITS] [--cracked-bits CRACKED_BITS]
                     [--preset {low_memory,medium_memory,high_memory,recommend}]
```
🎯 **How It Works**

- Loading: Memory-mapped loading of wordlist and rules
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
