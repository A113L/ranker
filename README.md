**RANKER v3.2**

- GPU-Accelerated Hashcat Rule Ranking Tool with Memory-Mapped File Loading

- Historic Performance Achievement: 21x Speed Boost - 8,000 → 170,000 words/second on RTX 3060 Ti

```
📁 File size: 0.52 GB
Loading wordlist: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 554M/554M [06:03<00:00, 1.52MB/s]
Processing wordlist from disk [Unique: 389,904,114 | Cracked: 118,724,863]:  92%|████████████████████████▉  | 57200000/62041432 [06:03<00:28, 170930.55 word/s]
✅ Optimized loading completed: 57,242,219 words in 363.98s (157,267 words/sec)████████████████████████████████████████████▉| 554M/554M [06:03<00:00, 3.09MB/s]
✅ Wordlist fully loaded. Continuing with remaining rule batches...
Processing wordlist from disk [Unique: 389,904,114 | Cracked: 118,724,863]:  92%|████████████████████████▉  | 57242219/62041432 [06:03<00:30, 157267.00 word/s]
Rule batches processed: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1241/1241 [06:03<00:00,  3.41 batch/s]
```
🎯 **What is RANKER?**

RANKER is a high-performance GPU-accelerated tool that ranks Hashcat rules based on their effectiveness and uniqueness against your target wordlists and cracked passwords.

✨ **Key Features**
- 🚀 Blazing Fast: 170,000+ words/second on RTX 3060 Ti (21x improvement)

- 💾 Memory-Mapped Loading: 10-50x faster file processing with mmap

- 🎮 GPU Acceleration: OpenCL-powered parallel processing

- 🧠 Smart VRAM Management: Automatic optimization for any GPU

- 📊 Comprehensive Rule Support: 100,000+ Hashcat rules

- ⚡ Bulk Processing: 200,000+ words per GPU batch

- 🔧 Auto-Tuning: Dynamic parameter optimization

🛠 **Installation**

*Requirements*

- Python 3.8+

- PyOpenCL

- NumPy

- NVIDIA/AMD/Intel GPU with OpenCL support

- 4GB+ VRAM recommended

**Quick Install**

```
pip install pyopencl numpy tqdm
git clone https://github.com/A1131/ranker.git
cd ranker
```

🚀 **Usage**

*Basic Usage*

```
python ranker.py -w wordlist.txt -r rules.txt -c cracked.txt -o output.csv -k 10000
```

*Advanced Usage with Auto-Optimization*

```
python ranker.py \
  -w rockyou.txt \
  -r best64.rule \
  -c cracked_passwords.txt \
  -o rule_ranking.csv \
  -k 5000 \
  --preset auto
```

**Command Line Arguments**

```
-w, --wordlist	Base wordlist path	Required
-r, --rules	Hashcat rules file	Required
-c, --cracked	Cracked passwords file	Required
-o, --output	Output CSV path	ranker_output.csv
-k, --topk	Top K rules to save	1000
--preset	Performance preset	auto
```

Performance Presets

RANKER automatically detects your GPU and optimizes parameters:

```
--preset auto: (Recommended) Auto-calculated based on VRAM
--preset low_memory: For GPUs with < 4GB VRAM
--preset medium_memory: For GPUs with 4-8GB VRAM
--preset high_memory: For GPUs with > 8GB VRAM
```

📊 **Output Files**

- output.csv: Detailed ranking with scores and metadata
- output_optimized.rule: Top K optimized rules for Hashcat

**Output Columns**

- Rank: Rule ranking position
- Combined_Score: (10 × Effectiveness) + Uniqueness
- Effectiveness_Score: Matches in cracked list
- Uniqueness_Score: New words not in base wordlist
- Rule_Data: Original Hashcat rule

🔧 **Technical Innovations**

*Memory-Mapped File Loading*

```
with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
    # Direct file access - no Python I/O overhead
```

*GPU Optimization*

- 256-thread work groups aligned with GPU architecture
- Double buffering for zero GPU idle time
- Bulk processing (200k words/batch)

**Smart Memory Management**

- Automatic VRAM detection and optimization
- 15% safety margin for stability
- Retry logic for memory allocation

**Historic Achievement**

First tool to achieve 170,000 words/second on RTX 3060 Ti - previously required high-end workstation GPUs. This represents a 21x performance improvement through software optimization alone.

📈 **Real-World Impact**

```
# Before: Multi-day process
100 million words × 100,000 rules = "Weekend project"

# After: Interactive tool  
100 million words × 100,000 rules = "Lunch break task"
```

🐛 **Troubleshooting**

Common Issues

*OpenCL not found:*

```
# Install GPU drivers
sudo apt update && sudo apt install ocl-icd-libopencl1
```

*Memory allocation errors:*

```
# Use low memory preset
python ranker.py --preset low_memory -w wordlist.txt -r rules.txt -c cracked.txt
```

*Slow file loading:*

- Ensure files are on SSD storage
- Use binary file formats when possible

📄 **Licence**

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 **Credits**

- Hashcat community for rule sets and inspiration
- PyOpenCL developers for GPU bindings
- Cybersecurity researchers worldwide
- 0xVavaldi for inspiration - https://github.com/0xVavaldi

⭐ Star this repo if you find it useful!

*RANKER v3.2 - Democratizing high-performance rule analysis for security professionals worldwide.* 🚀

**Website**

https://hcrt.pages.dev/ranker.static_workflow
