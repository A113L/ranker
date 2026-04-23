#!/usr/bin/env python3
"""
Ranker v5.0 – GPU-Accelerated Hashcat Rule Ranking
===================================================
Multi-Pass MAB with Early Elimination for large rule sets.
Optionally runs in legacy exhaustive mode (v3.2).
All GPU‑compatible Hashcat rules are implemented.
MAX_RULE_LEN = 255, comprehensive rule application.
"""

import pyopencl as cl
import numpy as np
import argparse
import csv
from tqdm import tqdm
import math
import warnings
import os
from time import time
import mmap
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

# ====================================================================
# --- CONSTANTS ---
# ====================================================================
MAX_WORD_LEN = 256                # Maximum length of a base word
MAX_OUTPUT_LEN = 512              # Maximum length of a transformed word
MAX_RULE_LEN = 255                # Increased to support any Hashcat rule
MAX_RULES_IN_BATCH = 1024
LOCAL_WORK_SIZE = 256

# Default values (will be adjusted based on VRAM)
DEFAULT_WORDS_PER_GPU_BATCH = 150000
DEFAULT_GLOBAL_HASH_MAP_BITS = 35
DEFAULT_CRACKED_HASH_MAP_BITS = 33

# VRAM usage thresholds
VRAM_SAFETY_MARGIN = 0.15
MIN_BATCH_SIZE = 25000
MIN_HASH_MAP_BITS = 28

# Memory reduction factors
MEMORY_REDUCTION_FACTOR = 0.7
MAX_ALLOCATION_RETRIES = 5

# Maximum OpenCL work-items per kernel dispatch.
# Keeping this at or below ~32 M prevents OUT_OF_RESOURCES / GPU watchdog
# (TDR on Windows, DRM timeout on Linux) on large rule × word batches.
# Lower this value (e.g. 8 * 1024 * 1024) if you still see crashes.
MAX_DISPATCH_ITEMS = 32 * 1024 * 1024

# Global variables for interrupt handling
interrupted = False
current_rules_list = None
current_ranking_output_path = None
current_top_k = 0
words_processed_total = None
total_unique_found = None
total_cracked_found = None

# ====================================================================
# --- COLOR CODES ---
# ====================================================================
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def red(text): return f"{Colors.RED}{text}{Colors.END}"
def green(text): return f"{Colors.GREEN}{text}{Colors.END}"
def yellow(text): return f"{Colors.YELLOW}{text}{Colors.END}"
def blue(text): return f"{Colors.BLUE}{text}{Colors.END}"
def cyan(text): return f"{Colors.CYAN}{text}{Colors.END}"
def bold(text): return f"{Colors.BOLD}{text}{Colors.END}"
def underline(text): return f"{Colors.UNDERLINE}{text}{Colors.END}"

# Suppress warnings
warnings.filterwarnings("ignore", message="overflow encountered in scalar multiply")
warnings.filterwarnings("ignore", message="overflow encountered in scalar add")
warnings.filterwarnings("ignore", message="overflow encountered in uint_scalars")
try:
    warnings.filterwarnings("ignore", message="The 'device_offset' argument of enqueue_copy is deprecated")
    warnings.filterwarnings("ignore", category=cl.CompilerWarning)
except AttributeError:
    pass

# ====================================================================
# --- PLATFORM AND DEVICE SELECTION ---
# ====================================================================
def list_platforms_and_devices():
    """List all available OpenCL platforms and devices"""
    platforms = cl.get_platforms()
    print(f"\n{blue('Available OpenCL Platforms and Devices:')}")
    print(f"{green('=' * 70)}")
    platform_info = []
    for i, platform in enumerate(platforms):
        platform_name = platform.name.strip()
        platform_vendor = platform.vendor.strip()
        print(f"{bold(f'Platform {i}:')} {platform_name}")
        print(f"{blue('Vendor:')} {platform_vendor}")
        try:
            devices = platform.get_devices()
            for j, device in enumerate(devices):
                device_type = "GPU" if device.type == cl.device_type.GPU else "CPU" if device.type == cl.device_type.CPU else "Accelerator"
                device_name = device.name.strip()
                device_memory = device.global_mem_size / (1024**3)  # GB
                print(f"  {bold(f'Device {i}-{j}:')} {device_name} ({device_type}) - {device_memory:.1f} GB")
                platform_info.append({
                    'platform_idx': i,
                    'device_idx': j,
                    'platform_name': platform_name,
                    'platform_vendor': platform_vendor,
                    'device_name': device_name,
                    'device_type': device_type,
                    'device_memory': device_memory
                })
        except Exception as e:
            print(f"  {red('Error getting devices:')} {e}")
        print(f"{green('=' * 70)}")
    return platform_info

def select_platform_and_device(platform_idx=None, device_idx=None):
    """Select specific platform and device, or auto-select if not specified"""
    platforms = cl.get_platforms()
    if not platforms:
        print(f"{red('No OpenCL platforms found!')}")
        exit(1)
    if platform_idx is not None:
        if platform_idx >= len(platforms):
            print(f"{red(f'Platform {platform_idx} not available. Available platforms:')}")
            list_platforms_and_devices()
            exit(1)
        platform = platforms[platform_idx]
    else:
        platform = None
        for p in platforms:
            vendor = p.vendor.strip().lower()
            if 'nvidia' in vendor:
                platform = p
                print(f"{green('Auto-selected NVIDIA platform')}")
                break
            elif 'amd' in vendor or 'advanced micro devices' in vendor:
                platform = p
                print(f"{green('Auto-selected AMD platform')}")
                break
            elif 'intel' in vendor:
                platform = p
                print(f"{green('Auto-selected Intel platform')}")
                break
        if platform is None:
            platform = platforms[0]
            print(f"{yellow('No preferred platform found, using first available:')} {platform.name.strip()}")
    try:
        devices = platform.get_devices()
    except Exception as e:
        print(f"{red('Error getting devices for platform:')} {e}")
        exit(1)
    if not devices:
        print(f"{red('No devices found on selected platform!')}")
        exit(1)
    if device_idx is not None:
        if device_idx >= len(devices):
            print(f"{red(f'Device {device_idx} not available. Available devices:')}")
            for j, d in enumerate(devices):
                print(f"  {bold(f'Device {j}:')} {d.name.strip()}")
            exit(1)
        device = devices[device_idx]
    else:
        device = None
        gpu_devices = [d for d in devices if d.type == cl.device_type.GPU]
        cpu_devices = [d for d in devices if d.type == cl.device_type.CPU]
        if gpu_devices:
            device = gpu_devices[0]
            print(f"{green('Auto-selected GPU device')}")
        elif cpu_devices:
            device = cpu_devices[0]
            print(f"{yellow('No GPU found, using CPU device (performance will be slower)')}")
        else:
            device = devices[0]
            print(f"{yellow('Using available device:')} {device.name.strip()}")
    return platform, device

# ====================================================================
# --- OPTIMIZED FILE LOADING FUNCTIONS ---
# ====================================================================
def estimate_word_count(path):
    """Fast word count estimation for large files without reading entire content"""
    print(f"{blue('Estimating words in:')} {path}...")
    try:
        file_size = os.path.getsize(path)
        sample_size = min(10 * 1024 * 1024, file_size)
        with open(path, 'rb') as f:
            sample = f.read(sample_size)
            lines = sample.count(b'\n')
            if file_size <= sample_size:
                total_lines = lines
            else:
                avg_line_length = sample_size / max(lines, 1)
                total_lines = int(file_size / avg_line_length)
        print(f"{green('Estimated words:')} {cyan(f'{total_lines:,}')}")
        return total_lines
    except Exception as e:
        print(f"{yellow('Could not estimate word count:')} {e}")
        return 1000000

def fast_fnv1a_hash_32(data):
    """Optimized FNV-1a hash for bytes - pure integer arithmetic (no numpy overhead)"""
    hash_val = 2166136261
    for byte in data:
        hash_val = (hash_val ^ byte) * 16777619 & 0xFFFFFFFF
    return hash_val

def optimized_wordlist_iterator(wordlist_path, max_len, batch_size):
    """Memory‑mapped iterator over words, returning batches of words and hashes"""
    print(f"{green('Using optimized memory-mapped loader...')}")
    file_size = os.path.getsize(wordlist_path)
    print(f"{blue('File size:')} {cyan(f'{file_size / (1024**3):.2f} GB')}")
    batch_elements = batch_size * max_len
    words_buffer = np.zeros(batch_elements, dtype=np.uint8)
    hashes_buffer = np.zeros(batch_size, dtype=np.uint32)
    load_start = time()
    total_words_loaded = 0
    try:
        with open(wordlist_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                pos = 0
                batch_count = 0
                file_size = len(mm)
                while pos < file_size and not interrupted:
                    end_pos = mm.find(b'\n', pos)
                    if end_pos == -1:
                        end_pos = file_size
                    line = mm[pos:end_pos].strip()
                    line_len = len(line)
                    pos = end_pos + 1
                    if line_len == 0 or line_len > max_len:
                        continue
                    start_idx = batch_count * max_len
                    end_idx = start_idx + line_len
                    words_buffer[start_idx:end_idx] = np.frombuffer(line, dtype=np.uint8, count=line_len)
                    hashes_buffer[batch_count] = fast_fnv1a_hash_32(line)
                    batch_count += 1
                    total_words_loaded += 1
                    if batch_count >= batch_size:
                        yield words_buffer.copy(), hashes_buffer.copy(), batch_count
                        batch_count = 0
                        words_buffer.fill(0)
                        hashes_buffer.fill(0)
                if batch_count > 0 and not interrupted:
                    yield words_buffer, hashes_buffer, batch_count
    except Exception as e:
        print(f"{red('Error in optimized loader:')} {e}")
        raise
    load_time = time() - load_start
    print(f"{green('Optimized loading completed:')} {cyan(f'{total_words_loaded:,}')} {bold('words in')} {load_time:.2f}s "
          f"({total_words_loaded/load_time:,.0f} words/sec)")

# ====================================================================
# --- INTERRUPT HANDLER ---
# ====================================================================
def signal_handler(sig, frame):
    global interrupted, current_rules_list, current_ranking_output_path, current_top_k
    global words_processed_total, total_unique_found, total_cracked_found
    print(f"\n{yellow('Interrupt received!')}")
    if interrupted:
        print(f"{red('Forced exit!')}")
        sys.exit(1)
    interrupted = True
    if current_rules_list is not None and current_ranking_output_path is not None:
        print(f"{blue('Saving current progress...')}")
        save_current_progress()
    else:
        print(f"{yellow('No data to save. Exiting...')}")
        sys.exit(1)

def save_current_progress():
    global current_rules_list, current_ranking_output_path, current_top_k
    global words_processed_total, total_unique_found, total_cracked_found
    try:
        base_path = os.path.splitext(current_ranking_output_path)[0]
        intermediate_output_path = f"{base_path}_INTERRUPTED.csv"
        intermediate_optimized_path = f"{base_path}_INTERRUPTED.rule"
        if current_rules_list:
            print(f"{blue('Saving intermediate results to:')} {intermediate_output_path}")
            for rule in current_rules_list:
                rule['combined_score'] = rule.get('effectiveness_score', 0) * 10 + rule.get('uniqueness_score', 0)
            ranked_rules = current_rules_list
            ranked_rules.sort(key=lambda rule: rule['combined_score'], reverse=True)
            with open(intermediate_output_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['Rank', 'Combined_Score', 'Effectiveness_Score', 'Uniqueness_Score', 'Rule_Data']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rank, rule in enumerate(ranked_rules, 1):
                    writer.writerow({
                        'Rank': rank,
                        'Combined_Score': rule['combined_score'],
                        'Effectiveness_Score': rule.get('effectiveness_score', 0),
                        'Uniqueness_Score': rule.get('uniqueness_score', 0),
                        'Rule_Data': rule['rule_data']
                    })
            print(f"{green('Intermediate ranking data saved:')} {cyan(f'{len(ranked_rules):,}')} {bold('rules')}")
            if current_top_k > 0:
                print(f"{blue('Saving intermediate optimized rules to:')} {intermediate_optimized_path}")
                available_rules = len(ranked_rules)
                final_count = min(current_top_k, available_rules)
                with open(intermediate_optimized_path, 'w', newline='\n', encoding='utf-8') as f:
                    f.write(":\n")
                    for rule in ranked_rules[:final_count]:
                        f.write(f"{rule['rule_data']}\n")
                print(f"{green('Intermediate optimized rules saved:')} {cyan(f'{final_count:,}')} {bold('rules')}")
        if words_processed_total is not None:
            print(f"\n{green('=' * 60)}")
            print(f"{bold('Progress Summary at Interruption')}")
            print(f"{green('=' * 60)}")
            print(f"{blue('Words Processed:')} {cyan(f'{int(words_processed_total):,}')}")
            if total_unique_found is not None:
                print(f"{blue('Unique Words Generated:')} {cyan(f'{int(total_unique_found):,}')}")
            if total_cracked_found is not None:
                print(f"{blue('True Cracks Found:')} {cyan(f'{int(total_cracked_found):,}')}")
            print(f"{green('=' * 60)}{Colors.END}\n")
        print(f"{green('Progress saved successfully. You can resume later using the intermediate files.')}")
    except Exception as e:
        print(f"{red('Error saving intermediate progress:')} {e}")
    sys.exit(0)

def setup_interrupt_handler(rules_list, ranking_output_path, top_k):
    global current_rules_list, current_ranking_output_path, current_top_k
    current_rules_list = rules_list
    current_ranking_output_path = ranking_output_path
    current_top_k = top_k
    signal.signal(signal.SIGINT, signal_handler)

def update_progress_stats(words_processed, unique_found, cracked_found):
    global words_processed_total, total_unique_found, total_cracked_found
    words_processed_total = words_processed
    total_unique_found = unique_found
    total_cracked_found = cracked_found

# ====================================================================
# --- HELPER FUNCTIONS (load_rules, load_cracked_hashes, encode_rule, save_ranking_data, ...) ---
# ====================================================================
def load_rules(path):
    """Loads Hashcat rules from file."""
    print(f"{blue('Loading rules from:')} {path}...")
    rules_size = 0
    try:
        rules_size = os.path.getsize(path) / (1024 * 1024)
        if rules_size > 10:
            print(f"{yellow('Large rules file detected:')} {rules_size:.1f} MB")
    except OSError:
        pass
    rules_list = []
    rule_id_counter = 0
    try:
        with open(path, 'r', encoding='latin-1') as f:
            for line in f:
                rule = line.strip()
                if not rule or rule.startswith('#'):
                    continue
                rules_list.append({'rule_data': rule, 'rule_id': rule_id_counter,
                                   'uniqueness_score': 0, 'effectiveness_score': 0})
                rule_id_counter += 1
    except FileNotFoundError:
        print(f"{red('Error:')} Rules file not found at: {path}")
        exit(1)
    print(f"{green('Loaded')} {cyan(f'{len(rules_list):,}')} {bold('rules.')}")
    return rules_list

def load_cracked_hashes(path, max_len):
    """Loads cracked passwords and returns their FNV‑1a hashes."""
    print(f"{blue('Loading cracked list for effectiveness check from:')} {path}...")
    cracked_hashes = []
    try:
        with open(path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                pos = 0
                file_size = len(mm)
                with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024,
                          desc=cyan('Cracked list'), colour='cyan',
                          bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                    while pos < file_size:
                        end_pos = mm.find(b'\n', pos)
                        if end_pos == -1:
                            end_pos = file_size
                        line = mm[pos:end_pos].strip()
                        advance = (end_pos + 1) - pos
                        pos = end_pos + 1
                        pbar.update(advance)
                        if 1 <= len(line) <= max_len:
                            cracked_hashes.append(fast_fnv1a_hash_32(line))
    except FileNotFoundError:
        print(f"{yellow('Warning:')} Cracked list file not found at: {path}. Effectiveness scores will be zero.")
        return np.array([], dtype=np.uint32)
    unique_hashes = np.unique(np.array(cracked_hashes, dtype=np.uint32))
    print(f"{green('Loaded')} {cyan(f'{len(unique_hashes):,}')} {bold('unique cracked password hashes.')}")
    return unique_hashes

def encode_rule(rule_str, rule_id):
    """
    Encodes a rule string into a sequence of uint32 values:
    - First uint32: rule ID
    - Following uint32s: rule string bytes packed 4 per uint32 (little‑endian)
    """
    rule_bytes = rule_str.encode('latin-1')
    rule_len = len(rule_bytes)
    num_uints = 1 + (rule_len + 3) // 4   # rule_id + ceil(rule_len/4)
    encoded = np.zeros(num_uints, dtype=np.uint32)
    encoded[0] = np.uint32(rule_id)
    # Pack rule bytes into the remaining uints
    for i in range(rule_len):
        uint_idx = 1 + (i // 4)
        byte_pos = i % 4
        encoded[uint_idx] |= (np.uint32(rule_bytes[i]) << (byte_pos * 8))
    return encoded

def encode_rule_fixed(rule_str, rule_id, max_rule_len=MAX_RULE_LEN):
    """
    Encodes a rule string into fixed-length byte array for GPU.
    Used in legacy mode.
    """
    rule_bytes = rule_str.encode('latin-1')
    rule_len = len(rule_bytes)
    encoded = np.zeros(max_rule_len, dtype=np.uint8)
    encoded[:rule_len] = np.frombuffer(rule_bytes, dtype=np.uint8, count=rule_len)
    return encoded

def save_ranking_data(ranking_list, output_path, legacy=False):
    """Saves the scoring and ranking data to a CSV file."""
    ranking_output_path = output_path
    print(f"{blue('Saving rule ranking data to:')} {ranking_output_path}...")
    for rule in ranking_list:
        rule['combined_score'] = rule.get('effectiveness_score', 0) * 10 + rule.get('uniqueness_score', 0)
    ranked_rules = ranking_list
    ranked_rules.sort(key=lambda rule: rule['combined_score'], reverse=True)
    print(f"{blue('Saving ALL')} {cyan(f'{len(ranked_rules):,}')} {bold('rules (including zero-score rules)')}")
    if not ranked_rules:
        print(f"{red('No rules to save. Ranking file not created.')}")
        return None
    try:
        with open(ranking_output_path, 'w', newline='', encoding='utf-8') as f:
            if legacy:
                fieldnames = ['Rank', 'Combined_Score', 'Effectiveness_Score', 'Uniqueness_Score', 'Rule_Data']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rank, rule in enumerate(ranked_rules, 1):
                    writer.writerow({
                        'Rank': rank,
                        'Combined_Score': rule['combined_score'],
                        'Effectiveness_Score': rule.get('effectiveness_score', 0),
                        'Uniqueness_Score': rule.get('uniqueness_score', 0),
                        'Rule_Data': rule['rule_data']
                    })
            else:
                # MAB mode includes extra columns
                fieldnames = ['Rank', 'Combined_Score', 'Effectiveness_Score', 'Uniqueness_Score',
                              'MAB_Success_Prob', 'Times_Tested', 'MAB_Trials', 'Selections',
                              'Total_Successes', 'Total_Trials', 'Eliminated', 'Eliminate_Reason', 'Rule_Data']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rank, rule in enumerate(ranked_rules, 1):
                    writer.writerow({
                        'Rank': rank,
                        'Combined_Score': rule.get('combined_score', 0),
                        'Effectiveness_Score': rule.get('effectiveness_score', 0),
                        'Uniqueness_Score': rule.get('uniqueness_score', 0),
                        'MAB_Success_Prob': rule.get('mab_success_prob', 0),
                        'Times_Tested': rule.get('times_tested', 0),
                        'MAB_Trials': rule.get('mab_trials', 0),
                        'Selections': rule.get('selections', 0),
                        'Total_Successes': rule.get('total_successes', 0),
                        'Total_Trials': rule.get('total_trials', 0),
                        'Eliminated': rule.get('eliminated', False),
                        'Eliminate_Reason': rule.get('eliminate_reason', ''),
                        'Rule_Data': rule['rule_data']
                    })
        print(f"{green('Ranking data saved successfully to')} {ranking_output_path}.")
        return ranking_output_path
    except Exception as e:
        print(f"{red('Error while saving ranking data to CSV file:')} {e}")
        return None

def load_and_save_optimized_rules(csv_path, output_path, top_k):
    """Loads ranking data from CSV, sorts, and saves the Top K rules."""
    if not csv_path:
        print(f"{yellow('Optimization skipped: Ranking CSV path is missing.')}")
        return
    print(f"{blue('Loading ranking from CSV:')} {csv_path} {bold('and saving Top')} {cyan(f'{top_k}')} {bold('Optimized Rules to:')} {output_path}...")
    ranked_data = []
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row['Combined_Score'] = int(row['Combined_Score'])
                    ranked_data.append(row)
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"{red('Error: Ranking CSV file not found at:')} {csv_path}")
        return
    except Exception as e:
        print(f"{red('Error while reading CSV:')} {e}")
        return
    print(f"{blue('Loaded')} {cyan(f'{len(ranked_data):,}')} {bold('total rules from CSV')}")
    ranked_data.sort(key=lambda row: row['Combined_Score'], reverse=True)
    available_rules = len(ranked_data)
    if top_k > available_rules:
        print(f"{yellow('Warning: Requested')} {cyan(f'{top_k:,}')} {bold('rules but only')} {cyan(f'{available_rules:,}')} {bold('available. Saving')} {cyan(f'{available_rules:,}')} {bold('rules.')}")
        final_optimized_list = ranked_data[:available_rules]
    else:
        final_optimized_list = ranked_data[:top_k]
    if not final_optimized_list:
        print(f"{red('No rules available after sorting/filtering. Optimized rule file not created.')}")
        return
    try:
        with open(output_path, 'w', newline='\n', encoding='utf-8') as f:
            f.write(":\n")  # Default rule
            for rule in final_optimized_list:
                f.write(f"{rule['Rule_Data']}\n")
        print(f"{green('Top')} {cyan(f'{len(final_optimized_list):,}')} {bold('optimized rules saved successfully to')} {output_path}.")
    except Exception as e:
        print(f"{red('Error while saving optimized rules to file:')} {e}")

# ====================================================================
# --- MEMORY MANAGEMENT FUNCTIONS ---
# ====================================================================
def get_gpu_memory_info(device):
    try:
        total_memory = device.global_mem_size
        available_memory = int(total_memory * (1 - VRAM_SAFETY_MARGIN))
        return total_memory, available_memory
    except Exception as e:
        print(f"{yellow('Warning: Could not query GPU memory:')} {e}")
        return 8 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024

def calculate_optimal_parameters_large_rules(available_vram, total_words, cracked_hashes_count, total_rules, reduction_factor=1.0):
    print(f"{blue('Calculating optimal parameters for')} {cyan(f'{available_vram / (1024**3):.1f} GB')} {bold('available VRAM')}")
    if reduction_factor < 1.0:
        print(f"{yellow('Applying memory reduction factor:')} {cyan(f'{reduction_factor:.2f}')}")
    available_vram = int(available_vram * reduction_factor)
    word_batch_bytes = MAX_WORD_LEN * np.uint8().itemsize
    hash_batch_bytes = np.uint32().itemsize
    rule_batch_bytes = MAX_RULES_IN_BATCH * MAX_RULE_LEN * np.uint8().itemsize
    counter_bytes = MAX_RULES_IN_BATCH * np.uint32().itemsize * 2
    base_memory = ((word_batch_bytes + hash_batch_bytes) * 2 + rule_batch_bytes + counter_bytes)
    if total_rules > 100000:
        suggested_batch_size = min(DEFAULT_WORDS_PER_GPU_BATCH, 150000)
    else:
        suggested_batch_size = DEFAULT_WORDS_PER_GPU_BATCH
    available_for_maps = available_vram - base_memory
    if available_for_maps <= 0:
        print(f"{yellow('Warning: Limited VRAM, using minimal configuration')}")
        available_for_maps = available_vram * 0.5
    print(f"{blue('Available for hash maps:')} {cyan(f'{available_for_maps / (1024**3):.2f} GB')}")
    global_bits = DEFAULT_GLOBAL_HASH_MAP_BITS
    cracked_bits = DEFAULT_CRACKED_HASH_MAP_BITS
    if total_words > 0:
        required_global_bits = max(MIN_HASH_MAP_BITS, math.ceil(math.log2(total_words)) + 8)
        global_bits = min(required_global_bits, DEFAULT_GLOBAL_HASH_MAP_BITS)
    if cracked_hashes_count > 0:
        required_cracked_bits = max(MIN_HASH_MAP_BITS, math.ceil(math.log2(cracked_hashes_count)) + 8)
        cracked_bits = min(required_cracked_bits, DEFAULT_CRACKED_HASH_MAP_BITS)
    global_map_bytes = (1 << (global_bits - 5)) * np.uint32().itemsize
    cracked_map_bytes = (1 << (cracked_bits - 5)) * np.uint32().itemsize
    total_map_memory = global_map_bytes + cracked_map_bytes
    while total_map_memory > available_for_maps and global_bits > MIN_HASH_MAP_BITS and cracked_bits > MIN_HASH_MAP_BITS:
        if global_bits > cracked_bits:
            global_bits -= 1
        else:
            cracked_bits -= 1
        global_map_bytes = (1 << (global_bits - 5)) * np.uint32().itemsize
        cracked_map_bytes = (1 << (cracked_bits - 5)) * np.uint32().itemsize
        total_map_memory = global_map_bytes + cracked_map_bytes
    memory_per_word = (word_batch_bytes + hash_batch_bytes +
                       (MAX_OUTPUT_LEN * np.uint8().itemsize) +
                       (rule_batch_bytes / MAX_RULES_IN_BATCH))
    max_batch_by_memory = int((available_vram - total_map_memory - base_memory) / memory_per_word)
    optimal_batch_size = min(suggested_batch_size, max_batch_by_memory)
    optimal_batch_size = max(MIN_BATCH_SIZE, optimal_batch_size)
    optimal_batch_size = (optimal_batch_size // LOCAL_WORK_SIZE) * LOCAL_WORK_SIZE
    if total_rules > 50000:
        optimal_batch_size = max(MIN_BATCH_SIZE, optimal_batch_size // 2)
    print(f"{green('Optimal configuration:')}")
    print(f"   {blue('-')} {bold('Batch size:')} {cyan(f'{optimal_batch_size:,} words')}")
    print(f"   {blue('-')} {bold('Rules per batch:')} {cyan(f'{MAX_RULES_IN_BATCH:,}')}")
    print(f"   {blue('-')} {bold('Global hash map:')} {cyan(f'{global_bits} bits')} ({global_map_bytes / (1024**2):.1f} MB)")
    print(f"   {blue('-')} {bold('Cracked hash map:')} {cyan(f'{cracked_bits} bits')} ({cracked_map_bytes / (1024**2):.1f} MB)")
    print(f"   {blue('-')} {bold('Total map memory:')} {cyan(f'{total_map_memory / (1024**3):.2f} GB')}")
    print(f"   {blue('-')} {bold('Estimated rule batches:')} {cyan(f'{(total_rules + MAX_RULES_IN_BATCH - 1) // MAX_RULES_IN_BATCH}')}")
    return optimal_batch_size, global_bits, cracked_bits

def get_recommended_parameters(device, total_words, cracked_hashes_count):
    total_vram, available_vram = get_gpu_memory_info(device)
    recommendations = {
        "low_memory":   {"description": "Low Memory Mode (for GPUs with < 4GB VRAM)",  "batch_size": 25000, "global_bits": 30, "cracked_bits": 28},
        "medium_memory":{"description": "Medium Memory Mode (for GPUs with 4-8GB VRAM)","batch_size": 75000, "global_bits": 33, "cracked_bits": 31},
        "high_memory":  {"description": "High Memory Mode (for GPUs with > 8GB VRAM)", "batch_size": 150000,"global_bits": 35, "cracked_bits": 33},
        "auto":         {"description": "Auto-calculated (Recommended)", "batch_size": None, "global_bits": None, "cracked_bits": None}
    }
    if total_vram < 4 * 1024**3:
        recommended_preset = "low_memory"
    elif total_vram < 8 * 1024**3:
        recommended_preset = "medium_memory"
    else:
        recommended_preset = "high_memory"
    auto_batch, auto_global, auto_cracked = calculate_optimal_parameters_large_rules(
        available_vram, total_words, cracked_hashes_count, total_words)
    recommendations["auto"]["batch_size"] = auto_batch
    recommendations["auto"]["global_bits"] = auto_global
    recommendations["auto"]["cracked_bits"] = auto_cracked
    return recommendations, recommended_preset

def create_opencl_buffers_with_retry(context, buffer_specs, max_retries=MAX_ALLOCATION_RETRIES):
    buffers = {}
    current_reduction = 1.0
    for retry in range(max_retries + 1):
        try:
            print(f"{blue('Attempt')} {cyan(f'{retry + 1}/{max_retries + 1}')} {bold('to allocate buffers')} (reduction: {current_reduction:.2f})")
            for name, spec in buffer_specs.items():
                flags = spec['flags']
                size = int(spec['size'] * current_reduction)
                if 'hostbuf' in spec:
                    buffers[name] = cl.Buffer(context, flags, size, hostbuf=spec['hostbuf'])
                else:
                    buffers[name] = cl.Buffer(context, flags, size)
            print(f"{green('Successfully allocated all buffers on attempt')} {cyan(f'{retry + 1}')}")
            return buffers
        except cl.MemoryError as e:
            if "MEM_OBJECT_ALLOCATION_FAILURE" in str(e) and retry < max_retries:
                print(f"{yellow('Memory allocation failed, reducing memory usage...')}")
                current_reduction *= MEMORY_REDUCTION_FACTOR
                for buf in buffers.values():
                    try:
                        buf.release()
                    except:
                        pass
                buffers = {}
            else:
                raise e
    raise cl.MemoryError(f"{red('Failed to allocate buffers after')} {cyan(f'{max_retries}')} {bold('retries')}")

# ====================================================================
# --- COMPREHENSIVE KERNEL SOURCE (Full Hashcat Rules) ---
# ====================================================================
def get_kernel_source(global_hash_map_bits, cracked_hash_map_bits):
    global_hash_map_mask = (1 << (global_hash_map_bits - 5)) - 1
    cracked_hash_map_mask = (1 << (cracked_hash_map_bits - 5)) - 1
    return f"""
// ============================================================================
// COMPREHENSIVE HASHCAT RULES KERNEL – WITH RULE CHAIN SUPPORT
// ============================================================================

#define MAX_WORD_LEN {MAX_WORD_LEN}
#define MAX_OUTPUT_LEN {MAX_OUTPUT_LEN}
#define MAX_RULE_LEN {MAX_RULE_LEN}
#define GLOBAL_HASH_MAP_MASK {global_hash_map_mask}
#define CRACKED_HASH_MAP_MASK {cracked_hash_map_mask}

// ----------------------------------------------------------------------------
// Basic utility functions
// ----------------------------------------------------------------------------
int is_lower(unsigned char c) {{ return (c >= 'a' && c <= 'z'); }}
int is_upper(unsigned char c) {{ return (c >= 'A' && c <= 'Z'); }}
int is_digit(unsigned char c) {{ return (c >= '0' && c <= '9'); }}
unsigned char to_lower(unsigned char c) {{ return is_upper(c) ? c + 32 : c; }}
unsigned char to_upper(unsigned char c) {{ return is_lower(c) ? c - 32 : c; }}
unsigned char toggle_case(unsigned char c) {{
    if (is_lower(c)) return c - 32;
    if (is_upper(c)) return c + 32;
    return c;
}}

unsigned int char_to_pos(unsigned char c) {{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    if (c >= 'a' && c <= 'z') return c - 'a' + 10;
    return 0xFFFFFFFF;
}}

unsigned int fnv1a_hash_32(const unsigned char* data, unsigned int len) {{
    unsigned int hash = 2166136261U;
    for (unsigned int i = 0; i < len; i++) {{
        hash ^= data[i];
        hash *= 16777619U;
    }}
    return hash;
}}

// ----------------------------------------------------------------------------
// Operation helpers (take input buffer, output buffer, lengths, and arguments)
// ----------------------------------------------------------------------------
static void duplicate_front(const unsigned char* in, int in_len,
                            unsigned char* out, int* out_len, int* changed, int n) {{
    if (n > in_len) n = in_len;
    int new_len = in_len + n;
    if (new_len > MAX_OUTPUT_LEN) return;
    for (int i = 0; i < in_len; i++) out[i] = in[i];
    for (int i = 0; i < n; i++) out[in_len + i] = in[i];
    *out_len = new_len;
    *changed = 1;
}}

static void duplicate_back(const unsigned char* in, int in_len,
                           unsigned char* out, int* out_len, int* changed, int n) {{
    if (n > in_len) n = in_len;
    int new_len = in_len + n;
    if (new_len > MAX_OUTPUT_LEN) return;
    for (int i = 0; i < in_len; i++) out[i] = in[i];
    for (int i = 0; i < n; i++) out[in_len + i] = in[in_len - n + i];
    *out_len = new_len;
    *changed = 1;
}}

static void duplicate_word(const unsigned char* in, int in_len,
                           unsigned char* out, int* out_len, int* changed, int times) {{
    int new_len = in_len * (times + 1);
    if (new_len > MAX_OUTPUT_LEN) return;
    for (int rep = 0; rep <= times; rep++) {{
        for (int i = 0; i < in_len; i++) {{
            out[rep * in_len + i] = in[i];
        }}
    }}
    *out_len = new_len;
    *changed = 1;
}}

static void rotate_left(const unsigned char* in, int in_len,
                        unsigned char* out, int* out_len, int* changed, int n) {{
    if (n <= 0) n = 1;
    n %= in_len;
    if (n == 0) {{
        *out_len = in_len;
        for (int i = 0; i < in_len; i++) out[i] = in[i];
        *changed = 0;
        return;
    }}
    *out_len = in_len;
    for (int i = 0; i < in_len; i++) {{
        out[i] = in[(i + n) % in_len];
    }}
    *changed = 1;
}}

static void rotate_right(const unsigned char* in, int in_len,
                         unsigned char* out, int* out_len, int* changed, int n) {{
    if (n <= 0) n = 1;
    n %= in_len;
    if (n == 0) {{
        *out_len = in_len;
        for (int i = 0; i < in_len; i++) out[i] = in[i];
        *changed = 0;
        return;
    }}
    *out_len = in_len;
    for (int i = 0; i < in_len; i++) {{
        out[i] = in[(i - n + in_len) % in_len];
    }}
    *changed = 1;
}}

// ----------------------------------------------------------------------------
// Single‑command application (returns 0 if successful, -1 if reject)
// ----------------------------------------------------------------------------
static int apply_single_command(const unsigned char* in, int in_len,
                                unsigned char* out, int* out_len,
                                const unsigned char* cmd, int cmd_len) {{
    int changed = 0;
    *out_len = 0;

    // --- Single‑character commands ---
    if (cmd_len == 1) {{
        switch (cmd[0]) {{
            case 'l':
                *out_len = in_len;
                for (int i = 0; i < in_len; i++) out[i] = to_lower(in[i]);
                changed = 1;
                break;
            case 'u':
                *out_len = in_len;
                for (int i = 0; i < in_len; i++) out[i] = to_upper(in[i]);
                changed = 1;
                break;
            case 'c':
                *out_len = in_len;
                if (in_len > 0) out[0] = to_upper(in[0]);
                for (int i = 1; i < in_len; i++) out[i] = to_lower(in[i]);
                changed = 1;
                break;
            case 'C':
                *out_len = in_len;
                if (in_len > 0) out[0] = to_lower(in[0]);
                for (int i = 1; i < in_len; i++) out[i] = to_upper(in[i]);
                changed = 1;
                break;
            case 't':
                *out_len = in_len;
                for (int i = 0; i < in_len; i++) out[i] = toggle_case(in[i]);
                changed = 1;
                break;
            case 'r':
                *out_len = in_len;
                for (int i = 0; i < in_len; i++) out[i] = in[in_len - 1 - i];
                changed = 1;
                break;
            case 'd':
                if (in_len * 2 <= MAX_OUTPUT_LEN) {{
                    *out_len = in_len * 2;
                    for (int i = 0; i < in_len; i++) {{
                        out[i] = in[i];
                        out[in_len + i] = in[i];
                    }}
                    changed = 1;
                }}
                break;
            case 'f':
                if (in_len * 2 <= MAX_OUTPUT_LEN) {{
                    *out_len = in_len * 2;
                    for (int i = 0; i < in_len; i++) {{
                        out[i] = in[i];
                        out[in_len + i] = in[in_len - 1 - i];
                    }}
                    changed = 1;
                }}
                break;
            case 'k':
                *out_len = in_len;
                for (int i = 0; i < in_len; i++) out[i] = in[i];
                if (in_len >= 2) {{
                    out[0] = in[1];
                    out[1] = in[0];
                    changed = 1;
                }}
                break;
            case 'K':
                *out_len = in_len;
                for (int i = 0; i < in_len; i++) out[i] = in[i];
                if (in_len >= 2) {{
                    out[in_len-2] = in[in_len-1];
                    out[in_len-1] = in[in_len-2];
                    changed = 1;
                }}
                break;
            case ':':
                *out_len = in_len;
                for (int i = 0; i < in_len; i++) out[i] = in[i];
                changed = 0;
                break;
            case 'q':
                if (in_len * 2 <= MAX_OUTPUT_LEN) {{
                    int idx = 0;
                    for (int i = 0; i < in_len; i++) {{
                        out[idx++] = in[i];
                        out[idx++] = in[i];
                    }}
                    *out_len = in_len * 2;
                    changed = 1;
                }}
                break;
            case 'E':
                *out_len = in_len;
                int cap = 1;
                for (int i = 0; i < in_len; i++) {{
                    if (cap && is_lower(in[i]))
                        out[i] = to_upper(in[i]);
                    else
                        out[i] = to_lower(in[i]);
                    cap = (in[i] == ' ' || in[i] == '-' || in[i] == '_');
                }}
                changed = 1;
                break;
            case '{{':
                rotate_left(in, in_len, out, out_len, &changed, 1);
                break;
            case '}}':
                rotate_right(in, in_len, out, out_len, &changed, 1);
                break;
            case '[':
                if (in_len > 1) {{
                    *out_len = in_len - 1;
                    for (int i = 1; i < in_len; i++) out[i-1] = in[i];
                    changed = 1;
                }}
                break;
            case ']':
                if (in_len > 1) {{
                    *out_len = in_len - 1;
                    for (int i = 0; i < in_len-1; i++) out[i] = in[i];
                    changed = 1;
                }}
                break;
            default:
                // unknown single char -> identity
                *out_len = in_len;
                for (int i = 0; i < in_len; i++) out[i] = in[i];
                changed = 0;
                break;
        }}
        return changed;
    }}

    // --- Two‑character commands ---
    if (cmd_len == 2) {{
        unsigned char cmd_char = cmd[0];
        unsigned char arg = cmd[1];
        int n = (int)char_to_pos(arg);
        if (n == 0xFFFFFFFF) n = -1;

        switch (cmd_char) {{
            case 'T':
                if (n >= 0 && n < in_len) {{
                    *out_len = in_len;
                    for (int i = 0; i < in_len; i++) out[i] = in[i];
                    out[n] = toggle_case(in[n]);
                    changed = 1;
                }}
                break;
            case 'D':
                if (n >= 0 && n < in_len) {{
                    *out_len = in_len - 1;
                    for (int i = 0; i < n; i++) out[i] = in[i];
                    for (int i = n+1; i < in_len; i++) out[i-1] = in[i];
                    changed = 1;
                }}
                break;
            case 'L':
                if (n >= 0 && n < in_len) {{
                    *out_len = in_len - n;
                    for (int i = n; i < in_len; i++) out[i-n] = in[i];
                    changed = 1;
                }}
                break;
            case 'R':
                if (n >= 0 && n < in_len) {{
                    *out_len = n + 1;
                    for (int i = 0; i <= n; i++) out[i] = in[i];
                    changed = 1;
                }}
                break;
            case '+':
                if (n >= 0 && n < in_len) {{
                    *out_len = in_len;
                    for (int i = 0; i < in_len; i++) out[i] = in[i];
                    out[n] = in[n] + 1;
                    changed = 1;
                }}
                break;
            case '-':
                if (n >= 0 && n < in_len) {{
                    *out_len = in_len;
                    for (int i = 0; i < in_len; i++) out[i] = in[i];
                    out[n] = in[n] - 1;
                    changed = 1;
                }}
                break;
            case '.':
                if (n >= 0 && n < in_len) {{
                    *out_len = in_len;
                    for (int i = 0; i < in_len; i++) out[i] = in[i];
                    out[n] = in[n] + 1;
                    changed = 1;
                }}
                break;
            case ',':
                if (n >= 0 && n < in_len) {{
                    *out_len = in_len;
                    for (int i = 0; i < in_len; i++) out[i] = in[i];
                    out[n] = in[n] - 1;
                    changed = 1;
                }}
                break;
            case '\\'':
                if (n >= 0 && n < in_len) {{
                    *out_len = n;
                    for (int i = 0; i < n; i++) out[i] = in[i];
                    changed = 1;
                }}
                break;
            case '^':
                if (in_len + 1 <= MAX_OUTPUT_LEN) {{
                    out[0] = arg;
                    for (int i = 0; i < in_len; i++) out[i+1] = in[i];
                    *out_len = in_len + 1;
                    changed = 1;
                }}
                break;
            case '$':
                if (in_len + 1 <= MAX_OUTPUT_LEN) {{
                    for (int i = 0; i < in_len; i++) out[i] = in[i];
                    out[in_len] = arg;
                    *out_len = in_len + 1;
                    changed = 1;
                }}
                break;
            case '@':
                *out_len = 0;
                for (int i = 0; i < in_len; i++) {{
                    if (in[i] != arg) out[(*out_len)++] = in[i];
                    else changed = 1;
                }}
                break;
            case '!':
                for (int i = 0; i < in_len; i++) {{
                    if (in[i] == arg) return -1;
                }}
                *out_len = in_len;
                for (int i = 0; i < in_len; i++) out[i] = in[i];
                return 0;
            case '/':
                for (int i = 0; i < in_len; i++) {{
                    if (in[i] == arg) {{
                        *out_len = in_len;
                        for (int j = 0; j < in_len; j++) out[j] = in[j];
                        return 0;
                    }}
                }}
                return -1;
            case '(':
                if (in_len > 0 && in[0] == arg) {{
                    *out_len = in_len;
                    for (int i = 0; i < in_len; i++) out[i] = in[i];
                    return 0;
                }}
                return -1;
            case ')':
                if (in_len > 0 && in[in_len-1] == arg) {{
                    *out_len = in_len;
                    for (int i = 0; i < in_len; i++) out[i] = in[i];
                    return 0;
                }}
                return -1;
            case 'y':
                if (n >= 0) duplicate_front(in, in_len, out, out_len, &changed, n);
                break;
            case 'Y':
                if (n >= 0) duplicate_back(in, in_len, out, out_len, &changed, n);
                break;
            case 'z':
                if (n > 0 && in_len + n <= MAX_OUTPUT_LEN) {{
                    out[0] = in[0];
                    for (int i = 0; i < n; i++) out[i+1] = in[0];
                    for (int i = 1; i < in_len; i++) out[n + i] = in[i];
                    *out_len = in_len + n;
                    changed = 1;
                }}
                break;
            case 'Z':
                if (n > 0 && in_len + n <= MAX_OUTPUT_LEN) {{
                    for (int i = 0; i < in_len; i++) out[i] = in[i];
                    for (int i = 0; i < n; i++) out[in_len + i] = in[in_len-1];
                    *out_len = in_len + n;
                    changed = 1;
                }}
                break;
            case 'p':
                if (n >= 0) duplicate_word(in, in_len, out, out_len, &changed, n);
                break;
            case '{{':
                if (n >= 0) rotate_left(in, in_len, out, out_len, &changed, n);
                break;
            case '}}':
                if (n >= 0) rotate_right(in, in_len, out, out_len, &changed, n);
                break;
            case '[':
                if (n >= 0 && n < in_len) {{
                    *out_len = in_len - n;
                    for (int i = n; i < in_len; i++) out[i-n] = in[i];
                    changed = 1;
                }}
                break;
            case ']':
                if (n >= 0 && n < in_len) {{
                    *out_len = in_len - n;
                    for (int i = 0; i < *out_len; i++) out[i] = in[i];
                    changed = 1;
                }}
                break;
            case '_':
                if (n >= 0 && in_len != n) return -1;
                *out_len = in_len;
                for (int i = 0; i < in_len; i++) out[i] = in[i];
                return 0;
            case 'e':
                *out_len = in_len;
                int cap_sep = 1;
                for (int i = 0; i < in_len; i++) {{
                    if (cap_sep && is_lower(in[i]))
                        out[i] = to_upper(in[i]);
                    else
                        out[i] = to_lower(in[i]);
                    cap_sep = (in[i] == arg);
                }}
                changed = 1;
                break;
            default:
                *out_len = in_len;
                for (int i = 0; i < in_len; i++) out[i] = in[i];
                changed = 0;
                break;
        }}
        return changed;
    }}

    // --- Three‑character commands ---
    if (cmd_len == 3) {{
        unsigned char cmd_char = cmd[0];
        unsigned char a1 = cmd[1];
        unsigned char a2 = cmd[2];
        int n1 = (int)char_to_pos(a1);
        int n2 = (int)char_to_pos(a2);

        switch (cmd_char) {{
            case 's':
                *out_len = in_len;
                for (int i = 0; i < in_len; i++) {{
                    out[i] = (in[i] == a1) ? a2 : in[i];
                    if (in[i] == a1) changed = 1;
                }}
                break;
            case 'x':
                if (n1 >= 0 && n2 > 0 && n1 < in_len) {{
                    int end = n1 + n2;
                    if (end > in_len) end = in_len;
                    *out_len = end - n1;
                    for (int i = n1; i < end; i++) out[i-n1] = in[i];
                    changed = 1;
                }}
                break;
            case 'O':
                if (n1 >= 0 && n2 > 0 && n1 < in_len) {{
                    int end = n1 + n2;
                    if (end > in_len) end = in_len;
                    *out_len = in_len - (end - n1);
                    for (int i = 0; i < n1; i++) out[i] = in[i];
                    for (int i = end; i < in_len; i++) out[i - n2] = in[i];
                    changed = 1;
                }}
                break;
            case 'i':
                if (n1 >= 0 && in_len + 1 <= MAX_OUTPUT_LEN) {{
                    if (n1 > in_len) n1 = in_len;
                    *out_len = in_len + 1;
                    for (int i = 0; i < n1; i++) out[i] = in[i];
                    out[n1] = a2;
                    for (int i = n1; i < in_len; i++) out[i+1] = in[i];
                    changed = 1;
                }}
                break;
            case 'o':
                if (n1 >= 0 && n1 < in_len) {{
                    *out_len = in_len;
                    for (int i = 0; i < in_len; i++) out[i] = in[i];
                    out[n1] = a2;
                    changed = 1;
                }}
                break;
            case '*':
                if (n1 >= 0 && n2 >= 0 && n1 < in_len && n2 < in_len && n1 != n2) {{
                    *out_len = in_len;
                    for (int i = 0; i < in_len; i++) out[i] = in[i];
                    unsigned char temp = out[n1];
                    out[n1] = out[n2];
                    out[n2] = temp;
                    changed = 1;
                }}
                break;
            case '3':
                if (n1 >= 0) {{
                    int count = 0;
                    *out_len = in_len;
                    for (int i = 0; i < in_len; i++) out[i] = in[i];
                    for (int i = 0; i < in_len; i++) {{
                        if (in[i] == a2) count++;
                        if (count == n1 + 1 && i+1 < in_len) {{
                            out[i+1] = toggle_case(in[i+1]);
                            changed = 1;
                            break;
                        }}
                    }}
                }}
                break;
            case '%':
                if (n1 >= 0) {{
                    int cnt = 0;
                    for (int i = 0; i < in_len; i++) {{
                        if (in[i] == a2) cnt++;
                    }}
                    if (cnt < n1) return -1;
                }}
                *out_len = in_len;
                for (int i = 0; i < in_len; i++) out[i] = in[i];
                return 0;
            case '=':
                if (n1 >= 0 && n1 < in_len && in[n1] != a2) return -1;
                *out_len = in_len;
                for (int i = 0; i < in_len; i++) out[i] = in[i];
                return 0;
            default:
                *out_len = in_len;
                for (int i = 0; i < in_len; i++) out[i] = in[i];
                changed = 0;
                break;
        }}
        return changed;
    }}

    // Unknown command length -> identity
    *out_len = in_len;
    for (int i = 0; i < in_len; i++) out[i] = in[i];
    return 0;
}}

// ----------------------------------------------------------------------------
// Main rule application (iterates over the rule string, applying commands)
// ----------------------------------------------------------------------------
void apply_hashcat_rule(const unsigned char* word, int word_len,
                        const unsigned char* rule, int rule_len,
                        unsigned char* output, int* out_len, int* changed) {{
    // Two working buffers
    unsigned char buf0[MAX_OUTPUT_LEN];
    unsigned char buf1[MAX_OUTPUT_LEN];
    unsigned char* in_buf = (unsigned char*)word;
    int in_len = word_len;
    int cur_changed = 0;
    int final_changed = 0;

    int pos = 0;
    while (pos < rule_len) {{
        // Determine command length
        unsigned char cmd_char = rule[pos];
        int cmd_len = 1;
        if (cmd_char == 's' || cmd_char == 'x' || cmd_char == 'O' || cmd_char == 'i' ||
            cmd_char == 'o' || cmd_char == '*' || cmd_char == '3' || cmd_char == '%' || cmd_char == '=') {{
            cmd_len = 3;
        }} else if (pos + 1 < rule_len && (cmd_char == 'T' || cmd_char == 'D' || cmd_char == 'L' ||
                                         cmd_char == 'R' || cmd_char == '+' || cmd_char == '-' ||
                                         cmd_char == '.' || cmd_char == ',' || cmd_char == '\\'' ||
                                         cmd_char == '^' || cmd_char == '$' || cmd_char == '@' ||
                                         cmd_char == '!' || cmd_char == '/' || cmd_char == '(' ||
                                         cmd_char == ')' || cmd_char == 'y' || cmd_char == 'Y' ||
                                         cmd_char == 'z' || cmd_char == 'Z' || cmd_char == 'p' ||
                                         cmd_char == '{{' || cmd_char == '}}' || cmd_char == '[' ||
                                         cmd_char == ']' || cmd_char == '_' || cmd_char == 'e')) {{
            cmd_len = 2;
        }}
        // Ensure we don't go beyond rule length
        if (pos + cmd_len > rule_len) {{
            // Partial command at end – treat as identity
            break;
        }}

        // Apply the command to the current input buffer
        int result = apply_single_command(in_buf, in_len, buf0, out_len, rule + pos, cmd_len);
        if (result == -1) {{
            // Reject: no output
            *out_len = 0;
            *changed = -1;
            return;
        }}
        if (result == 1) {{
            final_changed = 1;
        }}

        // Swap buffers for next iteration
        // New input becomes output of this command
        unsigned char* temp = (unsigned char*)in_buf;
        in_buf = buf0;
        in_len = *out_len;
        // Copy result to buf1 for next iteration if needed (we'll swap)
        // For now, we'll use two buffers and swap pointers
        // But we need to copy to buf1 if we want to reuse buf0 for next output
        // Simpler: after applying, copy from buf0 to buf1 and swap
        for (int i = 0; i < in_len; i++) {{
            buf1[i] = buf0[i];
        }}
        in_buf = buf1;
        pos += cmd_len;
    }}

    // Final result is in in_buf
    *out_len = in_len;
    for (int i = 0; i < in_len; i++) {{
        output[i] = in_buf[i];
    }}
    *changed = final_changed;
}}

// ----------------------------------------------------------------------------
// Ranking kernel (unchanged)
// ----------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void ranker_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned char* rules_in,
    __global const unsigned int* rule_ids,
    __global unsigned int* global_hash_map,
    __global const unsigned int* cracked_hash_map,
    __global unsigned int* rule_uniqueness_counts,
    __global unsigned int* rule_effectiveness_counts,
    const unsigned int num_words,
    const unsigned int num_rules_in_batch,
    const unsigned int max_word_len,
    const unsigned int max_output_len)
{{
    unsigned int global_id = get_global_id(0);
    unsigned int word_per_rule_count = num_words * num_rules_in_batch;
    if (global_id >= word_per_rule_count) return;

    unsigned int word_idx = global_id / num_rules_in_batch;
    unsigned int rule_batch_idx = global_id % num_rules_in_batch;

    // Load word
    unsigned char word[MAX_WORD_LEN];
    unsigned int word_len = 0;
    for (unsigned int i = 0; i < max_word_len; i++) {{
        unsigned char c = base_words_in[word_idx * max_word_len + i];
        if (c == 0) break;
        word[i] = c;
        word_len++;
    }}

    // Load rule (fixed length byte array)
    unsigned int rule_start = rule_batch_idx * MAX_RULE_LEN;
    unsigned char rule_str[MAX_RULE_LEN];
    unsigned int rule_len = 0;
    for (unsigned int i = 0; i < MAX_RULE_LEN; i++) {{
        unsigned char c = rules_in[rule_start + i];
        if (c == 0) break;
        rule_str[i] = c;
        rule_len++;
    }}

    unsigned char result_temp[MAX_OUTPUT_LEN];
    int out_len = 0;
    int changed = 0;
    apply_hashcat_rule(word, word_len, rule_str, rule_len, result_temp, &out_len, &changed);

    if (changed > 0 && out_len > 0) {{
        unsigned int word_hash = fnv1a_hash_32(result_temp, out_len);

        unsigned int global_map_index = (word_hash >> 5) & GLOBAL_HASH_MAP_MASK;
        unsigned int bit_index = word_hash & 31;
        unsigned int check_bit = (1U << bit_index);
        __global unsigned int* global_map_ptr = &global_hash_map[global_map_index];
        unsigned int current_global_word = *global_map_ptr;

        if (!(current_global_word & check_bit)) {{
            atomic_or(global_map_ptr, check_bit);
            atomic_inc(&rule_uniqueness_counts[rule_batch_idx]);

            unsigned int cracked_map_index = (word_hash >> 5) & CRACKED_HASH_MAP_MASK;
            __global const unsigned int* cracked_map_ptr = &cracked_hash_map[cracked_map_index];
            unsigned int current_cracked_word = *cracked_map_ptr;

            if (current_cracked_word & check_bit) {{
                atomic_inc(&rule_effectiveness_counts[rule_batch_idx]);
            }}
        }}
    }}
}}

// ----------------------------------------------------------------------------
// Hash map initialisation kernel (unchanged)
// ----------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void hash_map_init_kernel(
    __global unsigned int* hash_map,
    __global const unsigned int* hashes,
    const unsigned int num_hashes,
    const unsigned int map_mask)
{{
    unsigned int global_id = get_global_id(0);
    if (global_id >= num_hashes) return;

    unsigned int word_hash = hashes[global_id];
    unsigned int map_index = (word_hash >> 5) & map_mask;
    unsigned int bit_index = word_hash & 31;
    unsigned int set_bit = (1U << bit_index);
    atomic_or(&hash_map[map_index], set_bit);
}}
"""

# ====================================================================
# --- EXHAUSTIVE RANKING (Legacy v3.2) ---
# ====================================================================
def rank_rules_exhaustive(wordlist_path, rules_path, cracked_list_path, ranking_output_path, top_k,
                          words_per_gpu_batch=None, global_hash_map_bits=None, cracked_hash_map_bits=None,
                          preset=None, device_id=None):
    """Original exhaustive ranking algorithm (v3.2) with full rule support."""
    start_time = time()

    # Load data
    total_words = estimate_word_count(wordlist_path)
    rules_list = load_rules(rules_path)
    total_rules = len(rules_list)
    setup_interrupt_handler(rules_list, ranking_output_path, top_k)

    cracked_hashes_np = load_cracked_hashes(cracked_list_path, MAX_WORD_LEN)
    cracked_hashes_count = len(cracked_hashes_np)

    print(f"\n{blue('Dataset Summary:')}")
    print(f"   {bold('Words:')} {cyan(f'{total_words:,}')}")
    print(f"   {bold('Rules:')} {cyan(f'{total_rules:,}')}")
    print(f"   {bold('Cracked hashes:')} {cyan(f'{cracked_hashes_count:,}')}")

    # OpenCL init
    try:
        if device_id is not None:
            platform, device = select_platform_and_device(device_id)
        else:
            platform, device = select_platform_and_device()
        context = cl.Context([device])
        queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        total_vram, available_vram = get_gpu_memory_info(device)
        print(f"\n{green('GPU:')} {cyan(device.name.strip())}")
        print(f"{blue('Platform:')} {cyan(platform.name.strip())}")
        print(f"{blue('Total VRAM:')} {cyan(f'{total_vram / (1024**3):.1f} GB')}")
        print(f"{blue('Available VRAM:')} {cyan(f'{available_vram / (1024**3):.1f} GB')}")

        if preset:
            recommendations, recommended_preset = get_recommended_parameters(device, total_words, cracked_hashes_count)
            if preset == "recommend":
                preset = recommended_preset
            if preset in recommendations:
                preset_config = recommendations[preset]
                print(f"{blue('Using')} {cyan(preset_config['description'])}")
                words_per_gpu_batch = preset_config['batch_size']
                global_hash_map_bits = preset_config['global_bits']
                cracked_hash_map_bits = preset_config['cracked_bits']
            else:
                print(f"{red('Unknown preset:')} {cyan(preset)}")
                return

        if words_per_gpu_batch is None or global_hash_map_bits is None or cracked_hash_map_bits is None:
            words_per_gpu_batch, global_hash_map_bits, cracked_hash_map_bits = calculate_optimal_parameters_large_rules(
                available_vram, total_words, cracked_hashes_count, total_rules)

        GLOBAL_HASH_MAP_WORDS = 1 << (global_hash_map_bits - 5)
        GLOBAL_HASH_MAP_MASK = (1 << (global_hash_map_bits - 5)) - 1
        CRACKED_HASH_MAP_WORDS = 1 << (cracked_hash_map_bits - 5)
        CRACKED_HASH_MAP_MASK = (1 << (cracked_hash_map_bits - 5)) - 1

        KERNEL_SOURCE = get_kernel_source(global_hash_map_bits, cracked_hash_map_bits)
        prg = cl.Program(context, KERNEL_SOURCE).build()
        kernel_ranker = prg.ranker_kernel
        kernel_init = prg.hash_map_init_kernel

    except Exception as e:
        print(f"{red('OpenCL initialization failed:')} {e}")
        return

    # Load word batches
    print(f"{blue('Loading wordlist...')}")
    word_batches = []
    word_iter = optimized_wordlist_iterator(wordlist_path, MAX_WORD_LEN, words_per_gpu_batch)
    for words_np, hashes_np, cnt in word_iter:
        word_batches.append((words_np, hashes_np, cnt))
    total_word_batches = len(word_batches)
    print(f"{green('Loaded')} {cyan(f'{total_word_batches}')} word batches")

    # Pre‑encode all rules into a single contiguous 2D array (total_rules × MAX_RULE_LEN).
    # A 2D ndarray allows filling rules_batch_np with one slice instead of a Python for-loop.
    print(f"{blue('Encoding rules...')}")
    encoded_rules_2d = np.zeros((total_rules, MAX_RULE_LEN), dtype=np.uint8)
    for i, rule in enumerate(rules_list):
        rb = rule['rule_data'].encode('latin-1')
        encoded_rules_2d[i, :len(rb)] = np.frombuffer(rb, dtype=np.uint8)

    # Split rules into batches
    rule_batch_starts = list(range(0, total_rules, MAX_RULES_IN_BATCH))
    total_rule_batches = len(rule_batch_starts)
    print(f"{blue('Processing configuration:')}")
    print(f"   {blue('Words per batch:')} {cyan(f'{words_per_gpu_batch:,}')}")
    print(f"   {blue('Rules per batch:')} {cyan(f'{MAX_RULES_IN_BATCH:,}')}")
    print(f"   {blue('Total rule batches:')} {cyan(f'{total_rule_batches:,}')}")

    # Allocate GPU buffers
    mf = cl.mem_flags
    words_buffer_size = words_per_gpu_batch * MAX_WORD_LEN * np.uint8().itemsize
    hashes_buffer_size = words_per_gpu_batch * np.uint32().itemsize
    rules_buffer_size = MAX_RULES_IN_BATCH * MAX_RULE_LEN * np.uint8().itemsize
    counters_size = MAX_RULES_IN_BATCH * np.uint32().itemsize
    global_map_bytes = GLOBAL_HASH_MAP_WORDS * np.uint32(4)
    cracked_map_bytes = CRACKED_HASH_MAP_WORDS * np.uint32(4)

    try:
        base_words_g = cl.Buffer(context, mf.READ_ONLY, words_buffer_size)
        base_hashes_g = cl.Buffer(context, mf.READ_ONLY, hashes_buffer_size)
        rules_g = cl.Buffer(context, mf.READ_ONLY, rules_buffer_size)
        global_hash_map_g = cl.Buffer(context, mf.READ_WRITE, global_map_bytes)
        cracked_hash_map_g = cl.Buffer(context, mf.READ_ONLY, cracked_map_bytes)
        rule_uniqueness_g = cl.Buffer(context, mf.READ_WRITE, counters_size)
        rule_effectiveness_g = cl.Buffer(context, mf.READ_WRITE, counters_size)
        # Pre-allocated reusable buffers – avoids per-iteration allocations inside the hot loop
        indices_np    = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
        indices_g     = cl.Buffer(context, mf.READ_ONLY, counters_size)
        rules_batch_np = np.zeros((MAX_RULES_IN_BATCH, MAX_RULE_LEN), dtype=np.uint8)
        if cracked_hashes_np.size > 0:
            cracked_temp_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cracked_hashes_np)
    except cl.MemoryError:
        print(f"{red('GPU memory allocation failed')}")
        return

    # Populate cracked hash map
    if cracked_hashes_np.size > 0:
        print(f"{blue('Initialising cracked hash map...')}")
        global_size_init = (int(math.ceil(cracked_hashes_np.size / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
        kernel_init(queue, global_size_init, (LOCAL_WORK_SIZE,),
                    cracked_hash_map_g, cracked_temp_g,
                    np.uint32(cracked_hashes_np.size), np.uint32(CRACKED_HASH_MAP_MASK)).wait()
        print(f"{green('Cracked hash map ready')}")

    # Processing loop
    words_processed_total = 0
    total_unique_found = 0
    total_cracked_found = 0
    mapped_uniqueness = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
    mapped_effectiveness = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)

    word_pbar = tqdm(total=total_words, desc="Processing words", unit=" words",
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                     position=0)
    rule_pbar = tqdm(total=total_rule_batches, desc="Rule batches", unit=" batches",
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                     position=1)

    for word_batch_idx, (words_np, hashes_np, num_words) in enumerate(word_batches):
        if interrupted:
            break

        # Upload word batch
        cl.enqueue_copy(queue, base_words_g, words_np)
        cl.enqueue_copy(queue, base_hashes_g, hashes_np).wait()

        # Initialise global hash map for this batch
        cl.enqueue_fill_buffer(queue, global_hash_map_g, np.uint32(0), 0, global_map_bytes).wait()
        global_size_init = (int(math.ceil(num_words / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
        kernel_init(queue, global_size_init, (LOCAL_WORK_SIZE,),
                    global_hash_map_g, base_hashes_g,
                    np.uint32(num_words), np.uint32(GLOBAL_HASH_MAP_MASK)).wait()

        # Process all rule batches for this word batch
        for rule_batch_idx, start_idx in enumerate(rule_batch_starts):
            if interrupted:
                break

            end_idx = min(start_idx + MAX_RULES_IN_BATCH, total_rules)
            num_rules = end_idx - start_idx
            if num_rules == 0:
                continue

            # Fill rules buffer via 2D slice – no Python for-loop, no per-iteration np.zeros
            rules_batch_np[:num_rules]  = encoded_rules_2d[start_idx:end_idx]
            rules_batch_np[num_rules:] = 0   # zero-pad remaining slots

            cl.enqueue_copy(queue, rules_g, rules_batch_np).wait()
            cl.enqueue_fill_buffer(queue, rule_uniqueness_g,   np.uint32(0), 0, counters_size)
            cl.enqueue_fill_buffer(queue, rule_effectiveness_g, np.uint32(0), 0, counters_size)

            # Upload indices via pre-allocated buffer (no new cl.Buffer each iteration)
            indices_np[:num_rules] = np.arange(num_rules, dtype=np.uint32)
            cl.enqueue_copy(queue, indices_g, indices_np).wait()

            global_size_aligned = (int(math.ceil(num_words * num_rules / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
            kernel_ranker(queue, global_size_aligned, (LOCAL_WORK_SIZE,),
                         base_words_g, rules_g,
                         indices_g,
                         global_hash_map_g, cracked_hash_map_g,
                         rule_uniqueness_g, rule_effectiveness_g,
                         np.uint32(num_words), np.uint32(num_rules),
                         np.uint32(MAX_WORD_LEN), np.uint32(MAX_OUTPUT_LEN)).wait()

            cl.enqueue_copy(queue, mapped_uniqueness,   rule_uniqueness_g).wait()
            cl.enqueue_copy(queue, mapped_effectiveness, rule_effectiveness_g).wait()

            # Vectorised batch totals; per-rule dict update still requires a loop
            u_arr = mapped_uniqueness[:num_rules].astype(np.int64)
            e_arr = mapped_effectiveness[:num_rules].astype(np.int64)
            for i in range(num_rules):
                rules_list[start_idx + i]['uniqueness_score']   += int(u_arr[i])
                rules_list[start_idx + i]['effectiveness_score'] += int(e_arr[i])
            batch_unique  = int(np.sum(u_arr))
            batch_cracked = int(np.sum(e_arr))

            total_unique_found += batch_unique
            total_cracked_found += batch_cracked
            words_processed_total += num_words
            update_progress_stats(words_processed_total, total_unique_found, total_cracked_found)

            # Update progress bars
            word_pbar.n = min(int(words_processed_total), total_words)
            word_pbar.set_description(f"Words: {words_processed_total:,}/{total_words:,} | Unique: {total_unique_found:,} | Cracked: {total_cracked_found:,}")
            rule_pbar.update(1)
            rule_pbar.set_description(f"Rules: {rule_batch_idx+1}/{total_rule_batches} (Word: {word_batch_idx+1}/{total_word_batches})")

        word_pbar.update(num_words)
        rule_pbar.n = 0
        rule_pbar.refresh()

    word_pbar.close()
    rule_pbar.close()

    if interrupted:
        print(f"\n{yellow('Processing interrupted, results saved.')}")
        return

    end_time = time()
    print(f"\n{green('=' * 60)}")
    print(f"{bold('Exhaustive Ranking Complete')}")
    print(f"{green('=' * 60)}")
    print(f"{blue('Total Words Processed:')} {cyan(f'{words_processed_total:,}')}")
    print(f"{blue('Total Unique Words:')} {cyan(f'{total_unique_found:,}')}")
    print(f"{blue('Total Cracks Found:')} {cyan(f'{total_cracked_found:,}')}")
    print(f"{blue('Execution Time:')} {cyan(f'{end_time - start_time:.2f} s')}")

    csv_path = save_ranking_data(rules_list, ranking_output_path, legacy=True)
    if top_k > 0:
        optimized_path = os.path.splitext(ranking_output_path)[0] + "_optimized.rule"
        load_and_save_optimized_rules(csv_path, optimized_path, top_k)

# ====================================================================
# --- MULTI‑PASS MAB WITH EARLY ELIMINATION (v4.0) ---
# ====================================================================
class MultiPassMAB:
    """Multi-Armed Bandit with early elimination for large rule sets.

    Optimised v5.1 – all hot paths are vectorised with NumPy:
      • select_rules  : single np.random.beta call over the full candidate array
      • update        : numpy advanced indexing instead of Python for-loop
      • _eliminate    : boolean-mask operations, no Python per-rule loops
      • get_statistics: np.sum instead of generator expressions
      • active_rules  : cached numpy array rebuilt only on change (_active_dirty flag)
    """

    def __init__(self, rules_list, exploration_factor=2.0, final_trials=50,
                 screening_trials=5, zero_success_elimination=True,
                 batch_size_words=150000):
        self.all_rules = rules_list
        self.num_rules = len(rules_list)
        self.successes = np.ones(self.num_rules, dtype=np.float32)
        self.failures = np.ones(self.num_rules, dtype=np.float32)
        self.trials = np.zeros(self.num_rules, dtype=np.uint32)
        self.words_processed = np.zeros(self.num_rules, dtype=np.uint64)
        self.zero_success_count = np.zeros(self.num_rules, dtype=np.uint32)
        self.exploration_factor = exploration_factor
        self.final_trials = final_trials
        self.screening_trials = screening_trials
        self.zero_success_elimination = zero_success_elimination
        self.batch_size_words = batch_size_words
        self.elimination_thresholds = {
            'phase1': {'min_trials': 5,   'min_success_rate': 0.0000001},
            'phase2': {'min_trials': 20,  'min_success_rate': 0.000001},
            'phase3': {'min_trials': 50,  'min_success_rate': 0.00001},
            'phase4': {'min_trials': 100, 'min_success_rate': 0.0001},
        }
        self.selection_count = np.zeros(self.num_rules, dtype=np.uint32)
        self.last_selected_iteration = np.zeros(self.num_rules, dtype=np.uint32)
        self.current_iteration = 0
        self.total_selections = 0
        self.total_updates = 0
        self.eliminated_rules = set()
        self.active_rules = set(range(self.num_rules))
        self.last_selected_batch = []
        self.elimination_stats = {
            'zero_success': 0,
            'low_success_rate': 0,
            'worse_than_threshold': 0,
            'total_eliminated': 0,
            'elimination_events': []
        }
        self.tested_rules = set()
        self.tested_rules_count = 0
        self.batch_zero_streaks = np.zeros(self.num_rules, dtype=np.uint32)

        # --- cache for active-rules numpy array ---
        self._active_array = np.arange(self.num_rules, dtype=np.int32)
        self._active_dirty = False   # starts clean because all rules are active

        print(f"{green('MAB INIT')}: Multi-Pass MAB with Early Elimination - {cyan(f'{self.num_rules:,}')} rules")
        print(f"{blue('MAB CONFIG')}: screening_trials={screening_trials}, final_trials={final_trials}, zero_elim={zero_success_elimination}")

    # ------------------------------------------------------------------
    # Internal helper: return (and cache) a numpy array of active rule ids
    # ------------------------------------------------------------------
    def _get_active_array(self) -> np.ndarray:
        if self._active_dirty:
            self._active_array = np.fromiter(self.active_rules, dtype=np.int32,
                                             count=len(self.active_rules))
            self._active_dirty = False
        return self._active_array

    # ------------------------------------------------------------------
    # Rule selection – fully vectorised Thompson sampling
    # ------------------------------------------------------------------
    def select_rules(self, batch_size, iteration=0):
        self.current_iteration = iteration
        active_arr = self._get_active_array()

        # ---- phase 1: fill with rules that still need screening ----
        trials_active = self.trials[active_arr]
        needs_mask = trials_active < self.screening_trials
        needs_arr = active_arr[needs_mask]

        if len(needs_arr) > 0:
            # Sort by (trials asc, selection_count desc) using a composite key
            sort_key = (self.trials[needs_arr].astype(np.int64) << 32
                        - self.selection_count[needs_arr].astype(np.int64))
            needs_arr = needs_arr[np.argsort(sort_key, kind='stable')]

        num_from_screening = min(batch_size, len(needs_arr))
        selected_arr = needs_arr[:num_from_screening].copy()

        # ---- phase 2: Thompson-sampling for remaining slots ----
        remaining = batch_size - num_from_screening
        if remaining > 0 and len(active_arr) > num_from_screening:
            # Exclude already-selected indices
            selected_set = set(selected_arr.tolist())
            avail_mask = np.array([idx not in selected_set for idx in active_arr], dtype=bool)
            available = active_arr[avail_mask]

            if len(available) > 0:
                alpha = self.successes[available]          # shape (n,)
                beta_v = self.failures[available]          # shape (n,)

                # Single vectorised call – the key optimisation vs original
                thompson = np.random.beta(alpha, beta_v)

                trials_needed = np.maximum(0, self.final_trials - self.trials[available]).astype(np.float32)
                trials_score = trials_needed / max(self.final_trials, 1)

                zero_pen = np.where(
                    (self.trials[available] >= self.screening_trials) & (self.successes[available] <= 1.0),
                    np.float32(-0.5), np.float32(0.0)
                )

                combined = trials_score * 10.0 + thompson * self.exploration_factor + zero_pen

                num_add = min(remaining, len(available))
                if num_add < len(available):
                    top_local = np.argpartition(-combined, num_add - 1)[:num_add]
                    top_local = top_local[np.argsort(-combined[top_local])]
                else:
                    top_local = np.argsort(-combined)
                selected_arr = np.concatenate([selected_arr, available[top_local]])

        selected = selected_arr.tolist()

        # ---- vectorised state update ----
        if len(selected_arr) > 0:
            self.trials[selected_arr] += 1
            self.selection_count[selected_arr] += 1
            self.last_selected_iteration[selected_arr] = self.current_iteration

        self.total_selections += 1

        new_tested = set(selected) - self.tested_rules
        if new_tested:
            self.tested_rules.update(new_tested)
            self.tested_rules_count = len(self.tested_rules)

        self.last_selected_batch = selected
        return selected

    # ------------------------------------------------------------------
    # Bandit update – vectorised numpy advanced indexing
    # ------------------------------------------------------------------
    def update(self, selected_indices, successes_array, words_tested):
        self.total_updates += 1

        sel = np.asarray(selected_indices, dtype=np.int32)
        if len(sel) == 0:
            self._eliminate_low_performers()
            self.exploration_factor *= 0.9999
            return

        n = len(sel)
        succ_raw = np.maximum(0, np.asarray(successes_array[:n], dtype=np.float32))
        fail_raw = np.maximum(0, words_tested - succ_raw)

        # Filter to active rules only (vectorised membership test via cached array)
        active_arr = self._get_active_array()
        active_set_arr = np.zeros(self.num_rules, dtype=bool)
        active_set_arr[active_arr] = True
        valid_mask = active_set_arr[sel]
        valid = sel[valid_mask]
        succ_v = succ_raw[valid_mask]
        fail_v = fail_raw[valid_mask]

        if len(valid) == 0:
            self._eliminate_low_performers()
            self.exploration_factor *= 0.9999
            return

        # Scale down for very large batches
        if words_tested > 1_000_000:
            scale = np.float32(1_000_000.0 / words_tested)
            succ_v = succ_v * scale
            fail_v = fail_v * scale

        # Core updates – all vectorised
        self.successes[valid] += succ_v
        self.failures[valid] += fail_v
        self.words_processed[valid] += words_tested

        # Zero-streak tracking
        zero_mask = succ_v == 0
        if np.any(zero_mask):
            self.batch_zero_streaks[valid[zero_mask]] += 1
        if np.any(~zero_mask):
            self.batch_zero_streaks[valid[~zero_mask]] = 0

        # zero_success_count
        zs_mask = self.successes[valid] <= 1.0
        if np.any(zs_mask):
            self.zero_success_count[valid[zs_mask]] += 1

        self._eliminate_low_performers()
        self.exploration_factor *= 0.9999

    # ------------------------------------------------------------------
    # Elimination – fully vectorised boolean-mask approach
    # ------------------------------------------------------------------
    def _eliminate_low_performers(self):
        if len(self.active_rules) < 100:
            return

        active_arr = self._get_active_array()
        n = len(active_arr)
        to_elim = np.zeros(n, dtype=bool)

        trials_a = self.trials[active_arr]
        succ_a  = self.successes[active_arr] - 1.0
        fail_a  = self.failures[active_arr]  - 1.0
        total_a = succ_a + fail_a
        # Guard against division by zero
        rate_a = np.where(total_a > 0, succ_a / np.maximum(total_a, 1e-10), np.float32(0.0))

        # Strategy 1 – zero successes after screening
        if self.zero_success_elimination:
            s1 = (trials_a >= self.screening_trials) & (succ_a <= 0.0)
            self.elimination_stats['zero_success'] += int(np.sum(s1 & ~to_elim))
            to_elim |= s1

        # Strategy 2 – phase-based success-rate thresholds
        for cfg in self.elimination_thresholds.values():
            s2 = ((trials_a >= cfg['min_trials'])
                  & (total_a > 0)
                  & (rate_a < cfg['min_success_rate'])
                  & ~to_elim)
            self.elimination_stats['low_success_rate'] += int(np.sum(s2))
            to_elim |= s2

        # Strategy 3 – far worse than top-100 average (every 25 updates)
        if len(self.active_rules) > 500 and self.total_updates % 25 == 0:
            sufficient = (~to_elim) & (trials_a >= self.screening_trials) & (total_a > 0)
            n_suf = int(np.sum(sufficient))
            if n_suf >= 100:
                rates_v = rate_a[sufficient]
                # np.partition is O(n) – much faster than full sort
                k = min(100, n_suf) - 1
                top_100_avg = float(np.mean(np.partition(rates_v, -k)[-k:]))
                threshold = top_100_avg / 1000.0
                s3 = np.zeros(n, dtype=bool)
                s3[sufficient] = rates_v < threshold
                s3 &= ~to_elim
                self.elimination_stats['worse_than_threshold'] += int(np.sum(s3))
                to_elim |= s3

        # Strategy 4 – consecutive zero batches
        if self.zero_success_elimination:
            s4 = ((~to_elim)
                  & (trials_a >= self.screening_trials)
                  & (self.batch_zero_streaks[active_arr] >= 3))
            self.elimination_stats['zero_success'] += int(np.sum(s4))
            to_elim |= s4

        if not np.any(to_elim):
            return

        elim_arr = active_arr[to_elim]
        max_eliminate = min(5000, len(elim_arr))
        elim_arr = elim_arr[:max_eliminate]
        elim_set = set(elim_arr.tolist())

        self.eliminated_rules.update(elim_set)
        self.active_rules -= elim_set
        self.tested_rules -= elim_set
        self.tested_rules_count = len(self.tested_rules)
        self._active_dirty = True   # cache must be rebuilt

        cnt = len(elim_arr)
        self.elimination_stats['total_eliminated'] += cnt
        if cnt >= 1000:
            tqdm.write(f"\n{yellow('MASS ELIMINATION')}: Removed {cyan(f'{cnt:,}')} rules, "
                       f"active now {cyan(f'{len(self.active_rules):,}')}")

    def get_statistics(self):
        active_arr = self._get_active_array()
        if len(active_arr) > 0:
            trials_a = self.trials[active_arr]
            avg_trials = float(np.mean(trials_a))
            avg_words  = float(np.mean(self.words_processed[active_arr]))
            succ_a  = self.successes[active_arr] - 1.0
            fail_a  = self.failures[active_arr]  - 1.0
            total_a = succ_a + fail_a
            mask = total_a > 0
            avg_rate = float(np.mean(succ_a[mask] / total_a[mask])) if np.any(mask) else 0.0
            need_screening = int(np.sum(trials_a < self.screening_trials))
            need_final     = int(np.sum(trials_a < self.final_trials))
        else:
            avg_trials = avg_words = avg_rate = 0.0
            need_screening = need_final = 0
        eliminated_pct = (len(self.eliminated_rules) / self.num_rules) * 100 if self.num_rules else 0.0
        return {
            'total_rules': self.num_rules,
            'active_rules': len(self.active_rules),
            'tested_rules': len(self.tested_rules),
            'eliminated_rules': len(self.eliminated_rules),
            'eliminated_percentage': eliminated_pct,
            'rules_needing_screening': need_screening,
            'rules_needing_final': need_final,
            'avg_trials_per_rule': float(avg_trials),
            'avg_words_processed_per_rule': float(avg_words),
            'avg_success_rate': float(avg_rate),
            'exploration_factor': float(self.exploration_factor),
            'total_selections': int(self.total_selections),
            'total_updates': int(self.total_updates),
            'elimination_stats': self.elimination_stats.copy()
        }

    def get_top_rules(self, n=100):
        if not self.active_rules:
            return []
        active = self._get_active_array()
        total_succ = self.successes[active] - 1
        total_fail = self.failures[active] - 1
        total_tested = total_succ + total_fail
        sufficient = (self.trials[active] >= self.screening_trials) & (total_tested > 0)
        if not np.any(sufficient):
            return []
        probs = np.zeros(len(active))
        probs[sufficient] = total_succ[sufficient] / total_tested[sufficient]
        valid_indices = np.where(sufficient)[0]
        top_k = min(n, len(valid_indices))
        if top_k == 0:
            return []
        top_local = valid_indices[np.argpartition(-probs[valid_indices], top_k)[:top_k]]
        top_indices = active[top_local]
        order = np.argsort(-probs[top_local])
        top_indices = top_indices[order]
        results = []
        for idx in top_indices[:n]:
            s = self.successes[idx] - 1
            f = self.failures[idx] - 1
            t = s + f
            results.append({
                'rule_id': idx,
                'rule_data': self.all_rules[idx]['rule_data'],
                'success_probability': s / t if t > 0 else 0.0,
                'trials': int(self.trials[idx]),
                'words_processed': int(self.words_processed[idx]),
                'selections': int(self.selection_count[idx]),
                'successes': int(s),
                'failures': int(f),
                'total_tested': int(t)
            })
        return results

# ====================================================================
# --- MAB RANKING FUNCTION (v4.0) ---
# ====================================================================
def rank_rules_mab(wordlist_path, rules_path, cracked_list_path, ranking_output_path, top_k,
                   words_per_gpu_batch=None, global_hash_map_bits=None, cracked_hash_map_bits=None,
                   preset=None, device_id=None, mab_exploration_factor=None, mab_final_trials=None,
                   mab_screening_trials=None, mab_zero_success_elimination=None):
    start_time = time()

    # Load data
    total_words = estimate_word_count(wordlist_path)
    rules_list = load_rules(rules_path)
    total_rules = len(rules_list)
    setup_interrupt_handler(rules_list, ranking_output_path, top_k)

    cracked_hashes_np = load_cracked_hashes(cracked_list_path, MAX_WORD_LEN)
    cracked_hashes_count = len(cracked_hashes_np)

    print(f"\n{blue('Dataset Summary:')}")
    print(f"   {bold('Words:')} {cyan(f'{total_words:,}')}")
    print(f"   {bold('Rules:')} {cyan(f'{total_rules:,}')}")
    print(f"   {bold('Cracked hashes:')} {cyan(f'{cracked_hashes_count:,}')}")

    # MAB initialisation
    exploration_factor = mab_exploration_factor if mab_exploration_factor is not None else 2.0
    final_trials = mab_final_trials if mab_final_trials is not None else 50
    screening_trials = mab_screening_trials if mab_screening_trials is not None else 5
    zero_elim = mab_zero_success_elimination if mab_zero_success_elimination is not None else True
    rule_bandit = MultiPassMAB(rules_list, exploration_factor, final_trials, screening_trials, zero_elim,
                               batch_size_words=words_per_gpu_batch or DEFAULT_WORDS_PER_GPU_BATCH)

    # OpenCL init
    try:
        if device_id is not None:
            platform, device = select_platform_and_device(device_id)
        else:
            platform, device = select_platform_and_device()
        context = cl.Context([device])
        queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        total_vram, available_vram = get_gpu_memory_info(device)
        print(f"\n{green('GPU:')} {cyan(device.name.strip())}")
        print(f"{blue('Platform:')} {cyan(platform.name.strip())}")
        print(f"{blue('Total VRAM:')} {cyan(f'{total_vram / (1024**3):.1f} GB')}")
        print(f"{blue('Available VRAM:')} {cyan(f'{available_vram / (1024**3):.1f} GB')}")

        if preset:
            recommendations, recommended_preset = get_recommended_parameters(device, total_words, cracked_hashes_count)
            if preset == "recommend":
                preset = recommended_preset
            if preset in recommendations:
                preset_config = recommendations[preset]
                print(f"{blue('Using')} {cyan(preset_config['description'])}")
                words_per_gpu_batch = preset_config['batch_size']
                global_hash_map_bits = preset_config['global_bits']
                cracked_hash_map_bits = preset_config['cracked_bits']
            else:
                print(f"{red('Unknown preset:')} {cyan(preset)}")
                return

        if words_per_gpu_batch is None or global_hash_map_bits is None or cracked_hash_map_bits is None:
            words_per_gpu_batch, global_hash_map_bits, cracked_hash_map_bits = calculate_optimal_parameters_large_rules(
                available_vram, total_words, cracked_hashes_count, total_rules)

        GLOBAL_HASH_MAP_WORDS = 1 << (global_hash_map_bits - 5)
        GLOBAL_HASH_MAP_MASK = (1 << (global_hash_map_bits - 5)) - 1
        CRACKED_HASH_MAP_WORDS = 1 << (cracked_hash_map_bits - 5)
        CRACKED_HASH_MAP_MASK = (1 << (cracked_hash_map_bits - 5)) - 1

        KERNEL_SOURCE = get_kernel_source(global_hash_map_bits, cracked_hash_map_bits)
        prg = cl.Program(context, KERNEL_SOURCE).build()
        kernel_ranker = prg.ranker_kernel
        kernel_init = prg.hash_map_init_kernel

    except Exception as e:
        print(f"{red('OpenCL initialization failed:')} {e}")
        return

    # Load word batches
    print(f"{blue('Loading wordlist...')}")
    word_batches = []
    word_iter = optimized_wordlist_iterator(wordlist_path, MAX_WORD_LEN, words_per_gpu_batch)
    for words_np, hashes_np, cnt in word_iter:
        word_batches.append((words_np, hashes_np, cnt))
    total_word_batches = len(word_batches)
    print(f"{green('Loaded')} {cyan(f'{total_word_batches}')} word batches")

    # Pre‑encode rules into fixed‑length byte arrays for GPU
    encoded_rules = [encode_rule_fixed(rule['rule_data'], rule['rule_id']) for rule in rules_list]

    # Allocate GPU buffers
    mf = cl.mem_flags
    words_buffer_size = words_per_gpu_batch * MAX_WORD_LEN * np.uint8().itemsize
    hashes_buffer_size = words_per_gpu_batch * np.uint32().itemsize
    rules_buffer_size = MAX_RULES_IN_BATCH * MAX_RULE_LEN * np.uint8().itemsize
    counters_size = MAX_RULES_IN_BATCH * np.uint32().itemsize
    global_map_bytes = GLOBAL_HASH_MAP_WORDS * np.uint32(4)
    cracked_map_bytes = CRACKED_HASH_MAP_WORDS * np.uint32(4)

    try:
        base_words_g = cl.Buffer(context, mf.READ_ONLY, words_buffer_size)
        base_hashes_g = cl.Buffer(context, mf.READ_ONLY, hashes_buffer_size)
        rules_g = cl.Buffer(context, mf.READ_ONLY, rules_buffer_size)
        global_hash_map_g = cl.Buffer(context, mf.READ_WRITE, global_map_bytes)
        cracked_hash_map_g = cl.Buffer(context, mf.READ_ONLY, cracked_map_bytes)
        rule_uniqueness_g = cl.Buffer(context, mf.READ_WRITE, counters_size)
        rule_effectiveness_g = cl.Buffer(context, mf.READ_WRITE, counters_size)
        # Pre-allocated indices buffer – reused every iteration to avoid per-iteration cl.Buffer allocation
        indices_np = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
        indices_g  = cl.Buffer(context, mf.READ_ONLY, counters_size)
        if cracked_hashes_np.size > 0:
            cracked_temp_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cracked_hashes_np)
    except cl.MemoryError:
        print(f"{red('GPU memory allocation failed')}")
        return

    # Populate cracked hash map
    if cracked_hashes_np.size > 0:
        print(f"{blue('Initialising cracked hash map...')}")
        global_size_init = (int(math.ceil(cracked_hashes_np.size / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
        kernel_init(queue, global_size_init, (LOCAL_WORK_SIZE,),
                    cracked_hash_map_g, cracked_temp_g,
                    np.uint32(cracked_hashes_np.size), np.uint32(CRACKED_HASH_MAP_MASK)).wait()
        print(f"{green('Cracked hash map ready')}")

    # Main MAB loop
    words_processed_total = 0
    total_unique_found = 0
    total_cracked_found = 0
    mapped_uniqueness = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
    mapped_effectiveness = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)

    # --- Progress bars: separate for screening and deep testing ---
    # Each outer MAB iteration processes all word batches, and pbar.update(1) fires once
    # per word-batch inside the inner loop – so multiply by total_word_batches.
    total_screening_iters = int(math.ceil(total_rules * screening_trials / MAX_RULES_IN_BATCH)) * total_word_batches
    screening_pbar = tqdm(total=total_screening_iters, desc="SCREEN Phase", unit="iter",
                          bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                          position=0)

    iteration = 0
    phase = "SCREENING"
    screening_complete = False
    deep_testing_complete = False
    deep_pbar = None  # Will be created later

    while not interrupted and not deep_testing_complete:
        if phase == "SCREENING":
            # Check if all active rules have reached screening_trials
            if rule_bandit.active_rules:
                active_arr = rule_bandit._get_active_array()
                min_trials = int(np.min(rule_bandit.trials[active_arr]))
                if min_trials >= screening_trials:
                    screening_complete = True
                    phase = "DEEP_TESTING"
                    screening_pbar.close()
                    # Vectorised: compute remaining trials needed for each survivor
                    remaining_trials = np.maximum(0, final_trials - rule_bandit.trials[active_arr])
                    needed = int(np.sum(remaining_trials))
                    if needed == 0:
                        deep_testing_complete = True
                        break
                    total_deep_iters = int(math.ceil(needed / MAX_RULES_IN_BATCH)) * total_word_batches
                    deep_pbar = tqdm(total=total_deep_iters, desc="DEEP Phase", unit="iter",
                                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                                     position=0)
                    print(f"\n{green('SCREENING PHASE COMPLETE')}: {cyan(f'{len(rule_bandit.active_rules):,}')} survivors")
                    print(f"{blue('Deep testing requires')} {cyan(f'{needed:,}')} {bold('additional trials')} ({cyan(f'{total_deep_iters}')} iterations)")
                    continue

        # Process a full pass through the wordlist
        for word_batch_idx, (words_np, hashes_np, num_words) in enumerate(word_batches):
            if interrupted:
                break

            # Upload word batch
            cl.enqueue_copy(queue, base_words_g, words_np)
            cl.enqueue_copy(queue, base_hashes_g, hashes_np).wait()

            # Initialise global hash map for this batch
            cl.enqueue_fill_buffer(queue, global_hash_map_g, np.uint32(0), 0, global_map_bytes).wait()
            global_size_init = (int(math.ceil(num_words / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
            kernel_init(queue, global_size_init, (LOCAL_WORK_SIZE,),
                        global_hash_map_g, base_hashes_g,
                        np.uint32(num_words), np.uint32(GLOBAL_HASH_MAP_MASK)).wait()

            # Select rules to test in this iteration
            selected_indices = rule_bandit.select_rules(batch_size=MAX_RULES_IN_BATCH, iteration=iteration)
            if not selected_indices:
                # Fallback: select rules with lowest trials (vectorised)
                fallback_arr = rule_bandit._get_active_array()
                if len(fallback_arr) == 0:
                    break
                order = np.argsort(rule_bandit.trials[fallback_arr])
                selected_indices = fallback_arr[order[:MAX_RULES_IN_BATCH]].tolist()
            num_rules = len(selected_indices)

            # Prepare rules buffer – vectorised row copy into a pre-allocated array
            rules_batch_np = np.zeros((MAX_RULES_IN_BATCH, MAX_RULE_LEN), dtype=np.uint8)
            sel_slice = selected_indices[:num_rules]
            for i, idx in enumerate(sel_slice):
                rules_batch_np[i] = encoded_rules[idx]

            # Split the rules batch into sub-dispatches to avoid OUT_OF_RESOURCES.
            # A single dispatch of (num_words * num_rules) work-items can exceed the
            # GPU driver's per-submission limit or trigger the watchdog timer (TDR/DRM)
            # when both dimensions are large (e.g. 150k words × 1024 rules = 153M items).
            # MAX_DISPATCH_ITEMS is set at the top of this file and can be lowered if needed.
            rules_per_sub = max(1, min(MAX_RULES_IN_BATCH,
                                       MAX_DISPATCH_ITEMS // max(num_words, 1)))

            # Accumulators across sub-batches (indexed by position in sel_slice)
            u_arr = np.zeros(num_rules, dtype=np.int64)
            e_arr = np.zeros(num_rules, dtype=np.int64)

            for sub_start in range(0, num_rules, rules_per_sub):
                sub_end = min(sub_start + rules_per_sub, num_rules)
                sub_num = sub_end - sub_start

                # Upload this slice of the pre-built rules array
                sub_rules_np = np.zeros((MAX_RULES_IN_BATCH, MAX_RULE_LEN), dtype=np.uint8)
                sub_rules_np[:sub_num] = rules_batch_np[sub_start:sub_end]
                cl.enqueue_copy(queue, rules_g, sub_rules_np).wait()

                # Zero counters for this sub-batch (wait() required before kernel launch)
                cl.enqueue_fill_buffer(queue, rule_uniqueness_g,   np.uint32(0), 0, counters_size).wait()
                cl.enqueue_fill_buffer(queue, rule_effectiveness_g, np.uint32(0), 0, counters_size).wait()

                # Map kernel slot 0..sub_num-1 to rule positions
                indices_np[:sub_num] = np.arange(sub_num, dtype=np.uint32)
                cl.enqueue_copy(queue, indices_g, indices_np).wait()

                global_size_aligned = (int(math.ceil(num_words * sub_num / LOCAL_WORK_SIZE))
                                       * LOCAL_WORK_SIZE,)
                kernel_ranker(queue, global_size_aligned, (LOCAL_WORK_SIZE,),
                              base_words_g, rules_g,
                              indices_g,
                              global_hash_map_g, cracked_hash_map_g,
                              rule_uniqueness_g, rule_effectiveness_g,
                              np.uint32(num_words), np.uint32(sub_num),
                              np.uint32(MAX_WORD_LEN), np.uint32(MAX_OUTPUT_LEN)).wait()

                cl.enqueue_copy(queue, mapped_uniqueness,   rule_uniqueness_g).wait()
                cl.enqueue_copy(queue, mapped_effectiveness, rule_effectiveness_g).wait()

                u_arr[sub_start:sub_end] = mapped_uniqueness[:sub_num].astype(np.int64)
                e_arr[sub_start:sub_end] = mapped_effectiveness[:sub_num].astype(np.int64)

            # Update MAB with the number of successes (effectiveness counts)
            rule_bandit.update(selected_indices, e_arr[:num_rules].astype(np.uint32), num_words)
            for i, idx in enumerate(sel_slice):
                rule_bandit.all_rules[idx]['uniqueness_score']  += int(u_arr[i])
                rule_bandit.all_rules[idx]['effectiveness_score'] += int(e_arr[i])
                rule_bandit.all_rules[idx]['total_successes'] = rule_bandit.all_rules[idx].get('total_successes', 0) + int(e_arr[i])
                rule_bandit.all_rules[idx]['total_trials']    = rule_bandit.all_rules[idx].get('total_trials', 0)    + num_words
                rule_bandit.all_rules[idx]['times_tested']    = rule_bandit.all_rules[idx].get('times_tested', 0)    + 1
            batch_unique  = int(np.sum(u_arr))
            batch_cracked = int(np.sum(e_arr))

            total_unique_found += batch_unique
            total_cracked_found += batch_cracked
            words_processed_total += num_words
            update_progress_stats(words_processed_total, total_unique_found, total_cracked_found)

            # Update progress bar
            iteration += 1
            stats = rule_bandit.get_statistics()
            if phase == "SCREENING":
                need = stats['rules_needing_screening']
                screening_pbar.set_description(f"SCREEN | Active: {stats['active_rules']:,} | Need: {need:,} | Elim: {stats['eliminated_rules']:,}")
                screening_pbar.update(1)
            else:
                need = stats['rules_needing_final']
                if deep_pbar is not None:
                    deep_pbar.set_description(f"DEEP   | Active: {stats['active_rules']:,} | Need: {need:,} | Elim: {stats['eliminated_rules']:,}")
                    deep_pbar.update(1)

            # Check for completion of deep testing
            if phase == "DEEP_TESTING" and stats['rules_needing_final'] == 0:
                deep_testing_complete = True
                break

        if deep_testing_complete:
            break

    if deep_pbar is not None:
        deep_pbar.close()
    else:
        screening_pbar.close()

    if interrupted:
        print(f"\n{yellow('Processing interrupted, results saved.')}")
        return

    # Final statistics and saving
    end_time = time()
    final_stats = rule_bandit.get_statistics()
    top_rules = rule_bandit.get_top_rules(20)

    print(f"\n{green('=' * 80)}")
    print(f"{bold('MULTI-PASS MAB RANKING COMPLETE')}")
    print(f"{green('=' * 80)}")
    print(f"{blue('Total Words Processed:')} {cyan(f'{words_processed_total:,}')}")
    print(f"{blue('Total Unique Words:')} {cyan(f'{total_unique_found:,}')}")
    print(f"{blue('Total Cracks Found:')} {cyan(f'{total_cracked_found:,}')}")
    print(f"{blue('Execution Time:')} {cyan(f'{end_time - start_time:.2f} s')}")
    print(f"\n{blue('Early Elimination Summary:')}")
    print("   " + blue('Original rules:')  + " " + cyan(f"{final_stats['total_rules']:,}"))
    print("   " + blue('Surviving rules:') + " " + cyan(f"{final_stats['active_rules']:,}"))
    print("   " + blue('Eliminated rules:') + " " + cyan(f"{final_stats['eliminated_rules']:,}") + f" ({final_stats['eliminated_percentage']:.1f}%)")
    print(f"\n{blue('Top 10 Surviving Rules:')}")
    for i, r in enumerate(top_rules[:10], 1):
        print(f"   {blue(f'{i:2}.')} {cyan(r['rule_data']):30} success={r['success_probability']:.6f} trials={r['trials']:,}")

    # Mark eliminated rules in the list
    for rule in rules_list:
        idx = rule['rule_id']
        if idx in rule_bandit.eliminated_rules:
            rule['eliminated'] = True
            if rule_bandit.successes[idx] <= 1.0:
                rule['eliminate_reason'] = 'zero_success'
            else:
                rule['eliminate_reason'] = 'low_success_rate'
        else:
            rule['eliminated'] = False
        rule['mab_trials'] = rule_bandit.trials[idx]
        rule['selections'] = rule_bandit.selection_count[idx]
        rule['mab_success_prob'] = (rule_bandit.successes[idx] - 1) / ((rule_bandit.successes[idx] + rule_bandit.failures[idx] - 2) + 1e-9)
        rule['combined_score'] = rule.get('effectiveness_score', 0) * 10 + rule.get('uniqueness_score', 0) + rule['mab_success_prob'] * 1000

    # Save results
    csv_path = save_ranking_data(rules_list, ranking_output_path, legacy=False)
    if top_k > 0:
        optimized_path = os.path.splitext(ranking_output_path)[0] + "_optimized.rule"
        load_and_save_optimized_rules(csv_path, optimized_path, top_k)

# ====================================================================
# --- MAIN ENTRY POINT ---
# ====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPU-Accelerated Hashcat Rule Ranking Tool")
    parser.add_argument('-w', '--wordlist', required=True, help='Path to base wordlist')
    parser.add_argument('-r', '--rules', required=True, help='Path to Hashcat rules file')
    parser.add_argument('-c', '--cracked', required=True, help='Path to cracked passwords list')
    parser.add_argument('-o', '--output', default='ranker_output.csv', help='Output CSV file')
    parser.add_argument('-k', '--topk', type=int, default=1000, help='Number of top rules to save')

    # Performance tuning
    parser.add_argument('--batch-size', type=int, help='Words per GPU batch')
    parser.add_argument('--global-bits', type=int, help='Global hash map bits')
    parser.add_argument('--cracked-bits', type=int, help='Cracked hash map bits')
    parser.add_argument('--preset', choices=['low_memory', 'medium_memory', 'high_memory', 'recommend'], help='Preset configuration')

    # MAB options
    parser.add_argument('--mab-exploration', type=float, default=2.0, help='MAB exploration factor')
    parser.add_argument('--mab-final-trials', type=int, default=50, help='Final trials for survivors')
    parser.add_argument('--mab-screening-trials', type=int, default=5, help='Trials before elimination')
    parser.add_argument('--mab-no-zero-eliminate', action='store_false', dest='mab_zero_success_elimination',
                        help='Disable zero‑success elimination')

    # Legacy mode
    parser.add_argument('--legacy', action='store_true', help='Run exhaustive (v3.2) mode instead of MAB')

    # Device selection
    parser.add_argument('--device', type=int, help='OpenCL device ID')
    parser.add_argument('--list-devices', action='store_true', help='List available devices and exit')

    args = parser.parse_args()

    if args.list_devices:
        list_platforms_and_devices()
        sys.exit(0)

    print(f"{green('=' * 80)}")
    print(f"{bold('HASHCAT RULE RANKER v5.0')}")
    if args.legacy:
        print(f"{bold('LEGACY MODE (v3.2) – Exhaustive Ranking')}")
    else:
        print(f"{bold('MULTI-PASS MAB MODE – Early Elimination')}")
    print(f"{green('=' * 80)}")
    print(f"{blue('GPU Rules:')} Full Hashcat rule set with {MAX_RULE_LEN}‑char support")
    print(f"{blue('Interrupt:')} Ctrl+C saves progress")
    print(f"{green('=' * 80)}")

    if args.legacy:
        rank_rules_exhaustive(
            wordlist_path=args.wordlist,
            rules_path=args.rules,
            cracked_list_path=args.cracked,
            ranking_output_path=args.output,
            top_k=args.topk,
            words_per_gpu_batch=args.batch_size,
            global_hash_map_bits=args.global_bits,
            cracked_hash_map_bits=args.cracked_bits,
            preset=args.preset,
            device_id=args.device
        )
    else:
        rank_rules_mab(
            wordlist_path=args.wordlist,
            rules_path=args.rules,
            cracked_list_path=args.cracked,
            ranking_output_path=args.output,
            top_k=args.topk,
            words_per_gpu_batch=args.batch_size,
            global_hash_map_bits=args.global_bits,
            cracked_hash_map_bits=args.cracked_bits,
            preset=args.preset,
            device_id=args.device,
            mab_exploration_factor=args.mab_exploration,
            mab_final_trials=args.mab_final_trials,
            mab_screening_trials=args.mab_screening_trials,
            mab_zero_success_elimination=args.mab_zero_success_elimination
        )
