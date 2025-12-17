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
from collections import deque
import gc

# ====================================================================
# --- RANKER v4.2: ULTRA-OPTIMIZED GPU RULE RANKING ---
# ====================================================================

# --- COLOR CODES FOR TERMINAL OUTPUT ---
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

# Color functions for easy use
def red(text): return f"{Colors.RED}{text}{Colors.END}"
def green(text): return f"{Colors.GREEN}{text}{Colors.END}"
def yellow(text): return f"{Colors.YELLOW}{text}{Colors.END}"
def blue(text): return f"{Colors.BLUE}{text}{Colors.END}"
def magenta(text): return f"{Colors.MAGENTA}{text}{Colors.END}"
def cyan(text): return f"{Colors.CYAN}{text}{Colors.END}"
def bold(text): return f"{Colors.BOLD}{text}{Colors.END}"
def underline(text): return f"{Colors.UNDERLINE}{text}{Colors.END}"

# --- WARNING FILTERS ---
warnings.filterwarnings("ignore", message="overflow encountered in scalar multiply")
warnings.filterwarnings("ignore", message="overflow encountered in scalar add")
warnings.filterwarnings("ignore", message="overflow encountered in uint_scalars")
try:
    warnings.filterwarnings("ignore", message="The 'device_offset' argument of enqueue_copy is deprecated")
    warnings.filterwarnings("ignore", category=cl.CompilerWarning)
except AttributeError:
    pass

# ====================================================================
# --- CONSTANTS ---
# ====================================================================
MAX_WORD_LEN = 256
MAX_OUTPUT_LEN = 512
MAX_RULE_ARGS = 4
MAX_RULES_IN_BATCH = 256  # Reduced from 1024 to prevent timeout
MAX_RULE_LEN = 16

# Performance tuning
DOUBLE_BUFFERING = True
UPDATE_FREQUENCY = 25  # Update progress every N batches

# Default values
DEFAULT_WORDS_PER_GPU_BATCH = 50000  # Reduced from 100000
DEFAULT_GLOBAL_HASH_MAP_BITS = 32
DEFAULT_CRACKED_HASH_MAP_BITS = 30

# VRAM usage thresholds
VRAM_SAFETY_MARGIN = 0.15
MIN_BATCH_SIZE = 10000
MIN_HASH_MAP_BITS = 28

# Memory reduction factors
MEMORY_REDUCTION_FACTOR = 0.7
MAX_ALLOCATION_RETRIES = 5

# Kernel timeout protection
KERNEL_TIMEOUT_MS = 30000  # 30 seconds max per kernel

# Global variables for interrupt handling
interrupted = False
current_rules_list = None
current_ranking_output_path = None
current_top_k = 0
words_processed_total = None
total_unique_found = None
total_cracked_found = None

# ====================================================================
# --- OPTIMIZED FILE LOADING FUNCTIONS ---
# ====================================================================

def estimate_word_count(path):
    """Fast word count estimation for large files without reading entire content"""
    print(f"{blue('📊')} {bold('Estimating words in:')} {path}...")
    
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
                
        print(f"{green('✅')} {bold('Estimated words:')} {cyan(f'{total_lines:,}')}")
        return total_lines
        
    except Exception as e:
        print(f"{yellow('⚠️')} {bold('Could not estimate word count:')} {e}")
        return 1000000

def fast_fnv1a_hash_32(data):
    """Optimized FNV-1a hash for bytes"""
    if isinstance(data, np.ndarray):
        hash_val = np.uint32(2166136261)
        for byte in data:
            hash_val ^= np.uint32(byte)
            hash_val *= np.uint32(16777619)
        return hash_val
    else:
        hash_val = 2166136261
        for byte in data:
            hash_val = (hash_val ^ byte) * 16777619 & 0xFFFFFFFF
        return hash_val

def bulk_hash_words(words_list):
    """Compute hashes for multiple words in bulk"""
    return [fast_fnv1a_hash_32(word) for word in words_list]

def optimized_wordlist_iterator(wordlist_path, max_len, batch_size):
    """Massively optimized wordlist loader using memory mapping"""
    print(f"{green('🚀')} {bold('Using optimized memory-mapped loader...')}")
    
    file_size = os.path.getsize(wordlist_path)
    print(f"{blue('📁')} {bold('File size:')} {cyan(f'{file_size / (1024**3):.2f} GB')}")
    
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
                
                with tqdm(total=file_size, desc="Loading wordlist", unit="B", unit_scale=True, 
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                    while pos < file_size and not interrupted:
                        end_pos = mm.find(b'\n', pos)
                        if end_pos == -1:
                            end_pos = file_size
                        
                        line = mm[pos:end_pos].strip()
                        line_len = len(line)
                        pos = end_pos + 1
                        pbar.update(pos - pbar.n)
                        
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
        print(f"{red('❌')} {bold('Error in optimized loader:')} {e}")
        raise
    
    load_time = time() - load_start
    print(f"{green('✅')} {bold('Optimized loading completed:')} {cyan(f'{total_words_loaded:,}')} {bold('words in')} {load_time:.2f}s "
          f"({total_words_loaded/load_time:,.0f} words/sec)")

# ====================================================================
# --- INTERRUPT HANDLER FUNCTIONS ---
# ====================================================================

def signal_handler(sig, frame):
    """Handle Ctrl+C interrupt signal"""
    global interrupted, current_rules_list, current_ranking_output_path, current_top_k
    global words_processed_total, total_unique_found, total_cracked_found
    
    print(f"\n{yellow('⚠️')} {bold('Interrupt received!')}")
    
    if interrupted:
        print(f"{red('❌')} {bold('Forced exit!')}")
        sys.exit(1)
        
    interrupted = True
    
    if current_rules_list is not None and current_ranking_output_path is not None:
        print(f"{blue('💾')} {bold('Saving current progress...')}")
        save_current_progress()
    else:
        print(f"{yellow('⚠️')} {bold('No data to save. Exiting...')}")
        sys.exit(1)

def save_current_progress():
    """Save current progress when interrupted"""
    global current_rules_list, current_ranking_output_path, current_top_k
    global words_processed_total, total_unique_found, total_cracked_found
    
    try:
        base_path = os.path.splitext(current_ranking_output_path)[0]
        intermediate_output_path = f"{base_path}_INTERRUPTED.csv"
        intermediate_optimized_path = f"{base_path}_INTERRUPTED.rule"
        
        if current_rules_list:
            print(f"{blue('💾')} {bold('Saving intermediate results to:')} {intermediate_output_path}")
            
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
            
            print(f"{green('✅')} {bold('Intermediate ranking data saved:')} {cyan(f'{len(ranked_rules):,}')} {bold('rules')}")
            
            if current_top_k > 0:
                print(f"{blue('💾')} {bold('Saving intermediate optimized rules to:')} {intermediate_optimized_path}")
                
                available_rules = len(ranked_rules)
                final_count = min(current_top_k, available_rules)
                
                with open(intermediate_optimized_path, 'w', newline='\n', encoding='utf-8') as f:
                    f.write(":\n")
                    for rule in ranked_rules[:final_count]:
                        f.write(f"{rule['rule_data']}\n")
                
                print(f"{green('✅')} {bold('Intermediate optimized rules saved:')} {cyan(f'{final_count:,}')} {bold('rules')}")
        
        if words_processed_total is not None:
            print(f"\n{green('=' * 60)}")
            print(f"{bold('📊 Progress Summary at Interruption')}")
            print(f"{green('=' * 60)}")
            print(f"{blue('📊')} {bold('Words Processed:')} {cyan(f'{int(words_processed_total):,}')}")
            if total_unique_found is not None:
                print(f"{blue('🎯')} {bold('Unique Words Generated:')} {cyan(f'{int(total_unique_found):,}')}")
            if total_cracked_found is not None:
                print(f"{blue('🔓')} {bold('True Cracks Found:')} {cyan(f'{int(total_cracked_found):,}')}")
            print(f"{green('=' * 60)}{Colors.END}\n")
            
        print(f"{green('✅')} {bold('Progress saved successfully.')}")
        
    except Exception as e:
        print(f"{red('❌')} {bold('Error saving intermediate progress:')} {e}")
    
    sys.exit(0)

def setup_interrupt_handler(rules_list, ranking_output_path, top_k):
    """Setup interrupt handler with current context"""
    global current_rules_list, current_ranking_output_path, current_top_k
    current_rules_list = rules_list
    current_ranking_output_path = ranking_output_path
    current_top_k = top_k
    
    signal.signal(signal.SIGINT, signal_handler)

def update_progress_stats(words_processed, unique_found, cracked_found):
    """Update progress statistics for interrupt handler"""
    global words_processed_total, total_unique_found, total_cracked_found
    words_processed_total = words_processed
    total_unique_found = unique_found
    total_cracked_found = cracked_found

# ====================================================================
# --- HELPER FUNCTIONS (OPTIMIZED) ---
# ====================================================================

def load_rules(path):
    """Loads Hashcat rules from file."""
    print(f"{blue('📊')} {bold('Loading rules from:')} {path}...")
    
    try:
        rules_size = os.path.getsize(path) / (1024 * 1024)
        if rules_size > 10:
            print(f"{yellow('📁')} {bold('Large rules file detected:')} {rules_size:.1f} MB")
    except OSError:
        rules_size = 0
    
    rules_list = []
    rule_id_counter = 0
    try:
        with open(path, 'r', encoding='latin-1') as f:
            for line in f:
                rule = line.strip()
                if not rule or rule.startswith('#'):
                    continue
                rules_list.append({'rule_data': rule, 'rule_id': rule_id_counter, 'uniqueness_score': 0, 'effectiveness_score': 0})
                rule_id_counter += 1
    except FileNotFoundError:
        print(f"{red('❌')} {bold('Error:')} Rules file not found at: {path}")
        exit(1)

    print(f"{green('✅')} {bold('Loaded')} {cyan(f'{len(rules_list):,}')} {bold('rules.')}")
    return rules_list

def load_cracked_hashes(path, max_len):
    """Loads a list of cracked passwords and returns their FNV-1a hashes."""
    print(f"{blue('📊')} {bold('Loading cracked list for effectiveness check from:')} {path}...")
    
    try:
        cracked_size = os.path.getsize(path) / (1024 * 1024)
        if cracked_size > 50:
            print(f"{yellow('📁')} {bold('Large cracked list detected:')} {cracked_size:.1f} MB - loading...")
    except OSError:
        cracked_size = 0
    
    cracked_hashes = []
    load_start = time()
    
    try:
        with open(path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                pos = 0
                file_size = len(mm)
                
                while pos < file_size:
                    end_pos = mm.find(b'\n', pos)
                    if end_pos == -1:
                        end_pos = file_size
                    
                    line = mm[pos:end_pos].strip()
                    pos = end_pos + 1
                    
                    if 1 <= len(line) <= max_len:
                        cracked_hashes.append(fast_fnv1a_hash_32(line))
    except FileNotFoundError:
        print(f"{yellow('⚠️')} {bold('Warning:')} Cracked list file not found at: {path}. Effectiveness scores will be zero.")
        return np.array([], dtype=np.uint32)

    load_time = time() - load_start
    unique_hashes = np.unique(np.array(cracked_hashes, dtype=np.uint32))
    
    if cracked_size > 50:
        print(f"{green('✅')} {bold('Cracked list loaded:')} {cyan(f'{len(unique_hashes):,}')} {bold('unique hashes in')} {load_time:.2f}s")
    else:
        print(f"{green('✅')} {bold('Loaded')} {cyan(f'{len(unique_hashes):,}')} {bold('unique cracked password hashes.')}")
        
    return unique_hashes

def encode_rule(rule_str, rule_id, max_args):
    """Encodes a rule as an array of uint32: [rule ID, arguments]"""
    rule_size_in_int = 2 + max_args
    encoded = np.zeros(rule_size_in_int, dtype=np.uint32)
    encoded[0] = np.uint32(rule_id)
    rule_chars = rule_str.encode('latin-1')
    args_int = 0
    
    for i, byte in enumerate(rule_chars[:4]):
        args_int |= (byte << (i * 8))
    
    encoded[1] = np.uint32(args_int)
    
    if len(rule_chars) > 4:
        args_int2 = 0
        for i, byte in enumerate(rule_chars[4:8]):
            args_int2 |= (byte << (i * 8))
        encoded[2] = np.uint32(args_int2)
    
    return encoded

def save_ranking_data(ranking_list, output_path):
    """Saves the scoring and ranking data to a separate CSV file."""
    ranking_output_path = output_path
    
    print(f"{blue('💾')} {bold('Saving rule ranking data to:')} {ranking_output_path}...")

    for rule in ranking_list:
        rule['combined_score'] = rule.get('effectiveness_score', 0) * 10 + rule.get('uniqueness_score', 0)

    ranked_rules = ranking_list
    ranked_rules.sort(key=lambda rule: rule['combined_score'], reverse=True)

    print(f"{blue('💾')} {bold('Saving ALL')} {cyan(f'{len(ranked_rules):,}')} {bold('rules')}")

    if not ranked_rules:
        print(f"{red('❌')} {bold('No rules to save. Ranking file not created.')}")
        return None

    try:
        with open(ranking_output_path, 'w', newline='', encoding='utf-8') as f:
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

        print(f"{green('✅')} {bold('Ranking data saved successfully to')} {ranking_output_path}.")
        return ranking_output_path
    except Exception as e:
        print(f"{red('❌')} {bold('Error while saving ranking data to CSV file:')} {e}")
        return None

def load_and_save_optimized_rules(csv_path, output_path, top_k):
    """Loads ranking data from a CSV, sorts, and saves the Top K rules."""
    if not csv_path:
        print(f"{yellow('⚠️')} {bold('Optimization skipped: Ranking CSV path is missing.')}")
        return

    print(f"{blue('🔧')} {bold('Loading ranking from CSV:')} {csv_path} {bold('and saving Top')} {cyan(f'{top_k}')} {bold('Optimized Rules to:')} {output_path}...")
    
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
        print(f"{red('❌')} {bold('Error: Ranking CSV file not found at:')} {csv_path}")
        return
    except Exception as e:
        print(f"{red('❌')} {bold('Error while reading CSV:')} {e}")
        return

    print(f"{blue('📊')} {bold('Loaded')} {cyan(f'{len(ranked_data):,}')} {bold('total rules from CSV')}")

    ranked_data.sort(key=lambda row: row['Combined_Score'], reverse=True)
    
    available_rules = len(ranked_data)
    if top_k > available_rules:
        print(f"{yellow('⚠️')}  {bold('Warning: Requested')} {cyan(f'{top_k:,}')} {bold('rules but only')} {cyan(f'{available_rules:,}')} {bold('available. Saving')} {cyan(f'{available_rules:,}')} {bold('rules.')}")
        final_optimized_list = ranked_data[:available_rules]
    else:
        final_optimized_list = ranked_data[:top_k]

    if not final_optimized_list:
        print(f"{red('❌')} {bold('No rules available after sorting/filtering. Optimized rule file not created.')}")
        return

    try:
        with open(output_path, 'w', newline='\n', encoding='utf-8') as f:
            f.write(":\n")
            for rule in final_optimized_list:
                f.write(f"{rule['Rule_Data']}\n")
        print(f"{green('✅')} {bold('Top')} {cyan(f'{len(final_optimized_list):,}')} {bold('optimized rules saved successfully to')} {output_path}.")
    except Exception as e:
        print(f"{red('❌')} {bold('Error while saving optimized rules to file:')} {e}")

# ====================================================================
# --- MEMORY MANAGEMENT FUNCTIONS ---
# ====================================================================

def get_gpu_memory_info(device):
    """Get total and available GPU memory in bytes"""
    try:
        total_memory = device.global_mem_size
        available_memory = int(total_memory * (1 - VRAM_SAFETY_MARGIN))
        return total_memory, available_memory
    except Exception as e:
        print(f"{yellow('⚠️')} {bold('Warning: Could not query GPU memory:')} {e}")
        return 8 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024

def calculate_optimal_parameters_large_rules(available_vram, total_words, cracked_hashes_count, total_rules, reduction_factor=1.0):
    """
    Calculate optimal parameters with consideration for large rule sets
    """
    print(f"{blue('🔧')} {bold('Calculating optimal parameters for')} {cyan(f'{available_vram / (1024**3):.1f} GB')} {bold('available VRAM')}")
    print(f"{blue('📊')} {bold('Dataset:')} {cyan(f'{total_words:,}')} {bold('words,')} {cyan(f'{total_rules:,}')} {bold('rules,')} {cyan(f'{cracked_hashes_count:,}')} {bold('cracked hashes')}")
    
    if reduction_factor < 1.0:
        print(f"{yellow('📉')} {bold('Applying memory reduction factor:')} {cyan(f'{reduction_factor:.2f}')}")
    
    available_vram = int(available_vram * reduction_factor)
    
    word_batch_bytes = MAX_WORD_LEN * np.uint8().itemsize
    hash_batch_bytes = np.uint32().itemsize
    rule_batch_bytes = MAX_RULES_IN_BATCH * (2 + MAX_RULE_ARGS) * np.uint32().itemsize
    counter_bytes = MAX_RULES_IN_BATCH * np.uint32().itemsize * 2
    
    base_memory = (
        (word_batch_bytes + hash_batch_bytes) * 2 +
        rule_batch_bytes + counter_bytes
    )
    
    if total_rules > 100000:
        suggested_batch_size = min(DEFAULT_WORDS_PER_GPU_BATCH, 25000)
    else:
        suggested_batch_size = DEFAULT_WORDS_PER_GPU_BATCH
    
    available_for_maps = available_vram - base_memory
    if available_for_maps <= 0:
        print(f"{yellow('⚠️')}  {bold('Warning: Limited VRAM, using minimal configuration')}")
        available_for_maps = available_vram * 0.5
    
    print(f"{blue('📊')} {bold('Available for hash maps:')} {cyan(f'{available_for_maps / (1024**3):.2f} GB')}")
    
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
    
    memory_per_word = (
        word_batch_bytes +
        hash_batch_bytes +
        (MAX_OUTPUT_LEN * np.uint8().itemsize) +
        (rule_batch_bytes / MAX_RULES_IN_BATCH)
    )
    
    max_batch_by_memory = int((available_vram - total_map_memory - base_memory) / memory_per_word)
    optimal_batch_size = min(suggested_batch_size, max_batch_by_memory)
    optimal_batch_size = max(MIN_BATCH_SIZE, optimal_batch_size)
    
    # Round to nearest multiple of 128 for better work group alignment
    optimal_batch_size = (optimal_batch_size // 128) * 128
    
    if total_rules > 50000:
        optimal_batch_size = max(MIN_BATCH_SIZE, optimal_batch_size // 2)
    
    print(f"{green('🎯')} {bold('Optimal configuration:')}")
    print(f"   {blue('-')} {bold('Batch size:')} {cyan(f'{optimal_batch_size:,} words')}")
    print(f"   {blue('-')} {bold('Rules per batch:')} {cyan(f'{MAX_RULES_IN_BATCH:,}')}")
    print(f"   {blue('-')} {bold('Global hash map:')} {cyan(f'{global_bits} bits')} ({global_map_bytes / (1024**2):.1f} MB)")
    print(f"   {blue('-')} {bold('Cracked hash map:')} {cyan(f'{cracked_bits} bits')} ({cracked_map_bytes / (1024**2):.1f} MB)")
    print(f"   {blue('-')} {bold('Total map memory:')} {cyan(f'{total_map_memory / (1024**3):.2f} GB')}")
    print(f"   {blue('-')} {bold('Estimated rule batches:')} {cyan(f'{(total_rules + MAX_RULES_IN_BATCH - 1) // MAX_RULES_IN_BATCH}')}")
    
    return optimal_batch_size, global_bits, cracked_bits

def get_recommended_parameters(device, total_words, cracked_hashes_count):
    """
    Get recommended parameter values based on GPU capabilities and dataset size
    """
    total_vram, available_vram = get_gpu_memory_info(device)
    
    recommendations = {
        "low_memory": {
            "description": "Low Memory Mode (for GPUs with < 4GB VRAM)",
            "batch_size": 10000,
            "global_bits": 28,
            "cracked_bits": 26
        },
        "medium_memory": {
            "description": "Medium Memory Mode (for GPUs with 4-8GB VRAM)",
            "batch_size": 25000,
            "global_bits": 30,
            "cracked_bits": 28
        },
        "high_memory": {
            "description": "High Memory Mode (for GPUs with > 8GB VRAM)",
            "batch_size": 50000,
            "global_bits": 32,
            "cracked_bits": 30
        },
        "auto": {
            "description": "Auto-calculated (Recommended)",
            "batch_size": None,
            "global_bits": None,
            "cracked_bits": None
        }
    }
    
    if total_vram < 4 * 1024**3:
        recommended_preset = "low_memory"
    elif total_vram < 8 * 1024**3:
        recommended_preset = "medium_memory"
    else:
        recommended_preset = "high_memory"
    
    auto_batch, auto_global, auto_cracked = calculate_optimal_parameters_large_rules(
        available_vram, total_words, cracked_hashes_count, total_words
    )
    recommendations["auto"]["batch_size"] = auto_batch
    recommendations["auto"]["global_bits"] = auto_global
    recommendations["auto"]["cracked_bits"] = auto_cracked
    
    return recommendations, recommended_preset

def create_opencl_buffers_with_retry(context, buffer_specs, max_retries=MAX_ALLOCATION_RETRIES):
    """
    Create OpenCL buffers with retry logic for MEM_OBJECT_ALLOCATION_FAILURE
    """
    buffers = {}
    current_reduction = 1.0
    
    for retry in range(max_retries + 1):
        try:
            print(f"{blue('🔄')} {bold('Attempt')} {cyan(f'{retry + 1}/{max_retries + 1}')} {bold('to allocate buffers')} (reduction: {current_reduction:.2f})")
            
            for name, spec in buffer_specs.items():
                flags = spec['flags']
                size = int(spec['size'] * current_reduction)
                
                if 'hostbuf' in spec:
                    buffers[name] = cl.Buffer(context, flags, size, hostbuf=spec['hostbuf'])
                else:
                    buffers[name] = cl.Buffer(context, flags, size)
            
            print(f"{green('✅')} {bold('Successfully allocated all buffers on attempt')} {cyan(f'{retry + 1}')}")
            return buffers
            
        except cl.MemoryError as e:
            if "MEM_OBJECT_ALLOCATION_FAILURE" in str(e) and retry < max_retries:
                print(f"{yellow('⚠️')}  {bold('Memory allocation failed, reducing memory usage...')}")
                current_reduction *= MEMORY_REDUCTION_FACTOR
                for buf in buffers.values():
                    try:
                        buf.release()
                    except:
                        pass
                buffers = {}
                gc.collect()
            else:
                raise e
                
    raise cl.MemoryError(f"{red('❌')} {bold('Failed to allocate buffers after')} {cyan(f'{max_retries}')} {bold('retries')}")

# ====================================================================
# --- ULTRA-OPTIMIZED KERNEL SOURCE ---
# ====================================================================

def get_kernel_source(global_hash_map_bits, cracked_hash_map_bits):
    global_hash_map_mask = (1 << (global_hash_map_bits - 5)) - 1
    cracked_hash_map_mask = (1 << (cracked_hash_map_bits - 5)) - 1
    
    return """
// ============================================================================
// ULTRA-OPTIMIZED HASHCAT RULES KERNEL
// ============================================================================

#define MAX_WORD_LEN """ + str(MAX_WORD_LEN) + """
#define MAX_OUTPUT_LEN """ + str(MAX_OUTPUT_LEN) + """
#define MAX_RULE_LEN """ + str(MAX_RULE_LEN) + """
#define MAX_RULES_IN_BATCH """ + str(MAX_RULES_IN_BATCH) + """

// ============================================================================
// UTILITY FUNCTIONS (INLINED FOR PERFORMANCE)
// ============================================================================

inline int is_lower(unsigned char c) {
    return (c >= 'a' && c <= 'z');
}

inline int is_upper(unsigned char c) {
    return (c >= 'A' && c <= 'Z');
}

inline int is_digit(unsigned char c) {
    return (c >= '0' && c <= '9');
}

inline unsigned char toggle_case(unsigned char c) {
    if (is_lower(c)) return c - 32;
    if (is_upper(c)) return c + 32;
    return c;
}

inline unsigned char to_lower(unsigned char c) {
    if (is_upper(c)) return c + 32;
    return c;
}

inline unsigned char to_upper(unsigned char c) {
    if (is_lower(c)) return c - 32;
    return c;
}

// Optimized FNV-1a Hash implementation
inline unsigned int fnv1a_hash_32(const unsigned char* data, unsigned int len) {
    unsigned int hash = 2166136261U;
    for (unsigned int i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 16777619U;
    }
    return hash;
}

// Helper function to convert char digit/letter to int position
inline unsigned int char_to_pos(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    if (c >= 'a' && c <= 'z') return c - 'a' + 36;
    return 0xFFFFFFFF; 
}

// ============================================================================
// OPTIMIZED RULE APPLICATION FUNCTION
// ============================================================================

inline void apply_rule(const unsigned char* word, int word_len,
                      const unsigned char* rule, int rule_len,
                      unsigned char* output, int* out_len, int* changed) {
    
    *out_len = 0;
    *changed = 0;
    
    // Clear output efficiently
    for (int i = 0; i < MAX_OUTPUT_LEN; i++) {
        output[i] = 0;
    }
    
    if (rule_len == 0 || word_len == 0) return;
    
    // ========================================================================
    // SIMPLE RULES (1 character) - Optimized with unrolling
    // ========================================================================
    
    if (rule_len == 1) {
        switch (rule[0]) {
            case 'l':  // Lowercase all letters
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {
                    output[i] = to_lower(word[i]);
                }
                *changed = 1;
                return;
                
            case 'u':  // Uppercase all letters
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {
                    output[i] = to_upper(word[i]);
                }
                *changed = 1;
                return;
                
            case 'c':  // Capitalize first letter, lowercase rest
                *out_len = word_len;
                if (word_len > 0) {
                    output[0] = to_upper(word[0]);
                    for (int i = 1; i < word_len; i++) {
                        output[i] = to_lower(word[i]);
                    }
                }
                *changed = 1;
                return;
                
            case 't':  // Toggle case of all letters
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {
                    output[i] = toggle_case(word[i]);
                }
                *changed = 1;
                return;
                
            case 'r':  // Reverse the entire word
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {
                    output[i] = word[word_len - 1 - i];
                }
                *changed = 1;
                return;
                
            case ':':  // Do nothing (identity)
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {
                    output[i] = word[i];
                }
                *changed = 0;
                return;
                
            case 'd':  // Duplicate word
                if (word_len * 2 <= MAX_OUTPUT_LEN) {
                    *out_len = word_len * 2;
                    for (int i = 0; i < word_len; i++) {
                        output[i] = word[i];
                        output[word_len + i] = word[i];
                    }
                    *changed = 1;
                }
                return;
                
            default:
                // Unknown simple rule, treat as identity
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {
                    output[i] = word[i];
                }
                *changed = 0;
                return;
        }
    }
    
    // ========================================================================
    // POSITION-BASED RULES (Tn, Dn, etc.)
    // ========================================================================
    
    else if (rule_len == 2) {
        unsigned char cmd = rule[0];
        unsigned char arg = rule[1];
        
        if (is_digit(arg)) {
            int n = arg - '0';
            
            switch (cmd) {
                case 'T':  // Toggle case at position n
                    *out_len = word_len;
                    for (int i = 0; i < word_len; i++) {
                        output[i] = word[i];
                        if (i == n) {
                            output[i] = toggle_case(word[i]);
                            *changed = 1;
                        }
                    }
                    return;
                    
                case 'D':  // Delete character at position n
                    *out_len = 0;
                    for (int i = 0; i < word_len; i++) {
                        if (i != n) {
                            output[(*out_len)++] = word[i];
                        } else {
                            *changed = 1;
                        }
                    }
                    return;
                    
                case 'L':  // Delete left of position n
                    *out_len = 0;
                    for (int i = n; i < word_len; i++) {
                        output[(*out_len)++] = word[i];
                        *changed = 1;
                    }
                    return;
                    
                case 'R':  // Delete right of position n
                    *out_len = n + 1;
                    if (*out_len > word_len) *out_len = word_len;
                    for (int i = 0; i < *out_len; i++) {
                        output[i] = word[i];
                    }
                    *changed = 1;
                    return;
            }
        }
        
        switch (cmd) {
            case '^':  // Prepend character X
                if (word_len + 1 <= MAX_OUTPUT_LEN) {
                    output[0] = arg;
                    for (int i = 0; i < word_len; i++) {
                        output[i + 1] = word[i];
                    }
                    *out_len = word_len + 1;
                    *changed = 1;
                }
                return;
                
            case '$':  // Append character X
                if (word_len + 1 <= MAX_OUTPUT_LEN) {
                    for (int i = 0; i < word_len; i++) {
                        output[i] = word[i];
                    }
                    output[word_len] = arg;
                    *out_len = word_len + 1;
                    *changed = 1;
                }
                return;
                
            case '@':  // Delete all instances of character X
                *out_len = 0;
                for (int i = 0; i < word_len; i++) {
                    if (word[i] != arg) {
                        output[(*out_len)++] = word[i];
                    } else {
                        *changed = 1;
                    }
                }
                return;
        }
    }
    
    // ========================================================================
    // SUBSTITUTION RULES (sXY)
    // ========================================================================
    
    else if (rule_len == 3 && rule[0] == 's') {
        unsigned char find = rule[1];
        unsigned char replace = rule[2];
        
        *out_len = word_len;
        for (int i = 0; i < word_len; i++) {
            output[i] = word[i];
            if (word[i] == find) {
                output[i] = replace;
                *changed = 1;
            }
        }
        return;
    }
    
    // ========================================================================
    // COMPLEX RULES (multi-character)
    // ========================================================================
    
    else if (rule_len >= 3) {
        if (rule[0] == 'x' && rule_len >= 3) {
            unsigned int n = char_to_pos(rule[1]);
            unsigned int m = char_to_pos(rule[2]);
            
            if (n != 0xFFFFFFFF && m != 0xFFFFFFFF) {
                if (n > m) {
                    unsigned int temp = n;
                    n = m;
                    m = temp;
                }
                if (n >= word_len) n = 0;
                if (m >= word_len) m = word_len - 1;
                
                *out_len = 0;
                for (unsigned int i = n; i <= m && i < word_len; i++) {
                    output[(*out_len)++] = word[i];
                }
                *changed = (*out_len > 0);
                return;
            }
        }
        
        else if (rule[0] == '*' && rule_len >= 3) {
            unsigned int n = char_to_pos(rule[1]);
            unsigned int m = char_to_pos(rule[2]);
            
            *out_len = word_len;
            for (int i = 0; i < word_len; i++) {
                output[i] = word[i];
            }
            if (n != 0xFFFFFFFF && m != 0xFFFFFFFF && n < word_len && m < word_len && n != m) {
                unsigned char temp = output[n];
                output[n] = output[m];
                output[m] = temp;
                *changed = 1;
            }
            return;
        }
        
        else if (rule[0] == 'i' && rule_len >= 3) {
            unsigned int n = char_to_pos(rule[1]);
            unsigned char x = rule[2];
            
            if (n != 0xFFFFFFFF && word_len + 1 <= MAX_OUTPUT_LEN) {
                *out_len = 0;
                for (int i = 0; i < word_len; i++) {
                    if (i == n) {
                        output[(*out_len)++] = x;
                    }
                    output[(*out_len)++] = word[i];
                }
                if (n >= word_len) {
                    output[(*out_len)++] = x;
                }
                *changed = 1;
                return;
            }
        }
    }
    
    // ========================================================================
    // DEFAULT: Unknown rule, treat as identity
    // ========================================================================
    
    *out_len = word_len;
    for (int i = 0; i < word_len; i++) {
        output[i] = word[i];
    }
    *changed = 0;
}

// ============================================================================
// MAIN KERNEL
// ============================================================================

__kernel void bfs_kernel(
    __global const uchar* base_words_in,
    __global const uint* rules_in,
    __global uint* rule_uniqueness_counts,
    __global uint* rule_effectiveness_counts,
    __global const uint* global_hash_map,
    __global const uint* cracked_hash_map,
    const uint num_words,
    const uint num_rules_in_batch,
    const uint max_word_len,
    const uint max_output_len,
    const uint global_map_mask,
    const uint cracked_map_mask)
{
    const uint global_id = get_global_id(0);
    const uint word_per_rule_count = num_words * num_rules_in_batch;
    
    if (global_id >= word_per_rule_count) return;

    const uint word_idx = global_id / num_rules_in_batch;
    const uint rule_batch_idx = global_id % num_rules_in_batch;

    // Load word directly from global memory
    uchar word[MAX_WORD_LEN];
    uint word_len = 0;
    const uint word_base_idx = word_idx * max_word_len;
    
    for (uint i = 0; i < max_word_len; i++) {
        uchar c = base_words_in[word_base_idx + i];
        if (c == 0) break;
        word[i] = c;
        word_len++;
    }
    
    // Load rule from global memory
    const uint rule_size_in_int = 2 + """ + str(MAX_RULE_ARGS) + """;
    __global const uint* rule_ptr = rules_in + rule_batch_idx * rule_size_in_int;
    uint rule_args_int = rule_ptr[1];
    uint rule_args_int2 = rule_ptr[2];
    
    uchar rule_str[MAX_RULE_LEN];
    uint rule_len = 0;
    
    for (uint j = 0; j < 4; j++) {
        uchar byte = (rule_args_int >> (j * 8)) & 0xFF;
        if (byte == 0) break;
        rule_str[rule_len++] = byte;
    }
    
    if (rule_len < MAX_RULE_LEN) {
        for (uint j = 0; j < 4; j++) {
            uchar byte = (rule_args_int2 >> (j * 8)) & 0xFF;
            if (byte == 0) break;
            rule_str[rule_len++] = byte;
        }
    }
    
    // Apply rule
    uchar result_temp[MAX_OUTPUT_LEN];
    int out_len = 0;
    int changed_flag = 0;
    
    apply_rule(word, word_len, rule_str, rule_len, result_temp, &out_len, &changed_flag);
    
    // DUAL-UNIQUENESS LOGIC
    if (changed_flag > 0 && out_len > 0) {
        uint word_hash = fnv1a_hash_32(result_temp, out_len);
        
        // 1. Check against Base Wordlist (Uniqueness)
        uint global_map_index = (word_hash >> 5) & """ + str(global_hash_map_mask) + """;
        uint bit_index = word_hash & 31;
        uint check_bit = (1U << bit_index);
        
        __global const uint* global_map_ptr = &global_hash_map[global_map_index];
        uint current_global_word = *global_map_ptr;
        
        if (!(current_global_word & check_bit)) {
            atomic_inc(&rule_uniqueness_counts[rule_batch_idx]);
            
            // 2. Check against Cracked List (Effectiveness)
            uint cracked_map_index = (word_hash >> 5) & """ + str(cracked_hash_map_mask) + """;
            __global const uint* cracked_map_ptr = &cracked_hash_map[cracked_map_index];
            uint current_cracked_word = *cracked_map_ptr;
            
            if (current_cracked_word & check_bit) {
                atomic_inc(&rule_effectiveness_counts[rule_batch_idx]);
            }
        }
    }
}

// ============================================================================
// ACCUMULATION KERNEL (Reduce host-device transfers)
// ============================================================================

__kernel void accumulate_results(
    __global const uint* batch_counts,
    __global uint* accumulated_counts,
    const uint batch_offset,
    const uint num_rules)
{
    uint gid = get_global_id(0);
    if (gid < num_rules) {
        atomic_add(&accumulated_counts[batch_offset + gid], batch_counts[gid]);
    }
}

// ============================================================================
// HASH MAP INITIALIZATION KERNEL
// ============================================================================

__kernel void hash_map_init_kernel(
    __global uint* global_hash_map,
    __global const uint* base_hashes,
    const uint num_hashes,
    const uint map_mask)
{
    uint global_id = get_global_id(0);
    if (global_id >= num_hashes) return;

    uint word_hash = base_hashes[global_id];
    uint map_index = (word_hash >> 5) & map_mask;
    uint bit_index = word_hash & 31;
    uint set_bit = (1U << bit_index);

    atomic_or(&global_hash_map[map_index], set_bit);
}
"""

# ====================================================================
# --- WORK GROUP SIZE MANAGEMENT ---
# ====================================================================

def get_optimal_work_group_size(device, kernel):
    """Get optimal work group size for a kernel on the given device"""
    try:
        # Query device for maximum work group size
        max_work_group_size = device.max_work_group_size
        print(f"{blue('🔧')} {bold('Device max work group size:')} {cyan(f'{max_work_group_size}')}")
        
        # Try to get preferred work group size for the kernel
        try:
            preferred_size = kernel.get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device)
            print(f"{blue('🔧')} {bold('Kernel preferred work group size multiple:')} {cyan(f'{preferred_size}')}")
            
            # Use 128 as base if preferred size is reasonable
            if preferred_size > 0 and preferred_size <= 256:
                optimal_size = min(256, max_work_group_size)
                optimal_size = (optimal_size // preferred_size) * preferred_size
            else:
                optimal_size = 128
        except:
            optimal_size = 128
        
        # Ensure optimal size doesn't exceed device limits
        optimal_size = min(optimal_size, max_work_group_size)
        
        print(f"{green('🎯')} {bold('Selected work group size:')} {cyan(f'{optimal_size}')}")
        return optimal_size
    
    except Exception as e:
        print(f"{yellow('⚠️')}  {bold('Warning: Could not query work group info:')} {e}")
        return 128

def calculate_work_sizes(total_work, preferred_local_size):
    """Calculate global and local work sizes that are compatible"""
    # Ensure total_work is at least 1
    total_work = max(1, total_work)
    
    # Round up global size to nearest multiple of local size
    global_size = ((total_work + preferred_local_size - 1) // preferred_local_size) * preferred_local_size
    
    # Ensure local size doesn't exceed total work
    local_size = min(preferred_local_size, total_work)
    
    # Ensure local size divides global size evenly
    if global_size % local_size != 0:
        # Find the largest divisor of global_size that's <= preferred_local_size
        for size in range(preferred_local_size, 0, -1):
            if global_size % size == 0:
                local_size = size
                break
    
    return (global_size,), (local_size,)

# ====================================================================
# --- MAIN RANKING FUNCTION WITH ALL OPTIMIZATIONS ---
# ====================================================================

def rank_rules_uniqueness_large(wordlist_path, rules_path, cracked_list_path, ranking_output_path, top_k, 
                               words_per_gpu_batch=None, global_hash_map_bits=None, cracked_hash_map_bits=None,
                               preset=None):
    start_time = time()
    
    # 0. PRELIMINARY DATA LOADING FOR MEMORY CALCULATION
    total_words = estimate_word_count(wordlist_path)
    rules_list = load_rules(rules_path)
    total_rules = len(rules_list)
    
    setup_interrupt_handler(rules_list, ranking_output_path, top_k)
    
    if total_rules > 100000:
        print(f"{green('🚀')} {bold('LARGE RULE SET DETECTED:')} {cyan(f'{total_rules:,}')} {bold('rules')}")
        print(f"   {bold('Using optimized processing for large rule sets...')}")
    
    cracked_hashes_np = load_cracked_hashes(cracked_list_path, MAX_WORD_LEN)
    cracked_hashes_count = len(cracked_hashes_np)
    
    # 1. OPENCL INITIALIZATION AND MEMORY DETECTION
    try:
        platform = cl.get_platforms()[0]
        devices = platform.get_devices(cl.device_type.GPU)
        if not devices:
            devices = platform.get_devices(cl.device_type.ALL)
        device = devices[0]
        
        # Create command queue
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        
        # Get GPU memory information
        total_vram, available_vram = get_gpu_memory_info(device)
        print(f"{green('🎮')} {bold('GPU:')} {cyan(device.name.strip())}")
        print(f"{blue('💾')} {bold('Total VRAM:')} {cyan(f'{total_vram / (1024**3):.1f} GB')}")
        print(f"{blue('💾')} {bold('Available VRAM:')} {cyan(f'{available_vram / (1024**3):.1f} GB')}")
        
        # Handle preset parameter specification
        if preset:
            recommendations, recommended_preset = get_recommended_parameters(device, total_words, cracked_hashes_count)
            
            if preset == "recommend":
                print(f"{green('🎯')} {bold('Recommended preset:')} {cyan(recommended_preset)}")
                preset = recommended_preset
            
            if preset in recommendations:
                preset_config = recommendations[preset]
                print(f"{blue('🔧')} {bold('Using')} {cyan(preset_config['description'])}")
                words_per_gpu_batch = preset_config['batch_size']
                global_hash_map_bits = preset_config['global_bits']
                cracked_hash_map_bits = preset_config['cracked_bits']
            else:
                print(f"{red('❌')} {bold('Unknown preset:')} {cyan(preset)}. {bold('Available presets:')} {list(recommendations.keys())}")
                return
        
        # Handle manual parameter specification
        using_manual_params = False
        if words_per_gpu_batch is not None or global_hash_map_bits is not None or cracked_hash_map_bits is not None:
            using_manual_params = True
            print(f"{blue('🔧')} {bold('Using manually specified parameters:')}")
            
            if words_per_gpu_batch is None:
                words_per_gpu_batch = DEFAULT_WORDS_PER_GPU_BATCH
            if global_hash_map_bits is None:
                global_hash_map_bits = DEFAULT_GLOBAL_HASH_MAP_BITS
            if cracked_hash_map_bits is None:
                cracked_hash_map_bits = DEFAULT_CRACKED_HASH_MAP_BITS
                
            print(f"   {blue('-')} {bold('Batch size:')} {cyan(f'{words_per_gpu_batch:,}')}")
            print(f"   {blue('-')} {bold('Global hash map:')} {cyan(f'{global_hash_map_bits} bits')}")
            print(f"   {blue('-')} {bold('Cracked hash map:')} {cyan(f'{cracked_hash_map_bits} bits')}")
            
            global_map_bytes = (1 << (global_hash_map_bits - 5)) * np.uint32().itemsize
            cracked_map_bytes = (1 << (cracked_hash_map_bits - 5)) * np.uint32().itemsize
            total_map_memory = global_map_bytes + cracked_map_bytes
            
            word_batch_bytes = words_per_gpu_batch * MAX_WORD_LEN * np.uint8().itemsize
            hash_batch_bytes = words_per_gpu_batch * np.uint32().itemsize
            rule_batch_bytes = MAX_RULES_IN_BATCH * (2 + MAX_RULE_ARGS) * np.uint32().itemsize
            counter_bytes = MAX_RULES_IN_BATCH * np.uint32().itemsize * 2
            
            total_batch_memory = (word_batch_bytes + hash_batch_bytes) * 2 + rule_batch_bytes + counter_bytes + total_map_memory
            
            if total_batch_memory > available_vram:
                print(f"{yellow('⚠️')}  {bold('Warning: Manual parameters exceed available VRAM!')}")
                print(f"   {bold('Required:')} {cyan(f'{total_batch_memory / (1024**3):.2f} GB')}")
                print(f"   {bold('Available:')} {cyan(f'{available_vram / (1024**3):.2f} GB')}")
                print(f"   {bold('Consider reducing batch size or hash map bits')}")
        else:
            # Auto-calculate optimal parameters
            words_per_gpu_batch, global_hash_map_bits, cracked_hash_map_bits = calculate_optimal_parameters_large_rules(
                available_vram, total_words, cracked_hashes_count, total_rules
            )

        # Calculate derived constants
        GLOBAL_HASH_MAP_WORDS = 1 << (global_hash_map_bits - 5)
        GLOBAL_HASH_MAP_BYTES = GLOBAL_HASH_MAP_WORDS * np.uint32(4)
        GLOBAL_HASH_MAP_MASK = (1 << (global_hash_map_bits - 5)) - 1
        
        CRACKED_HASH_MAP_WORDS = 1 << (cracked_hash_map_bits - 5)
        CRACKED_HASH_MAP_BYTES = CRACKED_HASH_MAP_WORDS * np.uint32(4)
        CRACKED_HASH_MAP_MASK = (1 << (cracked_hash_map_bits - 5)) - 1

        # Compile kernel
        KERNEL_SOURCE = get_kernel_source(global_hash_map_bits, cracked_hash_map_bits)
        prg = cl.Program(context, KERNEL_SOURCE).build(options=[
            "-cl-fast-relaxed-math",
            "-cl-mad-enable",
            "-cl-no-signed-zeros"
        ])
        
        kernel_bfs = prg.bfs_kernel
        kernel_init = prg.hash_map_init_kernel
        kernel_accumulate = prg.accumulate_results
        
        # Get optimal work group size
        local_work_size = get_optimal_work_group_size(device, kernel_bfs)
        
        print(f"{green('✅')} {bold('OpenCL initialized on device:')} {cyan(device.name.strip())}")
        print(f"{blue('🔧')} {bold('Using work group size:')} {cyan(f'{local_work_size}')}")
    except Exception as e:
        print(f"{red('❌')} {bold('OpenCL initialization or kernel compilation error:')} {e}")
        exit(1)

    # 2. DATA LOADING AND PRE-ENCODING
    rule_size_in_int = 2 + MAX_RULE_ARGS
    encoded_rules = [encode_rule(rule['rule_data'], rule['rule_id'], MAX_RULE_ARGS) for rule in rules_list]

    # 3. HASH MAP INITIALIZATION
    global_hash_map_np = np.zeros(GLOBAL_HASH_MAP_WORDS, dtype=np.uint32)
    print(f"{blue('📝')} {bold('Global Hash Map initialized:')} {cyan(f'{global_hash_map_np.nbytes / (1024*1024):.2f} MB')}")
    
    cracked_hash_map_np = np.zeros(CRACKED_HASH_MAP_WORDS, dtype=np.uint32)
    print(f"{blue('📝')} {bold('Cracked Hash Map initialized:')} {cyan(f'{cracked_hash_map_np.nbytes / (1024*1024):.2f} MB')}")

    # 4. OPENCL BUFFER SETUP WITH RETRY LOGIC
    mf = cl.mem_flags
    counters_size = MAX_RULES_IN_BATCH * np.uint32().itemsize
    
    # Define buffer specifications for retry logic
    buffer_specs = {
        'base_words_in': {
            'flags': mf.READ_ONLY,
            'size': words_per_gpu_batch * MAX_WORD_LEN * np.uint8().itemsize
        },
        'base_hashes': {
            'flags': mf.READ_ONLY,
            'size': words_per_gpu_batch * np.uint32().itemsize
        },
        'rules_in': {
            'flags': mf.READ_ONLY,
            'size': MAX_RULES_IN_BATCH * rule_size_in_int * np.uint32().itemsize
        },
        'global_hash_map': {
            'flags': mf.READ_WRITE,
            'size': global_hash_map_np.nbytes
        },
        'cracked_hash_map': {
            'flags': mf.READ_ONLY,
            'size': cracked_hash_map_np.nbytes
        },
        'rule_uniqueness_counts': {
            'flags': mf.READ_WRITE,
            'size': counters_size
        },
        'rule_effectiveness_counts': {
            'flags': mf.READ_WRITE,
            'size': counters_size
        },
        'accumulated_uniqueness': {
            'flags': mf.READ_WRITE,
            'size': total_rules * np.uint32().itemsize
        },
        'accumulated_effectiveness': {
            'flags': mf.READ_WRITE,
            'size': total_rules * np.uint32().itemsize
        }
    }

    if cracked_hashes_np.size > 0:
        buffer_specs['cracked_temp'] = {
            'flags': mf.READ_ONLY | mf.COPY_HOST_PTR,
            'size': cracked_hashes_np.nbytes,
            'hostbuf': cracked_hashes_np
        }

    try:
        buffers = create_opencl_buffers_with_retry(context, buffer_specs)
        
        # Extract buffers
        base_words_in_g = buffers['base_words_in']
        base_hashes_g = buffers['base_hashes']
        rules_in_g = buffers['rules_in']
        global_hash_map_g = buffers['global_hash_map']
        cracked_hash_map_g = buffers['cracked_hash_map']
        rule_uniqueness_counts_g = buffers['rule_uniqueness_counts']
        rule_effectiveness_counts_g = buffers['rule_effectiveness_counts']
        accumulated_uniqueness_g = buffers['accumulated_uniqueness']
        accumulated_effectiveness_g = buffers['accumulated_effectiveness']
        cracked_temp_g = buffers.get('cracked_temp', None)
        
    except cl.MemoryError as e:
        print(f"{red('❌')} {bold('Fatal: Could not allocate GPU memory even after retries:')} {e}")
        recommendations, _ = get_recommended_parameters(device, total_words, cracked_hashes_count)
        for preset_name, config in recommendations.items():
            if preset_name != "auto":
                print(f"   {bold('--preset')} {cyan(preset_name)}: {config['description']}")
        return

    # 5. INITIALIZE CRACKED HASH MAP (ONCE)
    if cracked_hashes_np.size > 0 and cracked_temp_g is not None:
        global_size_init_cracked, local_size_init_cracked = calculate_work_sizes(
            cracked_hashes_np.size, local_work_size
        )

        print(f"{blue('📊')} {bold('Populating static Cracked Hash Map on GPU...')}")
        try:
            kernel_init(queue, global_size_init_cracked, local_size_init_cracked,
                        cracked_hash_map_g,
                        cracked_temp_g,
                        np.uint32(cracked_hashes_np.size),
                        np.uint32(CRACKED_HASH_MAP_MASK)).wait()
            print(f"{green('✅')} {bold('Static Cracked Hash Map populated.')}")
        except cl.LogicError as e:
            print(f"{yellow('⚠️')}  {bold('Warning: Cracked hash map initialization failed:')} {e}")
            # Continue without cracked hash map
            cracked_hashes_np = np.array([], dtype=np.uint32)
        
    else:
        print(f"{yellow('⚠️')} {bold('Cracked list is empty, effectiveness scoring is disabled.')}")
        
    # 6. INITIALIZE ACCUMULATION BUFFERS
    cl.enqueue_fill_buffer(queue, accumulated_uniqueness_g, np.uint32(0), 0, total_rules * np.uint32().itemsize)
    cl.enqueue_fill_buffer(queue, accumulated_effectiveness_g, np.uint32(0), 0, total_rules * np.uint32().itemsize)
    queue.finish()
    
    # 7. PIPELINED RANKING LOOP SETUP
    word_iterator = optimized_wordlist_iterator(wordlist_path, MAX_WORD_LEN, words_per_gpu_batch)
    rule_batch_starts = list(range(0, total_rules, MAX_RULES_IN_BATCH))
    total_rule_batches = len(rule_batch_starts)
    
    print(f"{blue('📊')} {bold('Processing')} {cyan(f'{total_rules:,}')} {bold('rules in')} {cyan(f'{total_rule_batches:,}')} {bold('batches')} "
          f"{bold('(up to')} {cyan(f'{MAX_RULES_IN_BATCH:,}')} {bold('rules per batch)')}")
    
    # Load all word batches into memory
    print(f"{blue('📊')} {bold('Preloading word batches...')}")
    word_batches = []
    try:
        while True:
            try:
                base_words_np_batch, base_hashes_np_batch, num_words_batch = next(word_iterator)
                word_batches.append({
                    'words': base_words_np_batch,
                    'hashes': base_hashes_np_batch,
                    'count': num_words_batch
                })
            except StopIteration:
                break
    except Exception as e:
        print(f"{red('❌')} {bold('Error preloading word batches:')} {e}")
        return
    
    total_word_batches = len(word_batches)
    print(f"{green('✅')} {bold('Preloaded')} {cyan(f'{total_word_batches:,}')} {bold('word batches')}")
    
    # Use Python ints for counters
    words_processed_total = 0
    total_unique_found = 0
    total_cracked_found = 0
    
    mapped_uniqueness_np = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
    mapped_effectiveness_np = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
    
    # Create progress bars
    word_batch_pbar = tqdm(total=total_words, desc="Wordlist progress", unit=" word", 
                          bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                          position=0)
    rule_batch_pbar = tqdm(total=total_rule_batches, desc="Rule batches processed", unit=" batch", 
                          bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                          position=1)

    # 8. MAIN PROCESSING LOOP
    for word_batch_idx, word_batch_data in enumerate(word_batches):
        if interrupted:
            break
        
        # Upload current word batch
        cl.enqueue_copy(queue, base_words_in_g, word_batch_data['words'], is_blocking=True)
        cl.enqueue_copy(queue, base_hashes_g, word_batch_data['hashes'], is_blocking=True)
        
        # Clear global hash map for current batch
        cl.enqueue_fill_buffer(queue, global_hash_map_g, np.uint32(0), 0, GLOBAL_HASH_MAP_BYTES)
        
        # Initialize global hash map with current word batch
        global_size_init_global, local_size_init_global = calculate_work_sizes(
            word_batch_data['count'], local_work_size
        )
        
        try:
            kernel_init(queue, global_size_init_global, local_size_init_global,
                       global_hash_map_g,
                       base_hashes_g,
                       np.uint32(word_batch_data['count']),
                       np.uint32(GLOBAL_HASH_MAP_MASK)).wait()
        except cl.LogicError as e:
            print(f"{yellow('⚠️')}  {bold('Warning: Hash map initialization failed:')} {e}")
            continue
        
        # Process each rule batch
        for rule_batch_idx, rule_start_index in enumerate(rule_batch_starts):
            if interrupted:
                break
                
            rule_end_index = min(rule_start_index + MAX_RULES_IN_BATCH, total_rules)
            num_rules_in_batch = rule_end_index - rule_start_index
            
            current_rule_batch_list = encoded_rules[rule_start_index:rule_end_index]
            current_rules_np = np.concatenate(current_rule_batch_list)
            
            # Upload rule batch
            cl.enqueue_copy(queue, rules_in_g, current_rules_np, is_blocking=True)
            
            # Clear counters
            cl.enqueue_fill_buffer(queue, rule_uniqueness_counts_g, 
                                 np.uint32(0), 0, counters_size)
            cl.enqueue_fill_buffer(queue, rule_effectiveness_counts_g,
                                 np.uint32(0), 0, counters_size)
            
            # Calculate kernel size with proper work group alignment
            total_work_items = word_batch_data['count'] * num_rules_in_batch
            global_size, local_size = calculate_work_sizes(total_work_items, local_work_size)
            
            # Set kernel arguments
            kernel_args = [
                base_words_in_g,
                rules_in_g,
                rule_uniqueness_counts_g,
                rule_effectiveness_counts_g,
                global_hash_map_g,
                cracked_hash_map_g,
                np.uint32(word_batch_data['count']),
                np.uint32(num_rules_in_batch),
                np.uint32(MAX_WORD_LEN),
                np.uint32(MAX_OUTPUT_LEN),
                np.uint32(GLOBAL_HASH_MAP_MASK),
                np.uint32(CRACKED_HASH_MAP_MASK)
            ]
            
            try:
                # Execute kernel
                kernel_bfs.set_args(*kernel_args)
                cl.enqueue_nd_range_kernel(queue, kernel_bfs, global_size, local_size)
                
                # Accumulate results on GPU
                accum_global_size, accum_local_size = calculate_work_sizes(num_rules_in_batch, local_work_size)
                
                kernel_accumulate.set_args(
                    rule_uniqueness_counts_g,
                    accumulated_uniqueness_g,
                    np.uint32(rule_start_index),
                    np.uint32(num_rules_in_batch)
                )
                cl.enqueue_nd_range_kernel(queue, kernel_accumulate, accum_global_size, accum_local_size)
                
                kernel_accumulate.set_args(
                    rule_effectiveness_counts_g,
                    accumulated_effectiveness_g,
                    np.uint32(rule_start_index),
                    np.uint32(num_rules_in_batch)
                )
                cl.enqueue_nd_range_kernel(queue, kernel_accumulate, accum_global_size, accum_local_size)
                
                queue.finish()
                
            except cl.LogicError as e:
                print(f"{yellow('⚠️')}  {bold('Warning: Kernel execution failed for batch')} {cyan(f'{rule_batch_idx + 1}/{total_rule_batches}')}: {e}")
                # Continue with next batch
                continue
            
            # Update progress bars
            rule_batch_pbar.update(1)
            rule_batch_pbar.set_description(
                f"Rule batches: {rule_batch_idx + 1}/{total_rule_batches}"
            )
        
        # Update word count for completed word batch
        words_processed_total += word_batch_data['count']
        
        # Update word batch progress bar
        word_batch_pbar.update(word_batch_data['count'])
        word_batch_pbar.set_description(
            f"Wordlist: {words_processed_total:,}/{total_words:,} [Batch: {word_batch_idx+1}/{total_word_batches}]"
        )
        
        # Update interrupt handler stats
        update_progress_stats(words_processed_total, total_unique_found, total_cracked_found)
    
    # Check if we were interrupted
    if interrupted:
        print(f"\n{yellow('⚠️')} {bold('Processing was interrupted. Intermediate results have been saved.')}")
        return
    
    # 9. COPY ACCUMULATED RESULTS BACK
    print(f"{blue('📊')} {bold('Copying final results from GPU...')}")
    
    final_uniqueness_np = np.zeros(total_rules, dtype=np.uint32)
    final_effectiveness_np = np.zeros(total_rules, dtype=np.uint32)
    
    try:
        cl.enqueue_copy(queue, final_uniqueness_np, accumulated_uniqueness_g, is_blocking=True)
        cl.enqueue_copy(queue, final_effectiveness_np, accumulated_effectiveness_g, is_blocking=True)
        queue.finish()
        
        # Update rule scores
        for i in range(total_rules):
            rules_list[i]['uniqueness_score'] = int(final_uniqueness_np[i])
            rules_list[i]['effectiveness_score'] = int(final_effectiveness_np[i])
        
        total_unique_found = final_uniqueness_np.sum()
        total_cracked_found = final_effectiveness_np.sum()
        
    except Exception as e:
        print(f"{yellow('⚠️')}  {bold('Warning: Failed to copy results from GPU:')} {e}")
        # Use placeholder scores
        total_unique_found = 0
        total_cracked_found = 0
    
    # Final update for progress bars
    word_batch_pbar.close()
    rule_batch_pbar.close()
    
    end_time = time()
    
    # 10. FINAL REPORTING AND SAVING
    print(f"\n{green('=' * 60)}")
    print(f"{bold('🎉 Final Results Summary')}")
    print(f"{green('=' * 60)}")
    print(f"{blue('📊')} {bold('Total Words Processed:')} {cyan(f'{words_processed_total:,}')}")
    print(f"{blue('📊')} {bold('Total Rules Processed:')} {cyan(f'{total_rules:,}')}")
    print(f"{blue('🎯')} {bold('Total Unique Words Generated:')} {cyan(f'{total_unique_found:,}')}")
    print(f"{blue('🔓')} {bold('Total True Cracks Found:')} {cyan(f'{total_cracked_found:,}')}")
    print(f"{blue('⏱️')} {bold('Total Processing Time:')} {cyan(f'{end_time - start_time:.2f} seconds')}")
    
    # Calculate and display performance metrics
    total_operations = words_processed_total * total_rules
    if total_operations > 0:
        operations_per_second = total_operations / (end_time - start_time)
        print(f"{blue('⚡')} {bold('Processing Speed:')} {cyan(f'{operations_per_second:,.0f}')} {bold('operations/sec')}")
        
        if total_unique_found > 0:
            uniqueness_ratio = (total_unique_found / total_operations) * 100
            print(f"{blue('📈')} {bold('Uniqueness Ratio:')} {cyan(f'{uniqueness_ratio:.4f}%')}")
        
        if total_cracked_found > 0:
            effectiveness_ratio = (total_cracked_found / total_operations) * 100
            print(f"{blue('📈')} {bold('Effectiveness Ratio:')} {cyan(f'{effectiveness_ratio:.6f}%')}")
    
    print(f"{green('=' * 60)}{Colors.END}\n")

    # Save ranking data to CSV
    ranking_csv_path = save_ranking_data(rules_list, ranking_output_path)
    
    if ranking_csv_path:
        # Save optimized rules
        optimized_output_path = ranking_output_path.replace('.csv', '.rule')
        load_and_save_optimized_rules(ranking_csv_path, optimized_output_path, top_k)
    
    return ranking_csv_path


# ====================================================================
# --- MAIN FUNCTION ---
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RANKER v4.2: Ultra-Optimized GPU Rule Ranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -w wordlist.txt -r rules.txt -c cracked.txt -o ranking.csv --top-k 1000
  %(prog)s -w wordlist.txt -r rules.txt -c cracked.txt -o ranking.csv --preset high_memory
  %(prog)s -w wordlist.txt -r rules.txt -c cracked.txt -o ranking.csv --batch-size 50000
  %(prog)s -w wordlist.txt -r rules.txt -c cracked.txt -o ranking.csv --recommend
        """
    )
    
    parser.add_argument("-w", "--wordlist", required=True,
                       help="Path to base wordlist file")
    parser.add_argument("-r", "--rules", required=True,
                       help="Path to rules file")
    parser.add_argument("-c", "--cracked", required=True,
                       help="Path to cracked passwords file (for effectiveness scoring)")
    parser.add_argument("-o", "--output", required=True,
                       help="Output ranking CSV path")
    parser.add_argument("--top-k", type=int, default=1000,
                       help="Number of top rules to save in optimized file (default: 1000)")
    
    # Performance tuning parameters
    parser.add_argument("--batch-size", type=int,
                       help=f"Number of words per GPU batch (default: auto-calculated, typical: {DEFAULT_WORDS_PER_GPU_BATCH:,})")
    parser.add_argument("--global-bits", type=int,
                       help=f"Global hash map size in bits (default: auto-calculated, typical: {DEFAULT_GLOBAL_HASH_MAP_BITS})")
    parser.add_argument("--cracked-bits", type=int,
                       help=f"Cracked hash map size in bits (default: auto-calculated, typical: {DEFAULT_CRACKED_HASH_MAP_BITS})")
    
    # Preset modes
    parser.add_argument("--preset", choices=["low_memory", "medium_memory", "high_memory", "recommend"],
                       help="Use predefined parameter sets based on GPU memory")
    
    args = parser.parse_args()

    # Print banner
    print(f"{Colors.CYAN}{'='*70}")
    print(f"{Colors.BOLD}       RANKER v4.2: ULTRA-OPTIMIZED GPU RULE RANKING       ")
    print(f"{Colors.CYAN}{'='*70}{Colors.END}\n")
    
    print(f"{green('🚀')} {bold('Starting GPU-accelerated rule ranking...')}")
    print(f"{blue('📁')} {bold('Wordlist:')} {cyan(args.wordlist)}")
    print(f"{blue('📁')} {bold('Rules:')} {cyan(args.rules)}")
    print(f"{blue('📁')} {bold('Cracked list:')} {cyan(args.cracked)}")
    print(f"{blue('💾')} {bold('Output:')} {cyan(args.output)}")
    print(f"{blue('🎯')} {bold('Top K:')} {cyan(f'{args.top_k:,}')}\n")
    
    # Validate file existence
    for file_path, file_type in [(args.wordlist, "wordlist"),
                                 (args.rules, "rules"),
                                 (args.cracked, "cracked list")]:
        if not os.path.exists(file_path):
            print(f"{red('❌')} {bold('Error:')} {file_type} file not found: {file_path}")
            return
    
    try:
        # Run the main ranking function
        ranking_csv_path = rank_rules_uniqueness_large(
            wordlist_path=args.wordlist,
            rules_path=args.rules,
            cracked_list_path=args.cracked,
            ranking_output_path=args.output,
            top_k=args.top_k,
            words_per_gpu_batch=args.batch_size,
            global_hash_map_bits=args.global_bits,
            cracked_hash_map_bits=args.cracked_bits,
            preset=args.preset
        )
        
        if ranking_csv_path:
            print(f"{green('✅')} {bold('Ranking completed successfully!')}")
            print(f"{blue('📊')} {bold('Ranking data saved to:')} {cyan(ranking_csv_path)}")
            
            # Generate optimized rules file path
            optimized_rules_path = args.output.replace('.csv', '.rule')
            if os.path.exists(optimized_rules_path):
                print(f"{blue('🔧')} {bold('Optimized rules saved to:')} {cyan(optimized_rules_path)}")
        
        print(f"\n{green('🎉')} {bold('All operations completed successfully!')}")
        
    except KeyboardInterrupt:
        print(f"\n{yellow('⚠️')} {bold('Operation interrupted by user.')}")
    except Exception as e:
        print(f"\n{red('❌')} {bold('An error occurred during ranking:')}")
        import traceback
        print(f"{red(str(e))}")
        traceback.print_exc()


# ====================================================================
# --- ENTRY POINT ---
# ====================================================================

if __name__ == "__main__":
    main()

