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
# --- RANKER v3.2: OPTIMIZED LARGE FILE LOADING ---
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
MAX_WORD_LEN = 32
MAX_OUTPUT_LEN = MAX_WORD_LEN * 2
MAX_RULE_ARGS = 4
MAX_RULES_IN_BATCH = 1024
LOCAL_WORK_SIZE = 256

# Default values (will be adjusted based on VRAM)
DEFAULT_WORDS_PER_GPU_BATCH = 200000
DEFAULT_GLOBAL_HASH_MAP_BITS = 35
DEFAULT_CRACKED_HASH_MAP_BITS = 33

# VRAM usage thresholds (adjustable)
VRAM_SAFETY_MARGIN = 0.15  # 15% safety margin
MIN_BATCH_SIZE = 50000     # Minimum batch size to maintain performance
MIN_HASH_MAP_BITS = 28     # Minimum hash map size (256MB)

# Memory reduction factors for allocation failures
MEMORY_REDUCTION_FACTOR = 0.7  # Reduce memory by 30% on each retry
MAX_ALLOCATION_RETRIES = 5     # Maximum retries for memory allocation

# Global variables for interrupt handling
interrupted = False
current_rules_list = None
current_ranking_output_path = None
current_top_k = 0
words_processed_total = None
total_unique_found = None
total_cracked_found = None

# Rule IDs (updated for larger capacity)
START_ID_SIMPLE = 0
NUM_SIMPLE_RULES = 10
START_ID_TD = 10
NUM_TD_RULES = 20
START_ID_S = 30
NUM_S_RULES = 256 * 256
START_ID_A = 30 + NUM_S_RULES
NUM_A_RULES = 3 * 256
START_ID_GROUPB = START_ID_A + NUM_A_RULES
NUM_GROUPB_RULES = 13
START_ID_NEW = START_ID_GROUPB + NUM_GROUPB_RULES
NUM_NEW_RULES = 13

# ====================================================================
# --- OPTIMIZED FILE LOADING FUNCTIONS ---
# ====================================================================

def estimate_word_count(path):
    """Fast word count estimation for large files without reading entire content"""
    print(f"{blue('📊')} {bold('Estimating words in:')} {path}...")
    
    try:
        file_size = os.path.getsize(path)
        # Sample first 10MB to estimate average line length
        sample_size = min(10 * 1024 * 1024, file_size)
        
        with open(path, 'rb') as f:
            sample = f.read(sample_size)
            lines = sample.count(b'\n')
            
            if file_size <= sample_size:
                # Small file, exact count
                total_lines = lines
            else:
                # Estimate based on sample
                avg_line_length = sample_size / max(lines, 1)
                total_lines = int(file_size / avg_line_length)
                
        print(f"{green('✅')} {bold('Estimated words:')} {cyan(f'{total_lines:,}')}")
        return total_lines
        
    except Exception as e:
        print(f"{yellow('⚠️')} {bold('Could not estimate word count:')} {e}")
        return 1000000  # Fallback

def fast_fnv1a_hash_32(data):
    """Optimized FNV-1a hash for bytes"""
    if isinstance(data, np.ndarray):
        # Use numpy for bulk processing if available
        hash_val = np.uint32(2166136261)
        for byte in data:
            hash_val ^= np.uint32(byte)
            hash_val *= np.uint32(16777619)
        return hash_val
    else:
        # Standard implementation for bytes
        hash_val = 2166136261
        for byte in data:
            hash_val = (hash_val ^ byte) * 16777619 & 0xFFFFFFFF
        return hash_val

def bulk_hash_words(words_list):
    """Compute hashes for multiple words in bulk"""
    return [fast_fnv1a_hash_32(word) for word in words_list]

def optimized_wordlist_iterator(wordlist_path, max_len, batch_size):
    """
    Massively optimized wordlist loader using memory mapping and bulk processing
    """
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
                
                # Use a progress bar that updates based on file position
                with tqdm(total=file_size, desc="Loading wordlist", unit="B", unit_scale=True) as pbar:
                    while pos < file_size and not interrupted:
                        # Find next newline efficiently
                        end_pos = mm.find(b'\n', pos)
                        if end_pos == -1:
                            end_pos = file_size
                        
                        # Extract line directly from memory map
                        line = mm[pos:end_pos].strip()
                        line_len = len(line)
                        pos = end_pos + 1
                        pbar.update(pos - pbar.n)  # Update progress
                        
                        # Skip empty lines and validate length
                        if line_len == 0 or line_len > max_len:
                            continue
                            
                        # Copy to buffer
                        start_idx = batch_count * max_len
                        end_idx = start_idx + line_len
                        
                        # Use memoryview for efficient copying
                        words_buffer[start_idx:end_idx] = np.frombuffer(line, dtype=np.uint8, count=line_len)
                        
                        # Compute hash
                        hashes_buffer[batch_count] = fast_fnv1a_hash_32(line)
                        batch_count += 1
                        total_words_loaded += 1
                        
                        # Yield batch when full
                        if batch_count >= batch_size:
                            yield words_buffer.copy(), hashes_buffer.copy(), batch_count
                            batch_count = 0
                            words_buffer.fill(0)
                            hashes_buffer.fill(0)
                
                # Yield final partial batch
                if batch_count > 0 and not interrupted:
                    yield words_buffer, hashes_buffer, batch_count
                    
    except Exception as e:
        print(f"{red('❌')} {bold('Error in optimized loader:')} {e}")
        raise
    
    load_time = time() - load_start
    print(f"{green('✅')} {bold('Optimized loading completed:')} {cyan(f'{total_words_loaded:,}')} {bold('words in')} {load_time:.2f}s "
          f"({total_words_loaded/load_time:,.0f} words/sec)")

def parallel_hash_computation(words_batch, max_len, batch_count):
    """Compute hashes in parallel using multiple threads"""
    hashes = np.zeros(batch_count, dtype=np.uint32)
    
    def compute_hash(i):
        start = i * max_len
        end = start + max_len
        word_data = words_batch[start:end]
        # Find actual length (first zero byte)
        actual_len = np.argmax(word_data == 0)
        if actual_len == 0:
            actual_len = max_len
        return fast_fnv1a_hash_32(word_data[:actual_len])
    
    # Use ThreadPoolExecutor for parallel hash computation
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(compute_hash, range(batch_count)))
    
    hashes[:] = results
    return hashes

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
        # Create intermediate output path
        base_path = os.path.splitext(current_ranking_output_path)[0]
        intermediate_output_path = f"{base_path}_INTERRUPTED.csv"
        intermediate_optimized_path = f"{base_path}_INTERRUPTED.rule"
        
        # Save current ranking data
        if current_rules_list:
            print(f"{blue('💾')} {bold('Saving intermediate results to:')} {intermediate_output_path}")
            
            # Calculate combined score for current progress
            for rule in current_rules_list:
                rule['combined_score'] = rule.get('effectiveness_score', 0) * 10 + rule.get('uniqueness_score', 0)
            
            # Save all rules regardless of score
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
            
            # Save optimized rules if requested
            if current_top_k > 0:
                print(f"{blue('💾')} {bold('Saving intermediate optimized rules to:')} {intermediate_optimized_path}")
                
                available_rules = len(ranked_rules)
                final_count = min(current_top_k, available_rules)
                
                with open(intermediate_optimized_path, 'w', newline='\n', encoding='utf-8') as f:
                    f.write(":\n")  # Default rule
                    for rule in ranked_rules[:final_count]:
                        f.write(f"{rule['rule_data']}\n")
                
                print(f"{green('✅')} {bold('Intermediate optimized rules saved:')} {cyan(f'{final_count:,}')} {bold('rules')}")
        
        # Print progress summary
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
            
        print(f"{green('✅')} {bold('Progress saved successfully. You can resume later using the intermediate files.')}")
        
    except Exception as e:
        print(f"{red('❌')} {bold('Error saving intermediate progress:')} {e}")
    
    sys.exit(0)

def setup_interrupt_handler(rules_list, ranking_output_path, top_k):
    """Setup interrupt handler with current context"""
    global current_rules_list, current_ranking_output_path, current_top_k
    current_rules_list = rules_list
    current_ranking_output_path = ranking_output_path
    current_top_k = top_k
    
    # Setup signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

def update_progress_stats(words_processed, unique_found, cracked_found):
    """Update progress statistics for interrupt handler"""
    global words_processed_total, total_unique_found, total_cracked_found
    words_processed_total = words_processed
    total_unique_found = unique_found
    total_cracked_found = cracked_found

# ====================================================================
# --- ORIGINAL HELPER FUNCTIONS (OPTIMIZED) ---
# ====================================================================

def load_rules(path):
    """Loads Hashcat rules from file."""
    print(f"{blue('📊')} {bold('Loading rules from:')} {path}...")
    
    # Get file size for rules too
    try:
        rules_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
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
    
    # Get file size
    try:
        cracked_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
        if cracked_size > 50:
            print(f"{yellow('📁')} {bold('Large cracked list detected:')} {cracked_size:.1f} MB - loading...")
    except OSError:
        cracked_size = 0
    
    cracked_hashes = []
    load_start = time()
    
    try:
        # Use optimized loading for cracked list too
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
    
    # Pack up to 4 bytes into first integer
    for i, byte in enumerate(rule_chars[:4]):
        args_int |= (byte << (i * 8))
    
    encoded[1] = np.uint32(args_int)
    
    # Pack remaining bytes into second integer
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

    # Calculate a combined score for ranking
    for rule in ranking_list:
        rule['combined_score'] = rule.get('effectiveness_score', 0) * 10 + rule.get('uniqueness_score', 0)

    # FIXED: Save ALL rules regardless of score
    ranked_rules = ranking_list  # Save ALL rules instead of filtering
    
    ranked_rules.sort(key=lambda rule: rule['combined_score'], reverse=True)

    print(f"{blue('💾')} {bold('Saving ALL')} {cyan(f'{len(ranked_rules):,}')} {bold('rules (including zero-score rules)')}")

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

    # FIXED: Include ALL rules from CSV, don't filter by score
    print(f"{blue('📊')} {bold('Loaded')} {cyan(f'{len(ranked_data):,}')} {bold('total rules from CSV')}")

    ranked_data.sort(key=lambda row: row['Combined_Score'], reverse=True)
    
    # FIXED: Cap at available rules if top_k exceeds available count
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
            f.write(":\n")  # Default rule
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
        # Try to get global memory size
        total_memory = device.global_mem_size
        # For available memory, we'll use a conservative estimate
        # (OpenCL doesn't provide reliable available memory reporting)
        available_memory = int(total_memory * (1 - VRAM_SAFETY_MARGIN))
        return total_memory, available_memory
    except Exception as e:
        print(f"{yellow('⚠️')} {bold('Warning: Could not query GPU memory:')} {e}")
        # Fallback to conservative defaults
        return 8 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024  # 8GB total, 6GB available

def calculate_optimal_parameters_large_rules(available_vram, total_words, cracked_hashes_count, total_rules, reduction_factor=1.0):
    """
    Calculate optimal parameters with consideration for large rule sets
    """
    print(f"{blue('🔧')} {bold('Calculating optimal parameters for')} {cyan(f'{available_vram / (1024**3):.1f} GB')} {bold('available VRAM')}")
    print(f"{blue('📊')} {bold('Dataset:')} {cyan(f'{total_words:,}')} {bold('words,')} {cyan(f'{total_rules:,}')} {bold('rules,')} {cyan(f'{cracked_hashes_count:,}')} {bold('cracked hashes')}")
    
    if reduction_factor < 1.0:
        print(f"{yellow('📉')} {bold('Applying memory reduction factor:')} {cyan(f'{reduction_factor:.2f}')}")
    
    # Apply reduction factor
    available_vram = int(available_vram * reduction_factor)
    
    # Memory requirements per component (in bytes)
    word_batch_bytes = MAX_WORD_LEN * np.uint8().itemsize
    hash_batch_bytes = np.uint32().itemsize
    rule_batch_bytes = MAX_RULES_IN_BATCH * (2 + MAX_RULE_ARGS) * np.uint32().itemsize
    counter_bytes = MAX_RULES_IN_BATCH * np.uint32().itemsize * 2
    
    # Base memory needed (excluding hash maps)
    base_memory = (
        (word_batch_bytes + hash_batch_bytes) * 2 +  # Double buffering
        rule_batch_bytes + counter_bytes
    )
    
    # Adjust batch size based on rule count to avoid too many iterations
    if total_rules > 100000:
        # For very large rule sets, use smaller batches to fit in memory
        suggested_batch_size = min(DEFAULT_WORDS_PER_GPU_BATCH, 100000)
    else:
        suggested_batch_size = DEFAULT_WORDS_PER_GPU_BATCH
    
    # Available memory for hash maps
    available_for_maps = available_vram - base_memory
    if available_for_maps <= 0:
        print(f"{yellow('⚠️')}  {bold('Warning: Limited VRAM, using minimal configuration')}")
        available_for_maps = available_vram * 0.5
    
    print(f"{blue('📊')} {bold('Available for hash maps:')} {cyan(f'{available_for_maps / (1024**3):.2f} GB')}")
    
    # Calculate hash map sizes based on dataset size and available memory
    global_bits = DEFAULT_GLOBAL_HASH_MAP_BITS
    cracked_bits = DEFAULT_CRACKED_HASH_MAP_BITS
    
    # Adjust global hash map based on wordlist size
    if total_words > 0:
        required_global_bits = max(MIN_HASH_MAP_BITS, math.ceil(math.log2(total_words)) + 8)
        global_bits = min(required_global_bits, DEFAULT_GLOBAL_HASH_MAP_BITS)
    
    # Adjust cracked hash map based on cracked list size
    if cracked_hashes_count > 0:
        required_cracked_bits = max(MIN_HASH_MAP_BITS, math.ceil(math.log2(cracked_hashes_count)) + 8)
        cracked_bits = min(required_cracked_bits, DEFAULT_CRACKED_HASH_MAP_BITS)
    
    # Calculate memory usage for hash maps
    global_map_bytes = (1 << (global_bits - 5)) * np.uint32().itemsize
    cracked_map_bytes = (1 << (cracked_bits - 5)) * np.uint32().itemsize
    total_map_memory = global_map_bytes + cracked_map_bytes
    
    # Reduce bits if maps exceed available memory
    while total_map_memory > available_for_maps and global_bits > MIN_HASH_MAP_BITS and cracked_bits > MIN_HASH_MAP_BITS:
        if global_bits > cracked_bits:
            global_bits -= 1
        else:
            cracked_bits -= 1
        
        global_map_bytes = (1 << (global_bits - 5)) * np.uint32().itemsize
        cracked_map_bytes = (1 << (cracked_bits - 5)) * np.uint32().itemsize
        total_map_memory = global_map_bytes + cracked_map_bytes
    
    # Calculate optimal batch size considering rule count
    memory_per_word = (
        word_batch_bytes +  # base words
        hash_batch_bytes +  # base hashes
        (MAX_OUTPUT_LEN * np.uint8().itemsize) +  # result temp
        (rule_batch_bytes / MAX_RULES_IN_BATCH)  # rule memory per word
    )
    
    max_batch_by_memory = int((available_vram - total_map_memory - base_memory) / memory_per_word)
    optimal_batch_size = min(suggested_batch_size, max_batch_by_memory)
    optimal_batch_size = max(MIN_BATCH_SIZE, optimal_batch_size)
    
    # Round batch size to nearest multiple of LOCAL_WORK_SIZE for better performance
    optimal_batch_size = (optimal_batch_size // LOCAL_WORK_SIZE) * LOCAL_WORK_SIZE
    
    # Adjust for very large rule sets
    if total_rules > 50000:
        # Reduce batch size slightly to accommodate more rules in memory
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
            "batch_size": 50000,
            "global_bits": 30,
            "cracked_bits": 28
        },
        "medium_memory": {
            "description": "Medium Memory Mode (for GPUs with 4-8GB VRAM)",
            "batch_size": 150000,
            "global_bits": 33,
            "cracked_bits": 31
        },
        "high_memory": {
            "description": "High Memory Mode (for GPUs with > 8GB VRAM)",
            "batch_size": 300000,
            "global_bits": 35,
            "cracked_bits": 33
        },
        "auto": {
            "description": "Auto-calculated (Recommended)",
            "batch_size": None,
            "global_bits": None,
            "cracked_bits": None
        }
    }
    
    # Determine which preset to recommend
    if total_vram < 4 * 1024**3:
        recommended_preset = "low_memory"
    elif total_vram < 8 * 1024**3:
        recommended_preset = "medium_memory"
    else:
        recommended_preset = "high_memory"
    
    # Calculate auto parameters
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
    Returns: dict of buffer names to buffer objects
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
                # Clean up any partially allocated buffers
                for buf in buffers.values():
                    try:
                        buf.release()
                    except:
                        pass
                buffers = {}
            else:
                raise e
                
    raise cl.MemoryError(f"{red('❌')} {bold('Failed to allocate buffers after')} {cyan(f'{max_retries}')} {bold('retries')}")

# ====================================================================
# --- KERNEL SOURCE ---
# ====================================================================
def get_kernel_source(global_hash_map_bits, cracked_hash_map_bits):
    global_hash_map_mask = (1 << (global_hash_map_bits - 5)) - 1
    cracked_hash_map_mask = (1 << (cracked_hash_map_bits - 5)) - 1
    
    return f"""
// FNV-1a Hash implementation in OpenCL
unsigned int fnv1a_hash_32(const unsigned char* data, unsigned int len) {{
    unsigned int hash = 2166136261U;
    for (unsigned int i = 0; i < len; i++) {{
        hash ^= data[i];
        hash *= 16777619U;
    }}
    return hash;
}}

// Helper function to convert char digit/letter to int position
unsigned int char_to_pos(unsigned char c) {{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    return 0xFFFFFFFF; 
}}

__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void hash_map_init_kernel(
    __global unsigned int* global_hash_map,
    __global const unsigned int* base_hashes,
    const unsigned int num_hashes,
    const unsigned int map_mask)
{{
    unsigned int global_id = get_global_id(0);
    if (global_id >= num_hashes) return;

    unsigned int word_hash = base_hashes[global_id];
    unsigned int map_index = (word_hash >> 5) & map_mask;
    unsigned int bit_index = word_hash & 31;
    unsigned int set_bit = (1U << bit_index);

    atomic_or(&global_hash_map[map_index], set_bit);
}}

__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void bfs_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned int* rules_in,
    __global unsigned int* rule_uniqueness_counts,
    __global unsigned int* rule_effectiveness_counts,
    __global const unsigned int* global_hash_map,
    __global const unsigned int* cracked_hash_map,
    const unsigned int num_words,
    const unsigned int num_rules_in_batch,
    const unsigned int max_word_len,
    const unsigned int max_output_len,
    const unsigned int global_map_mask,
    const unsigned int cracked_map_mask)
{{
    unsigned int global_id = get_global_id(0);

    unsigned int word_per_rule_count = num_words * num_rules_in_batch;
    if (global_id >= word_per_rule_count) return;

    unsigned int word_idx = global_id / num_rules_in_batch;
    unsigned int rule_batch_idx = global_id % num_rules_in_batch;

    // --- Constant Setup ---
    unsigned int start_id_simple = {START_ID_SIMPLE};
    unsigned int end_id_simple = start_id_simple + {NUM_SIMPLE_RULES};
    unsigned int start_id_TD = {START_ID_TD};
    unsigned int end_id_TD = start_id_TD + {NUM_TD_RULES};
    unsigned int start_id_s = {START_ID_S};
    unsigned int end_id_s = start_id_s + {NUM_S_RULES};
    unsigned int start_id_A = {START_ID_A};
    unsigned int end_id_A = start_id_A + {NUM_A_RULES};
    unsigned int start_id_groupB = {START_ID_GROUPB};
    unsigned int end_id_groupB = start_id_groupB + {NUM_GROUPB_RULES};
    unsigned int start_id_new = {START_ID_NEW};
    unsigned int end_id_new = start_id_new + {NUM_NEW_RULES};

    unsigned char result_temp[{MAX_OUTPUT_LEN}];
    
    __global const unsigned char* current_word_ptr = base_words_in + word_idx * max_word_len;
    unsigned int rule_size_in_int = 2 + {MAX_RULE_ARGS};
    __global const unsigned int* current_rule_ptr_int = rules_in + rule_batch_idx * rule_size_in_int;

    unsigned int rule_id = current_rule_ptr_int[0];
    unsigned int rule_args_int = current_rule_ptr_int[1];
    unsigned int rule_args_int2 = current_rule_ptr_int[2];
    unsigned int rule_args_int3 = current_rule_ptr_int[3];

    unsigned int word_len = 0;
    for (unsigned int i = 0; i < max_word_len; i++) {{
        if (current_word_ptr[i] == 0) {{
            word_len = i;
            break;
        }}
    }}

    if (word_len == 0 && rule_id < start_id_A) return;
    unsigned int out_len = 0;
    bool changed_flag = false;

    for(unsigned int i = 0; i < max_output_len; i++) result_temp[i] = 0;

    // --- Unpack arguments ---
    unsigned char arg0 = (unsigned char)(rule_args_int & 0xFF);
    unsigned char arg1 = (unsigned char)((rule_args_int >> 8) & 0xFF);
    unsigned char arg2 = (unsigned char)((rule_args_int >> 16) & 0xFF);
    unsigned char arg3 = (unsigned char)(rule_args_int2 & 0xFF);

    // --- RULE APPLICATION LOGIC ---
    if (rule_id >= start_id_simple && rule_id < end_id_simple) {{
        switch(rule_id - start_id_simple) {{
            case 0: {{ // 'l' (lowercase)
                out_len = word_len;
                for (unsigned int i = 0; i < word_len; i++) {{
                    unsigned char c = current_word_ptr[i];
                    if (c >= 'A' && c <= 'Z') {{
                        result_temp[i] = c + 32;
                        changed_flag = true;
                    }} else {{
                        result_temp[i] = c;
                    }}
                }}
                break;
            }}
            case 1: {{ // 'u' (uppercase)
                out_len = word_len;
                for (unsigned int i = 0; i < word_len; i++) {{
                    unsigned char c = current_word_ptr[i];
                    if (c >= 'a' && c <= 'z') {{
                        result_temp[i] = c - 32;
                        changed_flag = true;
                    }} else {{
                        result_temp[i] = c;
                    }}
                }}
                break;
            }}
            case 2: {{ // 'c' (capitalize)
                out_len = word_len;
                if (word_len > 0) {{
                    if (current_word_ptr[0] >= 'a' && current_word_ptr[0] <= 'z') {{
                        result_temp[0] = current_word_ptr[0] - 32;
                        changed_flag = true;
                    }} else {{
                        result_temp[0] = current_word_ptr[0];
                    }}
                    for (unsigned int i = 1; i < word_len; i++) {{
                        unsigned char c = current_word_ptr[i];
                        if (c >= 'A' && c <= 'Z') {{
                            result_temp[i] = c + 32;
                            changed_flag = true;
                        }} else {{
                            result_temp[i] = c;
                        }}
                    }}
                }}
                break;
            }}
            case 3: {{ // 'C' (invert capitalize)
                out_len = word_len;
                if (word_len > 0) {{
                    if (current_word_ptr[0] >= 'A' && current_word_ptr[0] <= 'Z') {{
                        result_temp[0] = current_word_ptr[0] + 32;
                        changed_flag = true;
                    }} else {{
                        result_temp[0] = current_word_ptr[0];
                    }}
                    for (unsigned int i = 1; i < word_len; i++) {{
                        unsigned char c = current_word_ptr[i];
                        if (c >= 'a' && c <= 'z') {{
                            result_temp[i] = c - 32;
                            changed_flag = true;
                        }} else {{
                            result_temp[i] = c;
                        }}
                    }}
                }}
                break;
            }}
            case 4: {{ // 't' (toggle case)
                out_len = word_len;
                for (unsigned int i = 0; i < word_len; i++) {{
                    unsigned char c = current_word_ptr[i];
                    if (c >= 'a' && c <= 'z') {{
                        result_temp[i] = c - 32;
                        changed_flag = true;
                    }} else if (c >= 'A' && c <= 'Z') {{
                        result_temp[i] = c + 32;
                        changed_flag = true;
                    }} else {{
                        result_temp[i] = c;
                    }}
                }}
                break;
            }}
            case 5: {{ // 'r' (reverse)
                out_len = word_len;
                if (word_len > 1) {{
                    for (unsigned int i = 0; i < word_len; i++) {{
                        result_temp[i] = current_word_ptr[word_len - 1 - i];
                    }}
                    for (unsigned int i = 0; i < word_len; i++) {{
                        if (result_temp[i] != current_word_ptr[i]) {{
                            changed_flag = true;
                            break;
                        }}
                    }}
                }} else {{
                    for (unsigned int i = 0; i < word_len; i++) {{
                        result_temp[i] = current_word_ptr[i];
                    }}
                }}
                break;
            }}
            case 6: {{ // 'k' (swap first two chars)
                out_len = word_len;
                for(unsigned int i=0; i<word_len; i++) result_temp[i] = current_word_ptr[i];
                if (word_len >= 2) {{
                    result_temp[0] = current_word_ptr[1];
                    result_temp[1] = current_word_ptr[0];
                    changed_flag = true;
                }}
                break;
            }}
            case 7: {{ // ':' (identity/no change)
                out_len = word_len;
                for(unsigned int i=0; i<word_len; i++) result_temp[i] = current_word_ptr[i];
                changed_flag = false;
                break;
            }}
            case 8: {{ // 'd' (duplicate)
                out_len = word_len * 2;
                if (out_len >= max_output_len) {{
                    out_len = 0;
                    changed_flag = false;
                    break;
                }}
                for(unsigned int i=0; i<word_len; i++) {{
                    result_temp[i] = current_word_ptr[i];
                    result_temp[word_len+i] = current_word_ptr[i];
                }}
                changed_flag = true;
                break;
            }}
            case 9: {{ // 'f' (reflect: word + reverse(word))
                out_len = word_len * 2;
                if (out_len >= max_output_len) {{
                    out_len = 0;
                    changed_flag = false;
                    break;
                }}
                for(unsigned int i=0; i<word_len; i++) {{
                    result_temp[i] = current_word_ptr[i];
                    result_temp[word_len+i] = current_word_ptr[word_len-1-i];
                }}
                changed_flag = true;
                break;
            }}
        }}
    }} else if (rule_id >= start_id_TD && rule_id < end_id_TD) {{
        unsigned char operator_char = arg0;
        unsigned int pos_to_change = char_to_pos(arg1);
        
        if (operator_char == 'T') {{
            out_len = word_len;
            for (unsigned int i = 0; i < word_len; i++) {{
                result_temp[i] = current_word_ptr[i];
            }}
            if (pos_to_change != 0xFFFFFFFF && pos_to_change < word_len) {{
                unsigned char c = current_word_ptr[pos_to_change];
                if (c >= 'a' && c <= 'z') {{
                    result_temp[pos_to_change] = c - 32;
                    changed_flag = true;
                }} else if (c >= 'A' && c <= 'Z') {{
                    result_temp[pos_to_change] = c + 32;
                    changed_flag = true;
                }}
            }}
        }}
        else if (operator_char == 'D') {{
            unsigned int out_idx = 0;
            if (pos_to_change != 0xFFFFFFFF && pos_to_change < word_len) {{
                for (unsigned int i = 0; i < word_len; i++) {{
                    if (i != pos_to_change) {{
                        result_temp[out_idx++] = current_word_ptr[i];
                    }} else {{
                        changed_flag = true;
                    }}
                }}
            }} else {{
                for (unsigned int i = 0; i < word_len; i++) {{
                    result_temp[i] = current_word_ptr[i];
                }}
                out_idx = word_len;
            }}
            out_len = out_idx;
        }}
    }}
    else if (rule_id >= start_id_s && rule_id < end_id_s) {{
        out_len = word_len;
        for(unsigned int i=0; i<word_len; i++) result_temp[i] = current_word_ptr[i];
        
        unsigned char find = arg0;
        unsigned char replace = arg1;
        for(unsigned int i = 0; i < word_len; i++) {{
            if (current_word_ptr[i] == find) {{
                result_temp[i] = replace;
                changed_flag = true;
            }}
        }}
    }} else if (rule_id >= start_id_A && rule_id < end_id_A) {{
        out_len = word_len;
        for(unsigned int i=0; i<word_len; i++) result_temp[i] = current_word_ptr[i];
        
        unsigned char cmd = arg0;
        unsigned char arg = arg1;
        
        if (cmd == '^') {{
            if (word_len + 1 >= max_output_len) {{
                out_len = 0;
                changed_flag = false;
            }} else {{
                for(unsigned int i=word_len; i>0; i--) {{
                    result_temp[i] = result_temp[i-1];
                }}
                result_temp[0] = arg;
                out_len++;
                changed_flag = true;
            }}
        }} else if (cmd == '$') {{
            if (word_len + 1 >= max_output_len) {{
                out_len = 0;
                changed_flag = false;
            }} else {{
                result_temp[out_len] = arg;
                out_len++;
                changed_flag = true;
            }}
        }} else if (cmd == '@') {{
            unsigned int temp_idx = 0;
            for(unsigned int i=0; i<word_len; i++) {{
                if (result_temp[i] != arg) {{
                    result_temp[temp_idx++] = result_temp[i];
                }} else {{
                    changed_flag = true;
                }}
            }}
            out_len = temp_idx;
        }}
    }}
    // --- GROUP B RULES ---
    else if (rule_id >= start_id_groupB && rule_id < end_id_groupB) {{ 
        
        for(unsigned int i=0; i<word_len; i++) result_temp[i] = current_word_ptr[i];
        out_len = word_len;

        unsigned char cmd = arg0;
        unsigned int N = (arg1 != 0) ? char_to_pos(arg1) : 0xFFFFFFFF;
        unsigned int M = (arg2 != 0) ? char_to_pos(arg2) : 0xFFFFFFFF;
        unsigned char X = arg2;

        if (cmd == 'p') {{ // 'p' (Duplicate N times)
            if (N != 0xFFFFFFFF) {{
                unsigned int num_dupes = N;
                unsigned int total_len = word_len * (num_dupes + 1); 

                if (total_len >= max_output_len || num_dupes == 0) {{
                    out_len = 0; 
                }} else {{
                    for (unsigned int j = 1; j <= num_dupes; j++) {{
                        unsigned int offset = word_len * j;
                        for (unsigned int i = 0; i < word_len; i++) {{
                            result_temp[offset + i] = current_word_ptr[i];
                        }}
                    }}
                    out_len = total_len;
                    changed_flag = true;
                }}
            }}
        }} 
        
        else if (cmd == 'q') {{ // 'q' (Duplicate all characters)
            unsigned int total_len = word_len * 2;
            if (total_len >= max_output_len) {{
                out_len = 0;
            }} else {{
                for (unsigned int i = 0; i < word_len; i++) {{
                    result_temp[i * 2] = current_word_ptr[i];
                    result_temp[i * 2 + 1] = current_word_ptr[i];
                }}
                out_len = total_len;
                changed_flag = true;
            }}
        }}

        else if (cmd == '{{') {{ // '{{' (Rotate Left)
            if (word_len > 0) {{
                unsigned char first_char = current_word_ptr[0];
                for (unsigned int i = 0; i < word_len - 1; i++) {{
                    result_temp[i] = current_word_ptr[i + 1];
                }}
                result_temp[word_len - 1] = first_char;
                changed_flag = true;
            }}
        }} 
        
        else if (cmd == '}}') {{ // '}}' (Rotate Right)
            if (word_len > 0) {{
                unsigned char last_char = current_word_ptr[word_len - 1];
                for (unsigned int i = word_len - 1; i > 0; i--) {{
                    result_temp[i] = current_word_ptr[i - 1];
                }}
                result_temp[0] = last_char;
                changed_flag = true;
            }}
        }}
        
        else if (cmd == '[') {{ // '[' (Truncate Left)
            if (word_len > 0) {{
                out_len = word_len - 1;
                changed_flag = true;
            }}
        }} 
        
        else if (cmd == ']') {{ // ']' (Truncate Right)
            if (word_len > 0) {{
                out_len = word_len - 1;
                changed_flag = true;
            }}
        }} 
        
        else if (cmd == 'x') {{ // 'xNM' (Extract range)
            unsigned int start = N;
            unsigned int length = M;
            
            if (start != 0xFFFFFFFF && length != 0xFFFFFFFF && start < word_len && length > 0) {{
                unsigned int end = start + length;
                if (end > word_len) end = word_len;
                
                out_len = 0;
                for (unsigned int i = start; i < end; i++) {{
                    result_temp[out_len++] = current_word_ptr[i];
                }}
                changed_flag = true;
            }} else {{
                out_len = 0; 
            }}
        }}

        else if (cmd == 'i') {{ // 'iNX' (Insert char)
            unsigned int pos = N;
            unsigned char insert_char = X;

            if (pos != 0xFFFFFFFF && word_len + 1 < max_output_len) {{
                unsigned int final_pos = (pos > word_len) ? word_len : pos;
                out_len = word_len + 1;

                unsigned int current_idx = 0;
                for (unsigned int i = 0; i < out_len; i++) {{
                    if (i == final_pos) {{
                        result_temp[i] = insert_char;
                    }} else {{
                        result_temp[i] = current_word_ptr[current_idx++];
                    }}
                }}
                changed_flag = true;
            }} else {{
                out_len = 0;
            }}
        }}

        else if (cmd == 'o') {{ // 'oNX' (Overwrite char)
            unsigned int pos = N;
            unsigned char new_char = X;

            if (pos != 0xFFFFFFFF && pos < word_len) {{
                result_temp[pos] = new_char;
                changed_flag = true;
            }}
        }}

    }}
    // --- NEW COMPREHENSIVE RULES ---
    else if (rule_id >= start_id_new && rule_id < end_id_new) {{ 
        
        for(unsigned int i=0; i<word_len; i++) result_temp[i] = current_word_ptr[i];
        out_len = word_len;

        unsigned char cmd = arg0;
        unsigned int N = (arg1 != 0) ? char_to_pos(arg1) : 0xFFFFFFFF;
        unsigned int M = (arg2 != 0) ? char_to_pos(arg2) : 0xFFFFFFFF;
        unsigned char X = arg2;

        if (cmd == 'K') {{ // 'K' (Swap last two characters)
            if (word_len >= 2) {{
                result_temp[word_len - 1] = current_word_ptr[word_len - 2];
                result_temp[word_len - 2] = current_word_ptr[word_len - 1];
                changed_flag = true;
            }}
        }}
        else if (cmd == '*') {{ // '*NM' (Swap characters)
            if (N != 0xFFFFFFFF && M != 0xFFFFFFFF && N < word_len && M < word_len && N != M) {{
                unsigned char temp = result_temp[N];
                result_temp[N] = result_temp[M];
                result_temp[M] = temp;
                changed_flag = true;
            }}
        }}
        else if (cmd == 'E') {{ // 'E' (Title case)
            bool capitalize_next = true;
            for (unsigned int i = 0; i < word_len; i++) {{
                unsigned char c = current_word_ptr[i];
                if (c >= 'A' && c <= 'Z') {{
                    result_temp[i] = c + 32;
                    c = result_temp[i];
                }} else {{
                    result_temp[i] = c;
                }}
                
                if (capitalize_next && c >= 'a' && c <= 'z') {{
                    result_temp[i] = c - 32;
                    changed_flag = true;
                }}
                capitalize_next = (result_temp[i] == ' ');
            }}
            // REMOVED THE EXTRA BREAK STATEMENT THAT WAS CAUSING THE ERROR
        }}
    }}

    // --- DUAL-UNIQUENESS LOGIC ---
    if (changed_flag && out_len > 0) {{
        unsigned int word_hash = fnv1a_hash_32(result_temp, out_len);

        // 1. Check against Base Wordlist (Uniqueness)
        unsigned int global_map_index = (word_hash >> 5) & {global_hash_map_mask};
        unsigned int bit_index = word_hash & 31;
        unsigned int check_bit = (1U << bit_index);

        __global const unsigned int* global_map_ptr = &global_hash_map[global_map_index];
        unsigned int current_global_word = *global_map_ptr;

        if (!(current_global_word & check_bit)) {{
            atomic_inc(&rule_uniqueness_counts[rule_batch_idx]);

            // 2. Check against Cracked List (Effectiveness)
            unsigned int cracked_map_index = (word_hash >> 5) & {cracked_hash_map_mask};
            __global const unsigned int* cracked_map_ptr = &cracked_hash_map[cracked_map_index];
            unsigned int current_cracked_word = *cracked_map_ptr;

            if (current_cracked_word & check_bit) {{
                atomic_inc(&rule_effectiveness_counts[rule_batch_idx]);
            }}
        }}
    }}
}}
"""

# ====================================================================
# --- UPDATED MAIN RANKING FUNCTION WITH CONTINUOUS RULE PROCESSING ---
# ====================================================================

def rank_rules_uniqueness_large(wordlist_path, rules_path, cracked_list_path, ranking_output_path, top_k, 
                               words_per_gpu_batch=None, global_hash_map_bits=None, cracked_hash_map_bits=None,
                               preset=None):
    start_time = time()
    
    # 0. PRELIMINARY DATA LOADING FOR MEMORY CALCULATION
    total_words = estimate_word_count(wordlist_path)  # Use fast estimation
    rules_list = load_rules(rules_path)
    total_rules = len(rules_list)
    
    # Setup interrupt handler BEFORE starting processing
    setup_interrupt_handler(rules_list, ranking_output_path, top_k)
    
    # Check if we're dealing with a large rule set
    if total_rules > 100000:
        print(f"{green('🚀')} {bold('LARGE RULE SET DETECTED:')} {cyan(f'{total_rules:,}')} {bold('rules')}")
        print(f"   {bold('Using optimized processing for large rule sets...')}")
    
    # Load cracked hashes ONCE for both memory calculation and processing
    cracked_hashes_np = load_cracked_hashes(cracked_list_path, MAX_WORD_LEN)
    cracked_hashes_count = len(cracked_hashes_np)
    
    # 1. OPENCL INITIALIZATION AND MEMORY DETECTION
    try:
        platform = cl.get_platforms()[0]
        devices = platform.get_devices(cl.device_type.GPU)
        if not devices:
            devices = platform.get_devices(cl.device_type.ALL)
        device = devices[0]
        context = cl.Context([device])
        queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

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
            
            # Set defaults for any unspecified manual parameters
            if words_per_gpu_batch is None:
                words_per_gpu_batch = DEFAULT_WORDS_PER_GPU_BATCH
            if global_hash_map_bits is None:
                global_hash_map_bits = DEFAULT_GLOBAL_HASH_MAP_BITS
            if cracked_hash_map_bits is None:
                cracked_hash_map_bits = DEFAULT_CRACKED_HASH_MAP_BITS
                
            print(f"   {blue('-')} {bold('Batch size:')} {cyan(f'{words_per_gpu_batch:,}')}")
            print(f"   {blue('-')} {bold('Global hash map:')} {cyan(f'{global_hash_map_bits} bits')}")
            print(f"   {blue('-')} {bold('Cracked hash map:')} {cyan(f'{cracked_hash_map_bits} bits')}")
            
            # Validate manual parameters against available VRAM
            global_map_bytes = (1 << (global_hash_map_bits - 5)) * np.uint32().itemsize
            cracked_map_bytes = (1 << (cracked_hash_map_bits - 5)) * np.uint32().itemsize
            total_map_memory = global_map_bytes + cracked_map_bytes
            
            # Memory requirements for batch processing
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
            # Auto-calculate optimal parameters with large rule set consideration
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

        KERNEL_SOURCE = get_kernel_source(global_hash_map_bits, cracked_hash_map_bits)
        prg = cl.Program(context, KERNEL_SOURCE).build(options=["-cl-fast-relaxed-math"])
        
        # Use the correct kernel names (without _opt suffix)
        kernel_bfs = prg.bfs_kernel
        kernel_init = prg.hash_map_init_kernel
        
        print(f"{green('✅')} {bold('OpenCL initialized on device:')} {cyan(device.name.strip())}")
    except Exception as e:
        print(f"{red('❌')} {bold('OpenCL initialization or kernel compilation error:')} {e}")
        exit(1)

    # 2. DATA LOADING AND PRE-ENCODING
    rule_size_in_int = 2 + MAX_RULE_ARGS
    encoded_rules = [encode_rule(rule['rule_data'], rule['rule_id'], MAX_RULE_ARGS) for rule in rules_list]

    # 3. HASH MAP INITIALIZATION
    global_hash_map_np = np.zeros(GLOBAL_HASH_MAP_WORDS, dtype=np.uint32)
    print(f"{blue('📝')} {bold('Global Hash Map initialized:')} {cyan(f'{global_hash_map_np.nbytes / (1024*1024):.2f} MB')} {bold('allocated.')}")

    cracked_hash_map_np = np.zeros(CRACKED_HASH_MAP_WORDS, dtype=np.uint32)
    print(f"{blue('📝')} {bold('Cracked Hash Map initialized:')} {cyan(f'{cracked_hash_map_np.nbytes / (1024*1024):.2f} MB')} {bold('allocated.')}")

    # 4. OPENCL BUFFER SETUP WITH RETRY LOGIC
    mf = cl.mem_flags

    # Define counters_size here - this was the missing variable
    counters_size = MAX_RULES_IN_BATCH * np.uint32().itemsize

    # Define buffer specifications for retry logic
    buffer_specs = {
        # Double buffering for words and hashes
        'base_words_in_0': {
            'flags': mf.READ_ONLY,
            'size': words_per_gpu_batch * MAX_WORD_LEN * np.uint8().itemsize
        },
        'base_words_in_1': {
            'flags': mf.READ_ONLY,
            'size': words_per_gpu_batch * MAX_WORD_LEN * np.uint8().itemsize
        },
        'base_hashes_0': {
            'flags': mf.READ_ONLY,
            'size': words_per_gpu_batch * np.uint32().itemsize
        },
        'base_hashes_1': {
            'flags': mf.READ_ONLY,
            'size': words_per_gpu_batch * np.uint32().itemsize
        },
        # Rule input buffer (INCREASED CAPACITY)
        'rules_in': {
            'flags': mf.READ_ONLY,
            'size': MAX_RULES_IN_BATCH * rule_size_in_int * np.uint32().itemsize
        },
        # Global Hash Map (RW for base wordlist check)
        'global_hash_map': {
            'flags': mf.READ_WRITE,
            'size': global_hash_map_np.nbytes
        },
        # Cracked Hash Map (Read Only, filled once)
        'cracked_hash_map': {
            'flags': mf.READ_ONLY,
            'size': cracked_hash_map_np.nbytes
        },
        # Rule Counters (INCREASED CAPACITY)
        'rule_uniqueness_counts': {
            'flags': mf.READ_WRITE,
            'size': counters_size
        },
        'rule_effectiveness_counts': {
            'flags': mf.READ_WRITE,
            'size': counters_size
        }
    }

    # Add hostbuf for cracked hash map if available
    if cracked_hashes_np.size > 0:
        buffer_specs['cracked_temp'] = {
            'flags': mf.READ_ONLY | mf.COPY_HOST_PTR,
            'size': cracked_hashes_np.nbytes,
            'hostbuf': cracked_hashes_np
        }

    try:
        buffers = create_opencl_buffers_with_retry(context, buffer_specs)
        
        # Extract buffers for easier access
        base_words_in_g = [buffers['base_words_in_0'], buffers['base_words_in_1']]
        base_hashes_g = [buffers['base_hashes_0'], buffers['base_hashes_1']]
        rules_in_g = buffers['rules_in']
        global_hash_map_g = buffers['global_hash_map']
        cracked_hash_map_g = buffers['cracked_hash_map']
        rule_uniqueness_counts_g = buffers['rule_uniqueness_counts']
        rule_effectiveness_counts_g = buffers['rule_effectiveness_counts']
        cracked_temp_g = buffers.get('cracked_temp', None)
        
    except cl.MemoryError as e:
        print(f"{red('❌')} {bold('Fatal: Could not allocate GPU memory even after retries:')} {e}")
        print(f"{yellow('💡')} {bold('Try reducing batch size or hash map bits, or use a preset:')}")
        recommendations, _ = get_recommended_parameters(device, total_words, cracked_hashes_count)
        for preset_name, config in recommendations.items():
            if preset_name != "auto":
                print(f"   {bold('--preset')} {cyan(preset_name)}: {config['description']}")
        return

    current_word_buffer_idx = 0
    copy_events = [None, None]

    # 5. INITIALIZE CRACKED HASH MAP (ONCE)
    if cracked_hashes_np.size > 0 and cracked_temp_g is not None:
        global_size_init_cracked = (int(math.ceil(cracked_hashes_np.size / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
        local_size_init_cracked = (LOCAL_WORK_SIZE,)

        print(f"{blue('📊')} {bold('Populating static Cracked Hash Map on GPU...')}")
        kernel_init(queue, global_size_init_cracked, local_size_init_cracked,
                    cracked_hash_map_g,
                    cracked_temp_g,
                    np.uint32(cracked_hashes_np.size),
                    np.uint32(CRACKED_HASH_MAP_MASK)).wait()
        print(f"{green('✅')} {bold('Static Cracked Hash Map populated.')}")
    else:
        print(f"{yellow('⚠️')} {bold('Cracked list is empty, effectiveness scoring is disabled.')}")
        
    # 6. PIPELINED RANKING LOOP SETUP
    # USE OPTIMIZED LOADER INSTEAD OF OLD ONE
    word_iterator = optimized_wordlist_iterator(wordlist_path, MAX_WORD_LEN, words_per_gpu_batch)
    rule_batch_starts = list(range(0, total_rules, MAX_RULES_IN_BATCH))
    total_rule_batches = len(rule_batch_starts)
    
    print(f"{blue('📊')} {bold('Processing')} {cyan(f'{total_rules:,}')} {bold('rules in')} {cyan(f'{total_rule_batches:,}')} {bold('batches')} "
          f"{bold('(up to')} {cyan(f'{MAX_RULES_IN_BATCH:,}')} {bold('rules per batch)')}")
    
    # Use numpy arrays for counters to avoid overflow - CHANGE TO PYTHON INT
    words_processed_total = 0  # Changed from np.uint64 to Python int
    total_unique_found = 0     # Changed from np.uint64 to Python int
    total_cracked_found = 0    # Changed from np.uint64 to Python int
    
    mapped_uniqueness_np = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
    mapped_effectiveness_np = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
    
    num_words_batch_exec = 0
    
    # Update progress bar totals based on actual counts
    word_batch_pbar = tqdm(total=total_words, desc="Processing wordlist [Unique: 0 | Cracked: 0]", unit=" word")
    rule_batch_pbar = tqdm(total=0, desc="Rule batches processed", unit=" batch", bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    # A. Initial word batch load
    try:
        base_words_np_batch, base_hashes_np_batch, num_words_batch_exec = next(word_iterator)
        cl.enqueue_copy(queue, base_words_in_g[current_word_buffer_idx], base_words_np_batch)
        cl.enqueue_copy(queue, base_hashes_g[current_word_buffer_idx], base_hashes_np_batch).wait()
    except StopIteration:
        print(f"{yellow('⚠️')} {bold('Wordlist is empty or too small.')}")
        word_batch_pbar.close()
        rule_batch_pbar.close()
        return

    # B. Main Pipelined Loop
    wordlist_processing_complete = False
    current_rule_batch_idx = 0  # Track current rule batch
    
    # Initialize global hash map with first word batch
    print(f"{blue('📊')} {bold('Initializing global hash map with first word batch...')}")
    global_size_init_global = (int(math.ceil(num_words_batch_exec / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
    local_size_init_global = (LOCAL_WORK_SIZE,)
    kernel_init(queue, global_size_init_global, local_size_init_global,
                global_hash_map_g,
                base_hashes_g[current_word_buffer_idx],
                np.uint32(num_words_batch_exec),
                np.uint32(GLOBAL_HASH_MAP_MASK)).wait()
    print(f"{green('✅')} {bold('Global hash map initialized with first batch.')}")
    
    # Track processed word batches
    processed_word_batches = 0
    
    while current_rule_batch_idx < total_rule_batches:
        # Check for interrupt before processing
        if interrupted:
            break
        
        # Get current word batch index
        exec_idx = current_word_buffer_idx
        
        # 7. Process current rule batch with current word batch
        rule_start_index = rule_batch_starts[current_rule_batch_idx]
        rule_end_index = min(rule_start_index + MAX_RULES_IN_BATCH, total_rules)
        num_rules_in_batch = rule_end_index - rule_start_index

        current_rule_batch_list = encoded_rules[rule_start_index:rule_end_index]
        current_rules_np = np.concatenate(current_rule_batch_list)
        
        cl.enqueue_copy(queue, rules_in_g, current_rules_np, is_blocking=True)
        cl.enqueue_fill_buffer(queue, rule_uniqueness_counts_g, np.uint32(0), 0, counters_size)
        cl.enqueue_fill_buffer(queue, rule_effectiveness_counts_g, np.uint32(0), 0, counters_size)
        
        global_size = (num_words_batch_exec * num_rules_in_batch, )
        global_size_aligned = (int(math.ceil(global_size[0] / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)

        kernel_bfs.set_args(
            base_words_in_g[exec_idx],
            rules_in_g,
            rule_uniqueness_counts_g,
            rule_effectiveness_counts_g,
            global_hash_map_g,
            cracked_hash_map_g,
            np.uint32(num_words_batch_exec),
            np.uint32(num_rules_in_batch),
            np.uint32(MAX_WORD_LEN),
            np.uint32(MAX_OUTPUT_LEN),
            np.uint32(GLOBAL_HASH_MAP_MASK),
            np.uint32(CRACKED_HASH_MAP_MASK)
        )

        exec_event = cl.enqueue_nd_range_kernel(queue, kernel_bfs, global_size_aligned, (LOCAL_WORK_SIZE,))
        exec_event.wait()

        cl.enqueue_copy(queue, mapped_uniqueness_np, rule_uniqueness_counts_g, is_blocking=True)
        cl.enqueue_copy(queue, mapped_effectiveness_np, rule_effectiveness_counts_g, is_blocking=True)

        # Update counts for this rule batch
        batch_unique_found = 0
        batch_cracked_found = 0
        
        for i in range(num_rules_in_batch):
            rule_index = rule_start_index + i
            # Convert to Python int
            uniqueness_val = int(mapped_uniqueness_np[i])
            effectiveness_val = int(mapped_effectiveness_np[i])
            
            rules_list[rule_index]['uniqueness_score'] += uniqueness_val
            rules_list[rule_index]['effectiveness_score'] += effectiveness_val
            batch_unique_found += uniqueness_val
            batch_cracked_found += effectiveness_val
        
        # Update global totals
        total_unique_found += batch_unique_found
        total_cracked_found += batch_cracked_found
        
        # Update progress bars
        word_batch_pbar.set_description(
            f"Processing wordlist [Unique: {total_unique_found:,} | Cracked: {total_cracked_found:,} | Word Batch: {processed_word_batches+1}]"
        )
        word_batch_pbar.refresh()
        
        rule_batch_pbar.update(1)
        rule_batch_pbar.total = total_rule_batches
        rule_batch_pbar.set_description(f"Rule batches: {current_rule_batch_idx+1}/{total_rule_batches}")
        rule_batch_pbar.refresh()
        
        # Update interrupt handler stats
        update_progress_stats(words_processed_total, total_unique_found, total_cracked_found)
        
        # Move to next rule batch
        current_rule_batch_idx += 1
        
        # If we've processed all rule batches for this word batch, load next word batch
        if current_rule_batch_idx >= total_rule_batches and not wordlist_processing_complete:
            # Load next word batch
            try:
                next_batch_data = next(word_iterator)
                base_words_np_next, base_hashes_np_next, num_words_batch_next = next_batch_data
                
                # Switch buffers
                next_idx = 1 - current_word_buffer_idx
                cl.enqueue_copy(queue, base_words_in_g[next_idx], base_words_np_next)
                cl.enqueue_copy(queue, base_hashes_g[next_idx], base_hashes_np_next).wait()
                
                # Update global hash map with new word batch
                kernel_init(queue, global_size_init_global, local_size_init_global,
                            global_hash_map_g,
                            base_hashes_g[next_idx],
                            np.uint32(num_words_batch_next),
                            np.uint32(GLOBAL_HASH_MAP_MASK)).wait()
                
                # Update counters
                words_processed_total += num_words_batch_exec
                word_batch_pbar.update(num_words_batch_exec)
                
                # Switch to new buffer and reset rule processing
                current_word_buffer_idx = next_idx
                num_words_batch_exec = num_words_batch_next
                current_rule_batch_idx = 0  # Restart rule processing
                processed_word_batches += 1
                
            except StopIteration:
                wordlist_processing_complete = True
                print(f"{green('✅')} {bold('Wordlist fully processed.')}")
                break

    # Final update for last word batch
    words_processed_total += num_words_batch_exec
    word_batch_pbar.update(num_words_batch_exec)
    
    word_batch_pbar.close()
    rule_batch_pbar.close()
    
    # Check if we were interrupted
    if interrupted:
        print(f"\n{yellow('⚠️')} {bold('Processing was interrupted. Intermediate results have been saved.')}")
        return
        
    end_time = time()
    
    # 10. FINAL REPORTING AND SAVING
    print(f"\n{green('=' * 60)}")
    print(f"{bold('🎉 Final Results Summary')}")
    print(f"{green('=' * 60)}")
    print(f"{blue('📊')} {bold('Total Words Processed:')} {cyan(f'{words_processed_total:,}')}")
    print(f"{blue('📊')} {bold('Total Rules Processed:')} {cyan(f'{total_rules:,}')}")
    print(f"{blue('🎯')} {bold('Total Unique Words Generated:')} {cyan(f'{total_unique_found:,}')}")
    print(f"{blue('🔓')} {bold('Total True Cracks Found:')} {cyan(f'{total_cracked_found:,}')}")
    print(f"{blue('⏱️')}  {bold('Total Execution Time:')} {cyan(f'{end_time - start_time:.2f} seconds')}")
    print(f"{green('=' * 60)}{Colors.END}\n")

    csv_path = save_ranking_data(rules_list, ranking_output_path)
    if top_k > 0:
        optimized_output_path = os.path.splitext(ranking_output_path)[0] + "_optimized.rule"
        load_and_save_optimized_rules(csv_path, optimized_output_path, top_k)

# ====================================================================
# --- MAIN EXECUTION ---
# ====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPU-Accelerated Hashcat Rule Ranking Tool (Ranker v3.2 - Optimized Large File Loading)")
    parser.add_argument('-w', '--wordlist', required=True, help='Path to the base wordlist file.')
    parser.add_argument('-r', '--rules', required=True, help='Path to the Hashcat rules file to rank.')
    parser.add_argument('-c', '--cracked', required=True, help='Path to a list of cracked passwords for effectiveness scoring.')
    parser.add_argument('-o', '--output', default='ranker_output.csv', help='Path to save the final ranking CSV.')
    parser.add_argument('-k', '--topk', type=int, default=1000, help='Number of top rules to save to an optimized .rule file. Set to 0 to skip.')
    
    # Performance tuning flags
    parser.add_argument('--batch-size', type=int, default=None, 
                       help=f'Number of words to process in each GPU batch (default: auto-calculate based on VRAM)')
    parser.add_argument('--global-bits', type=int, default=None,
                       help=f'Bits for global hash map size (default: auto-calculate based on VRAM)')
    parser.add_argument('--cracked-bits', type=int, default=None,
                       help=f'Bits for cracked hash map size (default: auto-calculate based on VRAM)')
    
    # New preset argument for easy configuration
    parser.add_argument('--preset', type=str, default=None,
                       help='Use preset configuration: "low_memory", "medium_memory", "high_memory", "recommend" (auto-selects best)')
    
    args = parser.parse_args()

    print(f"{green('=' * 70)}")
    print(f"{bold('🎯 RANKER v3.2 - OPTIMIZED LARGE FILE LOADING')}")
    print(f"{green('=' * 70)}{Colors.END}")
    print(f"{blue('💡')} {bold('Features:')}")
    print(f"   {green('✓')} {bold('Memory-mapped file loading')}")
    print(f"   {green('✓')} {bold('Fast word count estimation')}")
    print(f"   {green('✓')} {bold('Bulk processing optimization')}")
    print(f"   {green('✓')} {bold('Parallel hash computation')}")
    print(f"{green('=' * 70)}{Colors.END}")

    rank_rules_uniqueness_large(
        wordlist_path=args.wordlist,
        rules_path=args.rules,
        cracked_list_path=args.cracked,
        ranking_output_path=args.output,
        top_k=args.topk,
        words_per_gpu_batch=args.batch_size,
        global_hash_map_bits=args.global_bits,
        cracked_hash_map_bits=args.cracked_bits,
        preset=args.preset
    )
