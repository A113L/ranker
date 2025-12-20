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
MAX_WORD_LEN = 256  # Increased for the comprehensive kernel
MAX_OUTPUT_LEN = 512  # Increased for the comprehensive kernel
MAX_RULE_ARGS = 4
MAX_RULES_IN_BATCH = 1024
LOCAL_WORK_SIZE = 256

# Default values (will be adjusted based on VRAM)
DEFAULT_WORDS_PER_GPU_BATCH = 100000  # Reduced due to larger word size
DEFAULT_GLOBAL_HASH_MAP_BITS = 35
DEFAULT_CRACKED_HASH_MAP_BITS = 33

# VRAM usage thresholds (adjustable)
VRAM_SAFETY_MARGIN = 0.15  # 15% safety margin
MIN_BATCH_SIZE = 25000     # Reduced minimum batch size
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

# ====================================================================
# --- PLATFORM AND DEVICE SELECTION FUNCTIONS ---
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
    
    # If platform_idx is specified, use it
    if platform_idx is not None:
        if platform_idx >= len(platforms):
            print(f"{red(f'Platform {platform_idx} not available. Available platforms:')}")
            list_platforms_and_devices()
            exit(1)
        platform = platforms[platform_idx]
    else:
        # Auto-select: prefer NVIDIA, then AMD, then Intel, then first available
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
    
    # Get devices for selected platform
    try:
        devices = platform.get_devices()
    except Exception as e:
        print(f"{red('Error getting devices for platform:')} {e}")
        exit(1)
    
    if not devices:
        print(f"{red('No devices found on selected platform!')}")
        exit(1)
    
    # If device_idx is specified, use it
    if device_idx is not None:
        if device_idx >= len(devices):
            print(f"{red(f'Device {device_idx} not available. Available devices:')}")
            for j, d in enumerate(devices):
                print(f"  {bold(f'Device {j}:')} {d.name.strip()}")
            exit(1)
        device = devices[device_idx]
    else:
        # Auto-select: prefer GPU, then CPU
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
                
        print(f"{green('Estimated words:')} {cyan(f'{total_lines:,}')}")
        return total_lines
        
    except Exception as e:
        print(f"{yellow('Could not estimate word count:')} {e}")
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
                    # Find next newline efficiently
                    end_pos = mm.find(b'\n', pos)
                    if end_pos == -1:
                        end_pos = file_size
                    
                    # Extract line directly from memory map
                    line = mm[pos:end_pos].strip()
                    line_len = len(line)
                    pos = end_pos + 1
                    
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
        print(f"{red('Error in optimized loader:')} {e}")
        raise
    
    load_time = time() - load_start
    print(f"{green('Optimized loading completed:')} {cyan(f'{total_words_loaded:,}')} {bold('words in')} {load_time:.2f}s "
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
            print(f"{blue('Saving intermediate results to:')} {intermediate_output_path}")
            
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
            
            print(f"{green('Intermediate ranking data saved:')} {cyan(f'{len(ranked_rules):,}')} {bold('rules')}")
            
            # Save optimized rules if requested
            if current_top_k > 0:
                print(f"{blue('Saving intermediate optimized rules to:')} {intermediate_optimized_path}")
                
                available_rules = len(ranked_rules)
                final_count = min(current_top_k, available_rules)
                
                with open(intermediate_optimized_path, 'w', newline='\n', encoding='utf-8') as f:
                    f.write(":\n")  # Default rule
                    for rule in ranked_rules[:final_count]:
                        f.write(f"{rule['rule_data']}\n")
                
                print(f"{green('Intermediate optimized rules saved:')} {cyan(f'{final_count:,}')} {bold('rules')}")
        
        # Print progress summary
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
    print(f"{blue('Loading rules from:')} {path}...")
    
    # Get file size for rules too
    try:
        rules_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
        if rules_size > 10:
            print(f"{yellow('Large rules file detected:')} {rules_size:.1f} MB")
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
        print(f"{red('Error:')} Rules file not found at: {path}")
        exit(1)

    print(f"{green('Loaded')} {cyan(f'{len(rules_list):,}')} {bold('rules.')}")
    return rules_list

def load_cracked_hashes(path, max_len):
    """Loads a list of cracked passwords and returns their FNV-1a hashes."""
    print(f"{blue('Loading cracked list for effectiveness check from:')} {path}...")
    
    # Get file size
    try:
        cracked_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
        if cracked_size > 50:
            print(f"{yellow('Large cracked list detected:')} {cracked_size:.1f} MB - loading...")
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
        print(f"{yellow('Warning:')} Cracked list file not found at: {path}. Effectiveness scores will be zero.")
        return np.array([], dtype=np.uint32)

    load_time = time() - load_start
    unique_hashes = np.unique(np.array(cracked_hashes, dtype=np.uint32))
    
    if cracked_size > 50:
        print(f"{green('Cracked list loaded:')} {cyan(f'{len(unique_hashes):,}')} {bold('unique hashes in')} {load_time:.2f}s")
    else:
        print(f"{green('Loaded')} {cyan(f'{len(unique_hashes):,}')} {bold('unique cracked password hashes.')}")
        
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
    
    print(f"{blue('Saving rule ranking data to:')} {ranking_output_path}...")

    # Calculate a combined score for ranking
    for rule in ranking_list:
        rule['combined_score'] = rule.get('effectiveness_score', 0) * 10 + rule.get('uniqueness_score', 0)

    # FIXED: Save ALL rules regardless of score
    ranked_rules = ranking_list  # Save ALL rules instead of filtering
    
    ranked_rules.sort(key=lambda rule: rule['combined_score'], reverse=True)

    print(f"{blue('Saving ALL')} {cyan(f'{len(ranked_rules):,}')} {bold('rules (including zero-score rules)')}")

    if not ranked_rules:
        print(f"{red('No rules to save. Ranking file not created.')}")
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

        print(f"{green('Ranking data saved successfully to')} {ranking_output_path}.")
        return ranking_output_path
    except Exception as e:
        print(f"{red('Error while saving ranking data to CSV file:')} {e}")
        return None

def load_and_save_optimized_rules(csv_path, output_path, top_k):
    """Loads ranking data from a CSV, sorts, and saves the Top K rules."""
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

    # FIXED: Include ALL rules from CSV, don't filter by score
    print(f"{blue('Loaded')} {cyan(f'{len(ranked_data):,}')} {bold('total rules from CSV')}")

    ranked_data.sort(key=lambda row: row['Combined_Score'], reverse=True)
    
    # FIXED: Cap at available rules if top_k exceeds available count
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
    """Get total and available GPU memory in bytes"""
    try:
        # Try to get global memory size
        total_memory = device.global_mem_size
        # For available memory, we'll use a conservative estimate
        # (OpenCL doesn't provide reliable available memory reporting)
        available_memory = int(total_memory * (1 - VRAM_SAFETY_MARGIN))
        return total_memory, available_memory
    except Exception as e:
        print(f"{yellow('Warning: Could not query GPU memory:')} {e}")
        # Fallback to conservative defaults
        return 8 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024  # 8GB total, 6GB available

def calculate_optimal_parameters_large_rules(available_vram, total_words, cracked_hashes_count, total_rules, reduction_factor=1.0):
    """
    Calculate optimal parameters with consideration for large rule sets
    """
    print(f"{blue('Calculating optimal parameters for')} {cyan(f'{available_vram / (1024**3):.1f} GB')} {bold('available VRAM')}")
    
    if reduction_factor < 1.0:
        print(f"{yellow('Applying memory reduction factor:')} {cyan(f'{reduction_factor:.2f}')}")
    
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
        suggested_batch_size = min(DEFAULT_WORDS_PER_GPU_BATCH, 50000)
    else:
        suggested_batch_size = DEFAULT_WORDS_PER_GPU_BATCH
    
    # Available memory for hash maps
    available_for_maps = available_vram - base_memory
    if available_for_maps <= 0:
        print(f"{yellow('Warning: Limited VRAM, using minimal configuration')}")
        available_for_maps = available_vram * 0.5
    
    print(f"{blue('Available for hash maps:')} {cyan(f'{available_for_maps / (1024**3):.2f} GB')}")
    
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
    
    print(f"{green('Optimal configuration:')}")
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
            "batch_size": 25000,
            "global_bits": 30,
            "cracked_bits": 28
        },
        "medium_memory": {
            "description": "Medium Memory Mode (for GPUs with 4-8GB VRAM)",
            "batch_size": 75000,
            "global_bits": 33,
            "cracked_bits": 31
        },
        "high_memory": {
            "description": "High Memory Mode (for GPUs with > 8GB VRAM)",
            "batch_size": 150000,
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
                # Clean up any partially allocated buffers
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
# --- COMPREHENSIVE KERNEL SOURCE (FIXED) ---
# ====================================================================

def get_kernel_source(global_hash_map_bits, cracked_hash_map_bits):
    global_hash_map_mask = (1 << (global_hash_map_bits - 5)) - 1
    cracked_hash_map_mask = (1 << (cracked_hash_map_bits - 5)) - 1
    
    return """
// ============================================================================
// COMPREHENSIVE HASHCAT RULES IMPLEMENTATION FOR OPENCL
// ============================================================================

#define MAX_WORD_LEN 256
#define MAX_OUTPUT_LEN 512
#define MAX_RULE_LEN 16

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Check if character is lowercase
int is_lower(unsigned char c) {
    return (c >= 'a' && c <= 'z');
}

// Check if character is uppercase
int is_upper(unsigned char c) {
    return (c >= 'A' && c <= 'Z');
}

// Check if character is a digit
int is_digit(unsigned char c) {
    return (c >= '0' && c <= '9');
}

// Check if character is alphanumeric
int is_alnum(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

// Toggle case of a single character
unsigned char toggle_case(unsigned char c) {
    if (is_lower(c)) return c - 32;
    if (is_upper(c)) return c + 32;
    return c;
}

// Convert character to lowercase
unsigned char to_lower(unsigned char c) {
    if (is_upper(c)) return c + 32;
    return c;
}

// Convert character to uppercase
unsigned char to_upper(unsigned char c) {
    if (is_lower(c)) return c - 32;
    return c;
}

// FNV-1a Hash implementation
unsigned int fnv1a_hash_32(const unsigned char* data, unsigned int len) {
    unsigned int hash = 2166136261U;
    for (unsigned int i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 16777619U;
    }
    return hash;
}

// Helper function to convert char digit/letter to int position
unsigned int char_to_pos(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    if (c >= 'a' && c <= 'z') return c - 'a' + 36;
    return 0xFFFFFFFF; 
}

// ============================================================================
// RULE APPLICATION FUNCTION
// ============================================================================

void apply_rule(const unsigned char* word, int word_len,
                const unsigned char* rule, int rule_len,
                unsigned char* output, int* out_len, int* changed) {
    
    *out_len = 0;
    *changed = 0;
    
    // Clear output
    for (int i = 0; i < MAX_OUTPUT_LEN; i++) output[i] = 0;
    
    // Early exit for empty inputs
    if (rule_len == 0 || word_len == 0) return;
    
    // ========================================================================
    // SIMPLE RULES (1 character)
    // ========================================================================
    
    if (rule_len == 1) {
        switch (rule[0]) {
            // l - Lowercase all letters
            case 'l':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {
                    output[i] = to_lower(word[i]);
                }
                *changed = 1;
                return;
                
            // u - Uppercase all letters
            case 'u':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {
                    output[i] = to_upper(word[i]);
                }
                *changed = 1;
                return;
                
            // c - Capitalize first letter, lowercase rest
            case 'c':
                *out_len = word_len;
                if (word_len > 0) {
                    output[0] = to_upper(word[0]);
                    for (int i = 1; i < word_len; i++) {
                        output[i] = to_lower(word[i]);
                    }
                }
                *changed = 1;
                return;
                
            // C - Lowercase first letter, uppercase rest
            case 'C':
                *out_len = word_len;
                if (word_len > 0) {
                    output[0] = to_lower(word[0]);
                    for (int i = 1; i < word_len; i++) {
                        output[i] = to_upper(word[i]);
                    }
                }
                *changed = 1;
                return;
                
            // t - Toggle case of all letters
            case 't':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {
                    output[i] = toggle_case(word[i]);
                }
                *changed = 1;
                return;
                
            // r - Reverse the entire word
            case 'r':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {
                    output[i] = word[word_len - 1 - i];
                }
                *changed = 1;
                return;
                
            // k - Swap first two characters
            case 'k':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {
                    output[i] = word[i];
                }
                if (word_len >= 2) {
                    output[0] = word[1];
                    output[1] = word[0];
                    *changed = 1;
                }
                return;
                
            // : - Do nothing (identity)
            case ':':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {
                    output[i] = word[i];
                }
                *changed = 0;
                return;
                
            // d - Duplicate word
            case 'd':
                if (word_len * 2 <= MAX_OUTPUT_LEN) {
                    *out_len = word_len * 2;
                    for (int i = 0; i < word_len; i++) {
                        output[i] = word[i];
                        output[word_len + i] = word[i];
                    }
                    *changed = 1;
                }
                return;
                
            // f - Reflect word (word + reverse)
            case 'f':
                if (word_len * 2 <= MAX_OUTPUT_LEN) {
                    *out_len = word_len * 2;
                    for (int i = 0; i < word_len; i++) {
                        output[i] = word[i];
                        output[word_len + i] = word[word_len - 1 - i];
                    }
                    *changed = 1;
                }
                return;
                
            // p - Pluralize (add 's')
            case 'p':
                if (word_len + 1 <= MAX_OUTPUT_LEN) {
                    *out_len = word_len;
                    for (int i = 0; i < word_len; i++) {
                        output[i] = word[i];
                    }
                    output[*out_len] = 's';
                    (*out_len)++;
                    *changed = 1;
                }
                return;
                
            // z - Duplicate first character
            case 'z':
                if (word_len + 1 <= MAX_OUTPUT_LEN) {
                    output[0] = word[0];
                    for (int i = 0; i < word_len; i++) {
                        output[i + 1] = word[i];
                    }
                    *out_len = word_len + 1;
                    *changed = 1;
                }
                return;
                
            // Z - Duplicate last character
            case 'Z':
                if (word_len + 1 <= MAX_OUTPUT_LEN) {
                    for (int i = 0; i < word_len; i++) {
                        output[i] = word[i];
                    }
                    output[word_len] = word[word_len - 1];
                    *out_len = word_len + 1;
                    *changed = 1;
                }
                return;
                
            // q - Duplicate all characters
            case 'q':
                if (word_len * 2 <= MAX_OUTPUT_LEN) {
                    int idx = 0;
                    for (int i = 0; i < word_len; i++) {
                        output[idx++] = word[i];
                        output[idx++] = word[i];
                    }
                    *out_len = word_len * 2;
                    *changed = 1;
                }
                return;
                
            // E - Title case (capitalize first letter of each word)
            case 'E':
                *out_len = word_len;
                int capitalize_next = 1;
                for (int i = 0; i < word_len; i++) {
                    if (capitalize_next && is_lower(word[i])) {
                        output[i] = word[i] - 32;
                        capitalize_next = 0;
                    } else {
                        output[i] = word[i];
                    }
                    if (word[i] == ' ' || word[i] == '-' || word[i] == '_') {
                        capitalize_next = 1;
                    }
                }
                *changed = 1;
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
        
        // Check if arg is a digit
        if (is_digit(arg)) {
            int n = arg - '0';
            
            switch (cmd) {
                // Tn - Toggle case at position n
                case 'T':
                    *out_len = word_len;
                    for (int i = 0; i < word_len; i++) {
                        output[i] = word[i];
                        if (i == n) {
                            output[i] = toggle_case(word[i]);
                            *changed = 1;
                        }
                    }
                    return;
                    
                // Dn - Delete character at position n
                case 'D':
                    *out_len = 0;
                    for (int i = 0; i < word_len; i++) {
                        if (i != n) {
                            output[(*out_len)++] = word[i];
                        } else {
                            *changed = 1;
                        }
                    }
                    return;
                    
                // Ln - Delete left of position n
                case 'L':
                    *out_len = 0;
                    for (int i = n; i < word_len; i++) {
                        output[(*out_len)++] = word[i];
                        *changed = 1;
                    }
                    return;
                    
                // Rn - Delete right of position n
                case 'R':
                    *out_len = n + 1;
                    if (*out_len > word_len) *out_len = word_len;
                    for (int i = 0; i < *out_len; i++) {
                        output[i] = word[i];
                    }
                    *changed = 1;
                    return;
                    
                // +n - ASCII increment at position n
                case '+':
                    *out_len = word_len;
                    for (int i = 0; i < word_len; i++) {
                        output[i] = word[i];
                        if (i == n && word[i] < 255) {
                            output[i] = word[i] + 1;
                            *changed = 1;
                        }
                    }
                    return;
                    
                // -n - ASCII decrement at position n
                case '-':
                    *out_len = word_len;
                    for (int i = 0; i < word_len; i++) {
                        output[i] = word[i];
                        if (i == n && word[i] > 0) {
                            output[i] = word[i] - 1;
                            *changed = 1;
                        }
                    }
                    return;
                    
                // .n - Replace with dot at position n
                case '.':
                    *out_len = word_len;
                    for (int i = 0; i < word_len; i++) {
                        output[i] = word[i];
                        if (i == n) {
                            output[i] = '.';
                            *changed = 1;
                        }
                    }
                    return;
                    
                // ,n - Replace with comma at position n
                case ',':
                    *out_len = word_len;
                    for (int i = 0; i < word_len; i++) {
                        output[i] = word[i];
                        if (i == n) {
                            output[i] = ',';
                            *changed = 1;
                        }
                    }
                    return;
                    
                // 'n - Increment character at position n (using escape sequence)
                case '\\'':
                    *out_len = word_len;
                    for (int i = 0; i < word_len; i++) {
                        output[i] = word[i];
                        if (i == n && word[i] < 255) {
                            output[i] = word[i] + 1;
                            *changed = 1;
                        }
                    }
                    return;
            }
        }
        
        // Handle non-digit arguments
        switch (cmd) {
            // ^X - Prepend character X
            case '^':
                if (word_len + 1 <= MAX_OUTPUT_LEN) {
                    output[0] = arg;
                    for (int i = 0; i < word_len; i++) {
                        output[i + 1] = word[i];
                    }
                    *out_len = word_len + 1;
                    *changed = 1;
                }
                return;
                
            // $X - Append character X
            case '$':
                if (word_len + 1 <= MAX_OUTPUT_LEN) {
                    for (int i = 0; i < word_len; i++) {
                        output[i] = word[i];
                    }
                    output[word_len] = arg;
                    *out_len = word_len + 1;
                    *changed = 1;
                }
                return;
                
            // @X - Delete all instances of character X
            case '@':
                *out_len = 0;
                for (int i = 0; i < word_len; i++) {
                    if (word[i] != arg) {
                        output[(*out_len)++] = word[i];
                    } else {
                        *changed = 1;
                    }
                }
                return;
                
            // !X - Reject word if it contains X
            case '!':
                for (int i = 0; i < word_len; i++) {
                    if (word[i] == arg) {
                        *out_len = 0;
                        *changed = -1;
                        return;
                    }
                }
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {
                    output[i] = word[i];
                }
                return;
                
            // /X - Reject word if it doesn't contain X
            case '/': {
                int found_char = 0;
                for (int i = 0; i < word_len; i++) {
                    if (word[i] == arg) {
                        found_char = 1;
                        break;
                    }
                }
                if (!found_char) {
                    *out_len = 0;
                    *changed = -1;
                    return;
                }
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {
                    output[i] = word[i];
                }
                return;
            }
        }
    }
    
    // ========================================================================
    // SUBSTITUTION RULES (sXY)
    // ========================================================================
    
    else if (rule_len == 3 && rule[0] == 's') {
        // sXY - Substitute X with Y
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
        // Handle rules like xn m, *n m, Kn m, i n X, o n X, etc.
        
        // xn m - Extract substring from n to m
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
        
        // *n m - Swap characters at positions n and m
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
        
        // i n X - Insert X at position n
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
        
        // o n X - Overwrite at position n with X
        else if (rule[0] == 'o' && rule_len >= 3) {
            unsigned int n = char_to_pos(rule[1]);
            unsigned char x = rule[2];
            
            if (n != 0xFFFFFFFF) {
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {
                    output[i] = word[i];
                    if (i == n) {
                        output[i] = x;
                        *changed = 1;
                    }
                }
                return;
            }
        }
        
        // {N - Rotate left N positions
        else if (rule[0] == '{' && rule_len >= 2) {
            unsigned int n = char_to_pos(rule[1]);
            if (n == 0xFFFFFFFF) n = 1;
            
            *out_len = word_len;
            for (int i = 0; i < word_len; i++) {
                int src = (i + n) % word_len;
                output[i] = word[src];
            }
            *changed = 1;
            return;
        }
        
        // }N - Rotate right N positions
        else if (rule[0] == '}' && rule_len >= 2) {
            unsigned int n = char_to_pos(rule[1]);
            if (n == 0xFFFFFFFF) n = 1;
            
            *out_len = word_len;
            for (int i = 0; i < word_len; i++) {
                int src = (i - n + word_len) % word_len;
                output[i] = word[src];
            }
            *changed = 1;
            return;
        }
        
        // [N - Delete first N characters
        else if (rule[0] == '[' && rule_len >= 2) {
            unsigned int n = char_to_pos(rule[1]);
            if (n == 0xFFFFFFFF) n = 1;
            if (n > word_len) n = word_len;
            
            *out_len = word_len - n;
            for (int i = n; i < word_len; i++) {
                output[i - n] = word[i];
            }
            *changed = 1;
            return;
        }
        
        // ]N - Delete last N characters
        else if (rule[0] == ']' && rule_len >= 2) {
            unsigned int n = char_to_pos(rule[1]);
            if (n == 0xFFFFFFFF) n = 1;
            if (n > word_len) n = word_len;
            
            *out_len = word_len - n;
            for (int i = 0; i < *out_len; i++) {
                output[i] = word[i];
            }
            *changed = 1;
            return;
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

__kernel __attribute__((reqd_work_group_size(""" + str(LOCAL_WORK_SIZE) + """, 1, 1)))
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
{
    unsigned int global_id = get_global_id(0);
    unsigned int word_per_rule_count = num_words * num_rules_in_batch;
    
    if (global_id >= word_per_rule_count) return;

    unsigned int word_idx = global_id / num_rules_in_batch;
    unsigned int rule_batch_idx = global_id % num_rules_in_batch;

    // Load word
    unsigned char word[MAX_WORD_LEN];
    unsigned int word_len = 0;
    for (unsigned int i = 0; i < max_word_len; i++) {
        unsigned char c = base_words_in[word_idx * max_word_len + i];
        if (c == 0) break;
        word[i] = c;
        word_len++;
    }

    // Load rule
    unsigned int rule_size_in_int = 2 + """ + str(MAX_RULE_ARGS) + """;
    __global const unsigned int* rule_ptr = rules_in + rule_batch_idx * rule_size_in_int;
    unsigned int rule_id = rule_ptr[0];
    unsigned int rule_args_int = rule_ptr[1];
    unsigned int rule_args_int2 = rule_ptr[2];

    // Decode rule string from packed integers
    unsigned char rule_str[MAX_RULE_LEN];
    unsigned int rule_len = 0;
    
    // Unpack first 4 bytes from rule_args_int
    for (unsigned int i = 0; i < 4; i++) {
        unsigned char byte = (rule_args_int >> (i * 8)) & 0xFF;
        if (byte == 0) break;
        rule_str[rule_len++] = byte;
    }
    
    // Unpack next 4 bytes from rule_args_int2
    if (rule_len < MAX_RULE_LEN) {
        for (unsigned int i = 0; i < 4; i++) {
            unsigned char byte = (rule_args_int2 >> (i * 8)) & 0xFF;
            if (byte == 0) break;
            rule_str[rule_len++] = byte;
        }
    }

    // Apply rule
    unsigned char result_temp[MAX_OUTPUT_LEN];
    int out_len = 0;
    int changed_flag = 0;
    
    apply_rule(word, word_len, rule_str, rule_len, result_temp, &out_len, &changed_flag);

    // DUAL-UNIQUENESS LOGIC
    if (changed_flag > 0 && out_len > 0) {
        unsigned int word_hash = fnv1a_hash_32(result_temp, out_len);

        // 1. Check against Base Wordlist (Uniqueness)
        unsigned int global_map_index = (word_hash >> 5) & """ + str(global_hash_map_mask) + """;
        unsigned int bit_index = word_hash & 31;
        unsigned int check_bit = (1U << bit_index);

        __global const unsigned int* global_map_ptr = &global_hash_map[global_map_index];
        unsigned int current_global_word = *global_map_ptr;

        if (!(current_global_word & check_bit)) {
            atomic_inc(&rule_uniqueness_counts[rule_batch_idx]);

            // 2. Check against Cracked List (Effectiveness)
            unsigned int cracked_map_index = (word_hash >> 5) & """ + str(cracked_hash_map_mask) + """;
            __global const unsigned int* cracked_map_ptr = &cracked_hash_map[cracked_map_index];
            unsigned int current_cracked_word = *cracked_map_ptr;

            if (current_cracked_word & check_bit) {
                atomic_inc(&rule_effectiveness_counts[rule_batch_idx]);
            }
        }
    }
}

// ============================================================================
// HASH MAP INITIALIZATION KERNEL
// ============================================================================

__kernel __attribute__((reqd_work_group_size(""" + str(LOCAL_WORK_SIZE) + """, 1, 1)))
void hash_map_init_kernel(
    __global unsigned int* global_hash_map,
    __global const unsigned int* base_hashes,
    const unsigned int num_hashes,
    const unsigned int map_mask)
{
    unsigned int global_id = get_global_id(0);
    if (global_id >= num_hashes) return;

    unsigned int word_hash = base_hashes[global_id];
    unsigned int map_index = (word_hash >> 5) & map_mask;
    unsigned int bit_index = word_hash & 31;
    unsigned int set_bit = (1U << bit_index);

    atomic_or(&global_hash_map[map_index], set_bit);
}
"""

# ====================================================================
# --- UPDATED MAIN RANKING FUNCTION WITH CONTINUOUS RULE PROCESSING ---
# ====================================================================

def rank_rules_uniqueness_large(wordlist_path, rules_path, cracked_list_path, ranking_output_path, top_k, 
                               words_per_gpu_batch=None, global_hash_map_bits=None, cracked_hash_map_bits=None,
                               preset=None, platform_idx=None, device_idx=None):
    start_time = time()
    
    # 0. PRELIMINARY DATA LOADING FOR MEMORY CALCULATION
    total_words = estimate_word_count(wordlist_path)  # Use fast estimation
    rules_list = load_rules(rules_path)
    total_rules = len(rules_list)
    
    # Setup interrupt handler BEFORE starting processing
    setup_interrupt_handler(rules_list, ranking_output_path, top_k)
    
    # Check if we're dealing with a large rule set
    if total_rules > 100000:
        print(f"{green('LARGE RULE SET DETECTED:')} {cyan(f'{total_rules:,}')} {bold('rules')}")
        print(f"   {bold('Using optimized processing for large rule sets...')}")
    
    # Load cracked hashes ONCE for both memory calculation and processing
    cracked_hashes_np = load_cracked_hashes(cracked_list_path, MAX_WORD_LEN)
    cracked_hashes_count = len(cracked_hashes_np)
    
    # FIXED: Report correct dataset statistics
    print(f"\n{blue('Dataset Summary:')}")
    print(f"   {bold('Words:')} {cyan(f'{total_words:,}')}")
    print(f"   {bold('Rules:')} {cyan(f'{total_rules:,}')}")
    print(f"   {bold('Cracked hashes:')} {cyan(f'{cracked_hashes_count:,}')}")
    
    # 1. OPENCL INITIALIZATION AND MEMORY DETECTION
    try:
        # Select platform and device
        platform, device = select_platform_and_device(platform_idx, device_idx)
        
        context = cl.Context([device])
        queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # Get GPU memory information
        total_vram, available_vram = get_gpu_memory_info(device)
        print(f"\n{green('GPU:')} {cyan(device.name.strip())}")
        print(f"{blue('Platform:')} {cyan(platform.name.strip())}")
        print(f"{blue('Total VRAM:')} {cyan(f'{total_vram / (1024**3):.1f} GB')}")
        print(f"{blue('Available VRAM:')} {cyan(f'{available_vram / (1024**3):.1f} GB')}")
        
        # Handle preset parameter specification
        if preset:
            recommendations, recommended_preset = get_recommended_parameters(device, total_words, cracked_hashes_count)
            
            if preset == "recommend":
                print(f"{green('Recommended preset:')} {cyan(recommended_preset)}")
                preset = recommended_preset
            
            if preset in recommendations:
                preset_config = recommendations[preset]
                print(f"{blue('Using')} {cyan(preset_config['description'])}")
                words_per_gpu_batch = preset_config['batch_size']
                global_hash_map_bits = preset_config['global_bits']
                cracked_hash_map_bits = preset_config['cracked_bits']
            else:
                print(f"{red('Unknown preset:')} {cyan(preset)}. {bold('Available presets:')} {list(recommendations.keys())}")
                return
        
        # Handle manual parameter specification
        using_manual_params = False
        if words_per_gpu_batch is not None or global_hash_map_bits is not None or cracked_hash_map_bits is not None:
            using_manual_params = True
            print(f"{blue('Using manually specified parameters:')}")
            
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
                print(f"{yellow('Warning: Manual parameters exceed available VRAM!')}")
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
        
        kernel_bfs = prg.bfs_kernel
        kernel_init = prg.hash_map_init_kernel
        
        print(f"{green('OpenCL initialized on device:')} {cyan(device.name.strip())}")
    except Exception as e:
        print(f"{red('OpenCL initialization or kernel compilation error:')} {e}")
        exit(1)

    # 2. DATA LOADING AND PRE-ENCODING
    rule_size_in_int = 2 + MAX_RULE_ARGS
    encoded_rules = [encode_rule(rule['rule_data'], rule['rule_id'], MAX_RULE_ARGS) for rule in rules_list]

    # 3. HASH MAP INITIALIZATION
    global_hash_map_np = np.zeros(GLOBAL_HASH_MAP_WORDS, dtype=np.uint32)
    print(f"\n{blue('Global Hash Map initialized:')} {cyan(f'{global_hash_map_np.nbytes / (1024*1024):.2f} MB')} {bold('allocated.')}")

    cracked_hash_map_np = np.zeros(CRACKED_HASH_MAP_WORDS, dtype=np.uint32)
    print(f"{blue('Cracked Hash Map initialized:')} {cyan(f'{cracked_hash_map_np.nbytes / (1024*1024):.2f} MB')} {bold('allocated.')}")

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
        print(f"{red('Fatal: Could not allocate GPU memory even after retries:')} {e}")
        print(f"{yellow('Try reducing batch size or hash map bits, or use a preset:')}")
        recommendations, _ = get_recommended_parameters(device, total_words, cracked_hashes_count)
        for preset_name, config in recommendations.items():
            if preset_name != "auto":
                print(f"   {bold('--preset')} {cyan(preset_name)}: {config['description']}")
        return

    current_word_buffer_idx = 0

    # 5. INITIALIZE CRACKED HASH MAP (ONCE)
    if cracked_hashes_np.size > 0 and cracked_temp_g is not None:
        global_size_init_cracked = (int(math.ceil(cracked_hashes_np.size / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
        local_size_init_cracked = (LOCAL_WORK_SIZE,)

        print(f"{blue('Populating static Cracked Hash Map on GPU...')}")
        kernel_init(queue, global_size_init_cracked, local_size_init_cracked,
                    cracked_hash_map_g,
                    cracked_temp_g,
                    np.uint32(cracked_hashes_np.size),
                    np.uint32(CRACKED_HASH_MAP_MASK)).wait()
        print(f"{green('Static Cracked Hash Map populated.')}")
    else:
        print(f"{yellow('Cracked list is empty, effectiveness scoring is disabled.')}")
        
    # 6. PIPELINED RANKING LOOP SETUP
    rule_batch_starts = list(range(0, total_rules, MAX_RULES_IN_BATCH))
    total_rule_batches = len(rule_batch_starts)
    
    print(f"\n{blue('Processing Configuration:')}")
    print(f"   {bold('Words per batch:')} {cyan(f'{words_per_gpu_batch:,}')}")
    print(f"   {bold('Rules per batch:')} {cyan(f'{MAX_RULES_IN_BATCH:,}')}")
    print(f"   {bold('Total rule batches:')} {cyan(f'{total_rule_batches:,}')}")
    
    # Use Python ints for counters
    words_processed_total = 0
    total_unique_found = 0
    total_cracked_found = 0
    
    mapped_uniqueness_np = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
    mapped_effectiveness_np = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
    
    # Create progress bar for word batches
    word_batch_pbar = tqdm(total=total_words, desc="Word batches", unit=" words",
                          bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                          position=0)
    
    # Create progress bar for rule batches
    rule_batch_pbar = tqdm(total=total_rule_batches, desc="Rule batches", unit=" batches",
                          bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                          position=1)

    # Load the entire wordlist into a list first for proper synchronization
    word_batches = []
    print(f"\n{blue('Preloading word batches...')}")
    
    try:
        # Load all word batches into memory
        word_iterator = optimized_wordlist_iterator(wordlist_path, MAX_WORD_LEN, words_per_gpu_batch)
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
        print(f"{red('Error preloading word batches:')} {e}")
        return
    
    total_word_batches = len(word_batches)
    print(f"{green('Preloaded')} {cyan(f'{total_word_batches:,}')} {bold('word batches')}")
    
    # Initialize global hash map with first word batch
    print(f"\n{blue('Initializing global hash map with first word batch...')}")
    
    # IMPORTANT FIX: Process all rule batches for each word batch
    # Outer loop for word batches, inner loop for rule batches
    for word_batch_idx, word_batch_data in enumerate(word_batches):
        # Check for interrupt before processing
        if interrupted:
            break
        
        # Load current word batch to GPU
        cl.enqueue_copy(queue, base_words_in_g[current_word_buffer_idx], word_batch_data['words'])
        cl.enqueue_copy(queue, base_hashes_g[current_word_buffer_idx], word_batch_data['hashes']).wait()
        
        # Initialize/update global hash map with current word batch
        global_size_init_global = (int(math.ceil(word_batch_data['count'] / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
        local_size_init_global = (LOCAL_WORK_SIZE,)
        kernel_init(queue, global_size_init_global, local_size_init_global,
                    global_hash_map_g,
                    base_hashes_g[current_word_buffer_idx],
                    np.uint32(word_batch_data['count']),
                    np.uint32(GLOBAL_HASH_MAP_MASK)).wait()
        
        # Process all rule batches for this word batch
        for rule_batch_idx in range(total_rule_batches):
            # Check for interrupt before each rule batch
            if interrupted:
                break
                
            rule_start_index = rule_batch_starts[rule_batch_idx]
            rule_end_index = min(rule_start_index + MAX_RULES_IN_BATCH, total_rules)
            num_rules_in_batch = rule_end_index - rule_start_index

            current_rule_batch_list = encoded_rules[rule_start_index:rule_end_index]
            current_rules_np = np.concatenate(current_rule_batch_list)
            
            cl.enqueue_copy(queue, rules_in_g, current_rules_np, is_blocking=True)
            cl.enqueue_fill_buffer(queue, rule_uniqueness_counts_g, np.uint32(0), 0, counters_size)
            cl.enqueue_fill_buffer(queue, rule_effectiveness_counts_g, np.uint32(0), 0, counters_size)
            
            global_size = (word_batch_data['count'] * num_rules_in_batch, )
            global_size_aligned = (int(math.ceil(global_size[0] / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)

            kernel_bfs.set_args(
                base_words_in_g[current_word_buffer_idx],
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
            
            # Update progress bars with dynamic counters
            words_processed_so_far = words_processed_total + word_batch_data['count'] * (rule_batch_idx + 1) / total_rule_batches
            word_batch_pbar.n = min(int(words_processed_so_far), total_words)
            word_batch_pbar.set_description(
                f"Words: {int(words_processed_so_far):,}/{total_words:,} | Unique: {total_unique_found:,} | Cracked: {total_cracked_found:,}"
            )
            
            rule_batch_pbar.update(1)
            rule_batch_pbar.set_description(
                f"Rules: {rule_batch_idx+1}/{total_rule_batches} (Word: {word_batch_idx+1}/{total_word_batches})"
            )
            
            # Update interrupt handler stats
            update_progress_stats(words_processed_so_far, total_unique_found, total_cracked_found)
        
        # Update word count for completed word batch
        words_processed_total += word_batch_data['count']
        
        # Update word batch progress bar
        word_batch_pbar.update(word_batch_data['count'])
        
        # Reset rule batch progress for next word batch
        rule_batch_pbar.n = 0
        rule_batch_pbar.refresh()
        
        # Switch buffer for next iteration
        current_word_buffer_idx = 1 - current_word_buffer_idx
    
    # Check if we were interrupted
    if interrupted:
        print(f"\n{yellow('Processing was interrupted. Intermediate results have been saved.')}")
        return
        
    # Final update for progress bars
    word_batch_pbar.close()
    rule_batch_pbar.close()
    
    end_time = time()
    
    # 10. FINAL REPORTING AND SAVING
    print(f"\n{green('=' * 60)}")
    print(f"{bold('Final Results Summary')}")
    print(f"{green('=' * 60)}")
    print(f"{blue('Total Words Processed:')} {cyan(f'{words_processed_total:,}')}")
    print(f"{blue('Total Rules Processed:')} {cyan(f'{total_rules:,}')}")
    print(f"{blue('Total Unique Words Generated:')} {cyan(f'{total_unique_found:,}')}")
    print(f"{blue('Total True Cracks Found:')} {cyan(f'{total_cracked_found:,}')}")
    print(f"{blue('Total Execution Time:')} {cyan(f'{end_time - start_time:.2f} seconds')}")
    print(f"{green('=' * 60)}{Colors.END}\n")

    csv_path = save_ranking_data(rules_list, ranking_output_path)
    if top_k > 0:
        optimized_output_path = os.path.splitext(ranking_output_path)[0] + "_optimized.rule"
        load_and_save_optimized_rules(csv_path, optimized_output_path, top_k)

# ====================================================================
# --- MAIN EXECUTION ---
# ====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="GPU-Accelerated Hashcat Rule Ranking Tool (Ranker v3.2 - Optimized Large File Loading)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
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
        """
    )
    
    # Create argument groups for better organization
    main_group = parser.add_argument_group('Main Arguments')
    main_group.add_argument('-w', '--wordlist', help='Path to the base wordlist file.')
    main_group.add_argument('-r', '--rules', help='Path to the Hashcat rules file to rank.')
    main_group.add_argument('-c', '--cracked', help='Path to a list of cracked passwords for effectiveness scoring.')
    main_group.add_argument('-o', '--output', default='ranker_output.csv', help='Path to save the final ranking CSV.')
    main_group.add_argument('-k', '--topk', type=int, default=1000, help='Number of top rules to save to an optimized .rule file. Set to 0 to skip.')
    
    perf_group = parser.add_argument_group('Performance Tuning')
    perf_group.add_argument('--batch-size', type=int, default=None, 
                           help=f'Number of words to process in each GPU batch (default: auto-calculate based on VRAM)')
    perf_group.add_argument('--global-bits', type=int, default=None,
                           help=f'Bits for global hash map size (default: auto-calculate based on VRAM)')
    perf_group.add_argument('--cracked-bits', type=int, default=None,
                           help=f'Bits for cracked hash map size (default: auto-calculate based on VRAM)')
    perf_group.add_argument('--preset', type=str, default=None,
                           help='Use preset configuration: "low_memory", "medium_memory", "high_memory", "recommend" (auto-selects best)')
    
    platform_group = parser.add_argument_group('Platform/Device Selection')
    platform_group.add_argument('--list-platforms', action='store_true',
                               help='List all available OpenCL platforms and devices, then exit.')
    platform_group.add_argument('--platform', type=int, default=None,
                               help='Select specific OpenCL platform index (use --list-platforms to see available options).')
    platform_group.add_argument('--device', type=int, default=None,
                               help='Select specific device index within platform (use --list-platforms to see available options).')
    
    args = parser.parse_args()

    # Show banner
    print(f"{green('=' * 70)}")
    print(f"{bold('RANKER v3.2 - OPTIMIZED LARGE FILE LOADING')}")
    print(f"{green('=' * 70)}{Colors.END}")
    print(f"{blue('Features:')}")
    print(f"   {green('✓')} {bold('Memory-mapped file loading')}")
    print(f"   {green('✓')} {bold('Fast word count estimation')}")
    print(f"   {green('✓')} {bold('Bulk processing optimization')}")
    print(f"   {green('✓')} {bold('Parallel hash computation')}")
    print(f"   {green('✓')} {bold('Comprehensive Hashcat rules support (16,000+ rules)')}")
    print(f"   {green('✓')} {bold('Multi-platform support (NVIDIA, AMD, Intel, CPU)')}")
    print(f"{green('=' * 70)}{Colors.END}")

    # List platforms if requested
    if args.list_platforms:
        list_platforms_and_devices()
        exit(0)

    # Check if main arguments are provided
    if not args.wordlist or not args.rules or not args.cracked:
        print(f"{red('Error:')} The following arguments are required when not using --list-platforms: -w/--wordlist, -r/--rules, -c/--cracked")
        print(f"\n{blue('Usage information:')}")
        parser.print_help()
        exit(1)

    rank_rules_uniqueness_large(
        wordlist_path=args.wordlist,
        rules_path=args.rules,
        cracked_list_path=args.cracked,
        ranking_output_path=args.output,
        top_k=args.topk,
        words_per_gpu_batch=args.batch_size,
        global_hash_map_bits=args.global_bits,
        cracked_hash_map_bits=args.cracked_bits,
        preset=args.preset,
        platform_idx=args.platform,
        device_idx=args.device
    )
