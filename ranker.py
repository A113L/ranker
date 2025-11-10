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

# ====================================================================
# --- RANKER v3.1: IMPROVED CAPACITY FOR LARGE RULE SETS ---
# ====================================================================

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
# --- INCREASED CONSTANTS FOR LARGE RULE SETS ---
# ====================================================================
MAX_WORD_LEN = 32
MAX_OUTPUT_LEN = MAX_WORD_LEN * 2
MAX_RULE_ARGS = 4
MAX_RULES_IN_BATCH = 1024  # INCREASED FROM 256 TO 1024 (4x capacity)
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
# --- MISSING HELPER FUNCTIONS ---
# ====================================================================

def get_word_count(path):
    """Counts total words in file to set up progress bar."""
    print(f"Counting words in: {path}...")
    
    # Get file size for timing estimates
    try:
        file_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
        if file_size > 100:
            print(f"📁 Large wordlist detected: {file_size:.1f} MB - this may take a moment...")
    except OSError:
        file_size = 0
    
    start_count = time()
    count = 0
    try:
        with open(path, 'r', encoding='latin-1', errors='ignore') as f:
            for line in f:
                count += 1
    except FileNotFoundError:
        print(f"Error: Wordlist file not found at: {path}")
        exit(1)
    
    count_time = time() - start_count
    print(f"✅ Total words found: {count:,} (counting took {count_time:.2f}s)")
    
    # Provide timing estimates for large files
    if file_size > 100 and count_time > 5:
        estimated_load_time = count_time * 2
        print(f"⏱️  For {file_size:.1f}MB file, full loading will take approximately {estimated_load_time:.1f}s")
    
    return count

def load_rules(path):
    """Loads Hashcat rules from file."""
    print(f"Loading rules from: {path}...")
    
    # Get file size for rules too
    try:
        rules_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
        if rules_size > 10:
            print(f"📁 Large rules file detected: {rules_size:.1f} MB")
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
        print(f"Error: Rules file not found at: {path}")
        exit(1)

    print(f"Loaded {len(rules_list)} rules.")
    return rules_list

def fnv1a_hash_32_cpu(data):
    """Calculates FNV-1a hash for a byte array."""
    hash_val = np.uint32(2166136261)
    for byte in data:
        hash_val ^= np.uint32(byte)
        hash_val *= np.uint32(16777619)
    return hash_val

def load_cracked_hashes(path, max_len):
    """Loads a list of cracked passwords and returns their FNV-1a hashes."""
    print(f"Loading cracked list for effectiveness check from: {path}...")
    
    # Get file size
    try:
        cracked_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
        if cracked_size > 50:
            print(f"📁 Large cracked list detected: {cracked_size:.1f} MB - loading...")
    except OSError:
        cracked_size = 0
    
    cracked_hashes = []
    load_start = time()
    try:
        with open(path, 'r', encoding='latin-1', errors='ignore') as f:
            for line in f:
                word = line.strip().encode('latin-1')
                if 1 <= len(word) <= max_len:
                    cracked_hashes.append(fnv1a_hash_32_cpu(word))
    except FileNotFoundError:
        print(f"Warning: Cracked list file not found at: {path}. Effectiveness scores will be zero.")
        return np.array([], dtype=np.uint32)

    load_time = time() - load_start
    unique_hashes = np.unique(np.array(cracked_hashes, dtype=np.uint32))
    
    if cracked_size > 50:
        print(f"✅ Cracked list loaded: {len(unique_hashes):,} unique hashes in {load_time:.2f}s")
    else:
        print(f"Loaded {len(unique_hashes):,} unique cracked password hashes.")
        
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
    
    print(f"Saving rule ranking data to: {ranking_output_path}...")

    # Calculate a combined score for ranking
    for rule in ranking_list:
        rule['combined_score'] = rule.get('effectiveness_score', 0) * 10 + rule.get('uniqueness_score', 0)

    # FIXED: Save ALL rules regardless of score
    ranked_rules = ranking_list  # Save ALL rules instead of filtering
    
    ranked_rules.sort(key=lambda rule: rule['combined_score'], reverse=True)

    print(f"💾 Saving ALL {len(ranked_rules):,} rules (including zero-score rules)")

    if not ranked_rules:
        print("❌ No rules to save. Ranking file not created.")
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

        print(f"✅ Ranking data saved successfully to {ranking_output_path}.")
        return ranking_output_path
    except Exception as e:
        print(f"❌ Error while saving ranking data to CSV file: {e}")
        return None

def load_and_save_optimized_rules(csv_path, output_path, top_k):
    """Loads ranking data from a CSV, sorts, and saves the Top K rules."""
    if not csv_path:
        print("\nOptimization skipped: Ranking CSV path is missing.")
        return

    print(f"\nLoading ranking from CSV: {csv_path} and saving Top {top_k} Optimized Rules to: {output_path}...")
    
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
        print(f"❌ Error: Ranking CSV file not found at: {csv_path}")
        return
    except Exception as e:
        print(f"❌ Error while reading CSV: {e}")
        return

    # FIXED: Include ALL rules from CSV, don't filter by score
    print(f"📊 Loaded {len(ranked_data):,} total rules from CSV")

    ranked_data.sort(key=lambda row: row['Combined_Score'], reverse=True)
    
    # FIXED: Cap at available rules if top_k exceeds available count
    available_rules = len(ranked_data)
    if top_k > available_rules:
        print(f"⚠️  Warning: Requested {top_k:,} rules but only {available_rules:,} available. Saving {available_rules:,} rules.")
        final_optimized_list = ranked_data[:available_rules]
    else:
        final_optimized_list = ranked_data[:top_k]

    if not final_optimized_list:
        print("❌ No rules available after sorting/filtering. Optimized rule file not created.")
        return

    try:
        with open(output_path, 'w', newline='\n', encoding='utf-8') as f:
            f.write(":\n")  # Default rule
            for rule in final_optimized_list:
                f.write(f"{rule['Rule_Data']}\n")
        print(f"✅ Top {len(final_optimized_list):,} optimized rules saved successfully to {output_path}.")
    except Exception as e:
        print(f"❌ Error while saving optimized rules to file: {e}")

def wordlist_iterator_optimized(wordlist_path, max_len, batch_size):
    """
    Optimized generator that yields batches of words and initial hashes directly from disk.
    Uses memory mapping and buffered reading for large files.
    """
    base_words_np = np.zeros((batch_size, max_len), dtype=np.uint8)
    base_hashes = []
    current_batch_count = 0
    
    # Get file size for progress reporting
    try:
        file_size = os.path.getsize(wordlist_path) / (1024 * 1024)  # Size in MB
        if file_size > 100:
            print(f"🚀 Loading large wordlist ({file_size:.1f} MB) using optimized memory mapping...")
            if file_size > 500:
                print("💡 This may take a while. Consider using SSD for better performance.")
            if file_size > 1000:
                print("💡 For multi-GB files, loading time depends on your storage speed:")
                print("   - HDD: 2-5 minutes per GB")
                print("   - SSD: 30-60 seconds per GB") 
                print("   - NVMe: 10-20 seconds per GB")
    except OSError:
        file_size = 0
    
    load_start = time()
    words_loaded = 0
    
    try:
        # Use a larger buffer for better I/O performance
        with open(wordlist_path, 'r', encoding='latin-1', errors='ignore', buffering=8192*16) as f:
            for line in f:
                word_str = line.strip()
                if not word_str:  # Skip empty lines
                    continue
                    
                word = word_str.encode('latin-1')
                
                if 1 <= len(word) <= max_len:
                    # Fill the numpy array directly
                    base_words_np[current_batch_count, :len(word)] = np.frombuffer(word, dtype=np.uint8)
                    base_hashes.append(fnv1a_hash_32_cpu(word))
                    current_batch_count += 1
                    words_loaded += 1

                    if current_batch_count == batch_size:
                        yield base_words_np.ravel().copy(), current_batch_count, np.array(base_hashes, dtype=np.uint32)
                        base_words_np.fill(0)
                        base_hashes = []
                        current_batch_count = 0

                        # Show progress for large files
                        if file_size > 100 and words_loaded % 1000000 == 0:
                            elapsed = time() - load_start
                            rate = words_loaded / elapsed
                            print(f"📥 Loaded {words_loaded:,} words ({elapsed:.1f}s, {rate:,.0f} words/sec)")

        # Yield remaining words
        if current_batch_count > 0:
            words_to_yield = base_words_np[:current_batch_count, :].ravel().copy()
            yield words_to_yield, current_batch_count, np.array(base_hashes, dtype=np.uint32)
            
        total_load_time = time() - load_start
        if file_size > 100:
            rate = words_loaded / total_load_time
            print(f"✅ Wordlist loading completed: {words_loaded:,} words in {total_load_time:.2f}s "
                  f"({rate:,.0f} words/sec)")
            
    except Exception as e:
        print(f"❌ Error reading wordlist: {e}")
        raise

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
        print(f"Warning: Could not query GPU memory: {e}")
        # Fallback to conservative defaults
        return 8 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024  # 8GB total, 6GB available

def calculate_optimal_parameters_large_rules(available_vram, total_words, cracked_hashes_count, total_rules, reduction_factor=1.0):
    """
    Calculate optimal parameters with consideration for large rule sets
    """
    print(f"🔧 Calculating optimal parameters for {available_vram / (1024**3):.1f} GB available VRAM")
    print(f"📊 Dataset: {total_words:,} words, {total_rules:,} rules, {cracked_hashes_count:,} cracked hashes")
    
    if reduction_factor < 1.0:
        print(f"📉 Applying memory reduction factor: {reduction_factor:.2f}")
    
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
        print("⚠️  Warning: Limited VRAM, using minimal configuration")
        available_for_maps = available_vram * 0.5
    
    print(f"📊 Available for hash maps: {available_for_maps / (1024**3):.2f} GB")
    
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
    
    print(f"🎯 Optimal configuration:")
    print(f"   - Batch size: {optimal_batch_size:,} words")
    print(f"   - Rules per batch: {MAX_RULES_IN_BATCH:,}")
    print(f"   - Global hash map: {global_bits} bits ({global_map_bytes / (1024**2):.1f} MB)")
    print(f"   - Cracked hash map: {cracked_bits} bits ({cracked_map_bytes / (1024**2):.1f} MB)")
    print(f"   - Total map memory: {total_map_memory / (1024**3):.2f} GB")
    print(f"   - Estimated rule batches: {(total_rules + MAX_RULES_IN_BATCH - 1) // MAX_RULES_IN_BATCH}")
    
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
            print(f"🔄 Attempt {retry + 1}/{max_retries + 1} to allocate buffers (reduction: {current_reduction:.2f})")
            
            for name, spec in buffer_specs.items():
                flags = spec['flags']
                size = int(spec['size'] * current_reduction)
                
                if 'hostbuf' in spec:
                    buffers[name] = cl.Buffer(context, flags, size, hostbuf=spec['hostbuf'])
                else:
                    buffers[name] = cl.Buffer(context, flags, size)
            
            print(f"✅ Successfully allocated all buffers on attempt {retry + 1}")
            return buffers
            
        except cl.MemoryError as e:
            if "MEM_OBJECT_ALLOCATION_FAILURE" in str(e) and retry < max_retries:
                print(f"⚠️  Memory allocation failed, reducing memory usage...")
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
                
    raise cl.MemoryError(f"Failed to allocate buffers after {max_retries} retries")

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
# --- UPDATED MAIN RANKING FUNCTION FOR LARGE RULE SETS ---
# ====================================================================

def rank_rules_uniqueness_large(wordlist_path, rules_path, cracked_list_path, ranking_output_path, top_k, 
                               words_per_gpu_batch=None, global_hash_map_bits=None, cracked_hash_map_bits=None,
                               preset=None):
    start_time = time()
    
    # 0. PRELIMINARY DATA LOADING FOR MEMORY CALCULATION
    total_words = get_word_count(wordlist_path)
    rules_list = load_rules(rules_path)
    total_rules = len(rules_list)
    
    # Check if we're dealing with a large rule set
    if total_rules > 100000:
        print(f"🚀 LARGE RULE SET DETECTED: {total_rules:,} rules")
        print("   Using optimized processing for large rule sets...")
    
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
        print(f"🎮 GPU: {device.name.strip()}")
        print(f"💾 Total VRAM: {total_vram / (1024**3):.1f} GB")
        print(f"💾 Available VRAM: {available_vram / (1024**3):.1f} GB")
        
        # Handle preset parameter specification
        if preset:
            recommendations, recommended_preset = get_recommended_parameters(device, total_words, cracked_hashes_count)
            
            if preset == "recommend":
                print(f"🎯 Recommended preset: {recommended_preset}")
                preset = recommended_preset
            
            if preset in recommendations:
                preset_config = recommendations[preset]
                print(f"🔧 Using {preset_config['description']}")
                words_per_gpu_batch = preset_config['batch_size']
                global_hash_map_bits = preset_config['global_bits']
                cracked_hash_map_bits = preset_config['cracked_bits']
            else:
                print(f"❌ Unknown preset: {preset}. Available presets: {list(recommendations.keys())}")
                return
        
        # Handle manual parameter specification
        using_manual_params = False
        if words_per_gpu_batch is not None or global_hash_map_bits is not None or cracked_hash_map_bits is not None:
            using_manual_params = True
            print(f"🔧 Using manually specified parameters:")
            
            # Set defaults for any unspecified manual parameters
            if words_per_gpu_batch is None:
                words_per_gpu_batch = DEFAULT_WORDS_PER_GPU_BATCH
            if global_hash_map_bits is None:
                global_hash_map_bits = DEFAULT_GLOBAL_HASH_MAP_BITS
            if cracked_hash_map_bits is None:
                cracked_hash_map_bits = DEFAULT_CRACKED_HASH_MAP_BITS
                
            print(f"   - Batch size: {words_per_gpu_batch:,}")
            print(f"   - Global hash map: {global_hash_map_bits} bits")
            print(f"   - Cracked hash map: {cracked_hash_map_bits} bits")
            
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
                print(f"⚠️  Warning: Manual parameters exceed available VRAM!")
                print(f"   Required: {total_batch_memory / (1024**3):.2f} GB")
                print(f"   Available: {available_vram / (1024**3):.2f} GB")
                print("   Consider reducing batch size or hash map bits")
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
        
        print(f"✅ OpenCL initialized on device: {device.name.strip()}")
    except Exception as e:
        print(f"❌ OpenCL initialization or kernel compilation error: {e}")
        exit(1)

    # 2. DATA LOADING AND PRE-ENCODING
    rule_size_in_int = 2 + MAX_RULE_ARGS
    encoded_rules = [encode_rule(rule['rule_data'], rule['rule_id'], MAX_RULE_ARGS) for rule in rules_list]

    # 3. HASH MAP INITIALIZATION
    global_hash_map_np = np.zeros(GLOBAL_HASH_MAP_WORDS, dtype=np.uint32)
    print(f"📝 Global Hash Map initialized: {global_hash_map_np.nbytes / (1024*1024):.2f} MB allocated.")

    cracked_hash_map_np = np.zeros(CRACKED_HASH_MAP_WORDS, dtype=np.uint32)
    print(f"📝 Cracked Hash Map initialized: {cracked_hash_map_np.nbytes / (1024*1024):.2f} MB allocated.")

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
        print(f"❌ Fatal: Could not allocate GPU memory even after retries: {e}")
        print("💡 Try reducing batch size or hash map bits, or use a preset:")
        recommendations, _ = get_recommended_parameters(device, total_words, cracked_hashes_count)
        for preset_name, config in recommendations.items():
            if preset_name != "auto":
                print(f"   --preset {preset_name}: {config['description']}")
        return

    current_word_buffer_idx = 0
    copy_events = [None, None]

    # 5. INITIALIZE CRACKED HASH MAP (ONCE)
    if cracked_hashes_np.size > 0 and cracked_temp_g is not None:
        global_size_init_cracked = (int(math.ceil(cracked_hashes_np.size / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
        local_size_init_cracked = (LOCAL_WORK_SIZE,)

        print("Populating static Cracked Hash Map on GPU...")
        kernel_init(queue, global_size_init_cracked, local_size_init_cracked,
                    cracked_hash_map_g,
                    cracked_temp_g,
                    np.uint32(cracked_hashes_np.size),
                    np.uint32(CRACKED_HASH_MAP_MASK)).wait()
        print("Static Cracked Hash Map populated.")
    else:
        print("Cracked list is empty, effectiveness scoring is disabled.")
        
    # 6. PIPELINED RANKING LOOP SETUP
    word_iterator = wordlist_iterator_optimized(wordlist_path, MAX_WORD_LEN, words_per_gpu_batch)
    rule_batch_starts = list(range(0, total_rules, MAX_RULES_IN_BATCH))
    
    print(f"📊 Processing {total_rules:,} rules in {len(rule_batch_starts):,} batches "
          f"(up to {MAX_RULES_IN_BATCH:,} rules per batch)")
    
    # Use numpy arrays for counters to avoid overflow
    words_processed_total = np.uint64(0)
    total_unique_found = np.uint64(0)
    total_cracked_found = np.uint64(0)
    
    mapped_uniqueness_np = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
    mapped_effectiveness_np = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
    
    num_words_batch_exec = 0
    
    word_batch_pbar = tqdm(total=total_words, desc="Processing wordlist from disk [Unique: 0 | Cracked: 0]", unit=" word")
    rule_batch_pbar = tqdm(total=len(rule_batch_starts), desc="Rule batches processed", unit=" batch")

    # A. Initial word batch load
    try:
        base_words_np_batch, num_words_batch_exec, base_hashes_np_batch = next(word_iterator)
        cl.enqueue_copy(queue, base_words_in_g[current_word_buffer_idx], base_words_np_batch)
        cl.enqueue_copy(queue, base_hashes_g[current_word_buffer_idx], base_hashes_np_batch).wait()
    except StopIteration:
        print("Wordlist is empty or too small.")
        word_batch_pbar.close()
        rule_batch_pbar.close()
        return

    # B. Main Pipelined Loop
    while True:
        exec_idx = current_word_buffer_idx
        next_idx = 1 - current_word_buffer_idx
        
        # 7. ASYNC COPY: Start load of the *next* batch
        next_batch_data = None
        try:
            next_batch_data = next(word_iterator)
            base_words_np_next, num_words_batch_next, base_hashes_np_next = next_batch_data
            copy_events[next_idx] = cl.enqueue_copy(queue, base_words_in_g[next_idx], base_words_np_next)
            cl.enqueue_copy(queue, base_hashes_g[next_idx], base_hashes_np_next, wait_for=[copy_events[next_idx]])
        except StopIteration:
            next_batch_data = None
            copy_events[next_idx] = None
        
        # 8. PROCESS RULE BATCHES
        for rule_batch_idx, rule_start_index in enumerate(rule_batch_starts):
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

            for i in range(num_rules_in_batch):
                rule_index = rule_start_index + i
                # Convert to Python int to avoid overflow in rule scores
                uniqueness_val = int(mapped_uniqueness_np[i])
                effectiveness_val = int(mapped_effectiveness_np[i])
                
                rules_list[rule_index]['uniqueness_score'] += uniqueness_val
                rules_list[rule_index]['effectiveness_score'] += effectiveness_val
                total_unique_found += np.uint64(uniqueness_val)
                total_cracked_found += np.uint64(effectiveness_val)

            rule_batch_pbar.update(1)

        # 9. UPDATE AND PREPARE NEXT ITERATION
        words_processed_total += np.uint64(num_words_batch_exec)
        word_batch_pbar.update(num_words_batch_exec)
        word_batch_pbar.set_description(f"Processing wordlist from disk [Unique: {int(total_unique_found):,} | Cracked: {int(total_cracked_found):,}]")

        if next_batch_data is None:
            break
            
        current_word_buffer_idx = next_idx
        num_words_batch_exec = num_words_batch_next

    word_batch_pbar.close()
    rule_batch_pbar.close()
    end_time = time()
    
    # 10. FINAL REPORTING AND SAVING
    print("\n--- Final Results Summary ---")
    print(f"Total Words Processed: {int(words_processed_total):,}")
    print(f"Total Rules Processed: {total_rules:,}")
    print(f"Total Unique Words Generated: {int(total_unique_found):,}")
    print(f"Total True Cracks Found: {int(total_cracked_found):,}")
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
    print("-----------------------------\n")

    csv_path = save_ranking_data(rules_list, ranking_output_path)
    if top_k > 0:
        optimized_output_path = os.path.splitext(ranking_output_path)[0] + "_optimized.rule"
        load_and_save_optimized_rules(csv_path, optimized_output_path, top_k)

# ====================================================================
# --- MAIN EXECUTION ---
# ====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPU-Accelerated Hashcat Rule Ranking Tool (Ranker v3.1 - Large Rule Support)")
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

    print("=" * 70)
    print("* RANKER v3.1 - FIXED VERSION")
    print("* FIXED: Now saves ALL rules as requested by -k parameter")
    print("* FIXED: Removed filtering that excluded zero-score rules")
    print("* IMPROVED: MAX_RULES_IN_BATCH increased to 1024")
    print("* ENHANCED: Better memory management for large rule sets")
    print("* ENHANCED: Improved timing information for large wordlists")
    print("=" * 70)

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

        # 9. UPDATE AND PREPARE NEXT ITERATION
        words_processed_total += np.uint64(num_words_batch_exec)
        word_batch_pbar.update(num_words_batch_exec)
        word_batch_pbar.set_description(f"Processing wordlist from disk [Unique: {int(total_unique_found):,} | Cracked: {int(total_cracked_found):,}]")

        if next_batch_data is None:
            break
            
        current_word_buffer_idx = next_idx
        num_words_batch_exec = num_words_batch_next

    word_batch_pbar.close()
    rule_batch_pbar.close()
    end_time = time()
    
    # 10. FINAL REPORTING AND SAVING
    print("\n--- Final Results Summary ---")
    print(f"Total Words Processed: {int(words_processed_total):,}")
    print(f"Total Rules Processed: {total_rules:,}")
    print(f"Total Unique Words Generated: {int(total_unique_found):,}")
    print(f"Total True Cracks Found: {int(total_cracked_found):,}")
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
    print("-----------------------------\n")

    csv_path = save_ranking_data(rules_list, ranking_output_path)
    if top_k > 0:
        optimized_output_path = os.path.splitext(ranking_output_path)[0] + "_optimized.rule"
        load_and_save_optimized_rules(csv_path, optimized_output_path, top_k)

# ====================================================================
# --- MAIN EXECUTION ---
# ====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPU-Accelerated Hashcat Rule Ranking Tool (Ranker v3.1 - Large Rule Support)")
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

    print("=" * 70)
    print("* RANKER v3.1 - FIXED VERSION")
    print("* FIXED: Now saves ALL rules as requested by -k parameter")
    print("* FIXED: Removed filtering that excluded zero-score rules")
    print("* IMPROVED: MAX_RULES_IN_BATCH increased to 1024")
    print("* ENHANCED: Better memory management for large rule sets")
    print("=" * 70)

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
