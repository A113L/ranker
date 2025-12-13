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
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====================================================================
# --- RULE BATCHING FUNCTIONS  OPTIMIZATION ---
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

def red(text): return f"{Colors.RED}{text}{Colors.END}"
def green(text): return f"{Colors.GREEN}{text}{Colors.END}"
def yellow(text): return f"{Colors.YELLOW}{text}{Colors.END}"
def blue(text): return f"{Colors.BLUE}{text}{Colors.END}"
def magenta(text): return f"{Colors.MAGENTA}{text}{Colors.END}"
def cyan(text): return f"{Colors.CYAN}{text}{Colors.END}"
def bold(text): return f"{Colors.BOLD}{text}{Colors.END}"
def underline(text): return f"{Colors.UNDERLINE}{text}{Colors.END}"

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

# Rule IDs
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

# Memory management
VRAM_SAFETY_MARGIN = 0.15  # 15% safety margin
MIN_BATCH_SIZE = 50000     # Minimum batch size to maintain performance
MIN_HASH_MAP_BITS = 28     # Minimum hash map size (256MB)

# Performance flags
PERFORMANCE_PROFILES = {
    'low_memory': {
        'description': 'Low Memory Mode (for GPUs with < 4GB VRAM)',
        'batch_size': 50000,
        'global_bits': 30,
        'cracked_bits': 28
    },
    'medium_memory': {
        'description': 'Medium Memory Mode (for GPUs with 4-8GB VRAM)',
        'batch_size': 150000,
        'global_bits': 33,
        'cracked_bits': 31
    },
    'high_memory': {
        'description': 'High Memory Mode (for GPUs with > 8GB VRAM)',
        'batch_size': 300000,
        'global_bits': 35,
        'cracked_bits': 33
    },
    'aggressive': {
        'description': 'Aggressive Mode (Maximum performance)',
        'batch_size': 400000,
        'global_bits': 37,
        'cracked_bits': 35
    },
    'balanced': {
        'description': 'Balanced Mode (Best balance of speed and memory)',
        'batch_size': 200000,
        'global_bits': 34,
        'cracked_bits': 32
    }
}

# ====================================================================
# --- OPTIMIZED RULE BATCHING FUNCTIONS ---
# ====================================================================

def encode_rules_batch(rules_list, start_idx, end_idx, max_args=4):
    """Encode a batch of rules into a compact numpy array for GPU transfer"""
    rule_size_in_int = 2 + max_args
    batch_size = end_idx - start_idx
    
    # Pre-allocate array for maximum efficiency
    encoded_batch = np.zeros(batch_size * rule_size_in_int, dtype=np.uint32)
    
    for i, rule_idx in enumerate(range(start_idx, end_idx)):
        rule = rules_list[rule_idx]
        rule_str = rule['rule_data']
        rule_id = rule['rule_id']
        
        # Calculate array position
        base_idx = i * rule_size_in_int
        
        # Store rule ID
        encoded_batch[base_idx] = np.uint32(rule_id)
        
        # Encode rule string bytes
        rule_bytes = rule_str.encode('latin-1')
        rule_len = len(rule_bytes)
        
        # Pack first 4 bytes into first integer
        args_int = 0
        for j, byte in enumerate(rule_bytes[:4]):
            args_int |= (byte << (j * 8))
        encoded_batch[base_idx + 1] = np.uint32(args_int)
        
        # Pack remaining bytes if needed
        if rule_len > 4:
            args_int2 = 0
            for j, byte in enumerate(rule_bytes[4:8]):
                args_int2 |= (byte << (j * 8))
            encoded_batch[base_idx + 2] = np.uint32(args_int2)
        
        # Pack more bytes if needed for complex rules
        if rule_len > 8:
            args_int3 = 0
            for j, byte in enumerate(rule_bytes[8:12]):
                args_int3 |= (byte << (j * 8))
            encoded_batch[base_idx + 3] = np.uint32(args_int3)
    
    return encoded_batch

def create_rule_batches(rules_list, batch_size=1024):
    """Create optimized batches of rules for GPU processing"""
    total_rules = len(rules_list)
    batches = []
    
    for start_idx in range(0, total_rules, batch_size):
        end_idx = min(start_idx + batch_size, total_rules)
        batches.append((start_idx, end_idx))
    
    return batches

def precompute_rule_batches(rules_list, batch_size=1024, num_threads=4):
    """Precompute all rule batches in parallel for maximum throughput"""
    print(f"{blue('⚡')} {bold('Precomputing rule batches in parallel...')}")
    
    batches = create_rule_batches(rules_list, batch_size)
    encoded_batches = [None] * len(batches)
    
    # Use ThreadPoolExecutor for parallel encoding
    def encode_single_batch(batch_idx, start, end):
        return batch_idx, encode_rules_batch(rules_list, start, end)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for batch_idx, (start, end) in enumerate(batches):
            futures.append(executor.submit(encode_single_batch, batch_idx, start, end))
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Encoding rules", unit="batch"):
            batch_idx, encoded = future.result()
            encoded_batches[batch_idx] = encoded
    
    print(f"{green('✅')} {bold('Precomputed')} {cyan(str(len(batches)))} {bold('rule batches')}")
    return batches, encoded_batches

# ====================================================================
# --- MEMORY MAPPED WORDLIST LOADER ---
# ====================================================================

class MemoryMappedWordlist:
    """Memory-mapped wordlist loader for efficient file access"""
    
    def __init__(self, filepath, max_word_len=32, batch_size=100000):
        self.filepath = filepath
        self.max_word_len = max_word_len
        self.batch_size = batch_size
        self.mmapped_file = None
        self.word_positions = []
        self.total_words = 0
        self.total_batches = 0
        self.current_batch = 0
        
    def __enter__(self):
        """Initialize memory mapping and scan file"""
        print(f"{blue('🗺️')}  {bold('Memory mapping wordlist...')}")
        
        # Open file and create memory map
        self.file_obj = open(self.filepath, 'rb')
        self.mmapped_file = mmap.mmap(self.file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Scan file to find word positions and count
        print(f"{blue('🔍')} {bold('Scanning wordlist...')}")
        
        pos = 0
        file_size = len(self.mmapped_file)
        
        with tqdm(total=file_size, desc="Scanning words", unit="B") as pbar:
            while pos < file_size:
                # Find next newline
                next_newline = self.mmapped_file.find(b'\n', pos)
                if next_newline == -1:  # Last line
                    next_newline = file_size
                
                word_start = pos
                word_end = next_newline
                word_length = word_end - word_start
                
                # Only store valid words
                if 1 <= word_length <= self.max_word_len:
                    self.word_positions.append((word_start, word_length))
                
                pos = next_newline + 1  # Skip newline
                pbar.update(word_end - word_start + 1)
        
        self.total_words = len(self.word_positions)
        self.total_batches = (self.total_words + self.batch_size - 1) // self.batch_size
        
        print(f"{green('✅')} {bold('Wordlist mapped:')} {cyan(f'{self.total_words:,}')} words in {cyan(f'{self.total_batches}')} batches")
        print(f"   {bold('File size:')} {cyan(f'{file_size / (1024**2):.1f} MB')}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up memory mapping"""
        if self.mmapped_file:
            self.mmapped_file.close()
        if self.file_obj:
            self.file_obj.close()
    
    def __iter__(self):
        """Iterate through word batches"""
        self.current_batch = 0
        return self
    
    def __next__(self):
        """Get next batch of words"""
        if self.current_batch >= self.total_batches:
            raise StopIteration
        
        # Calculate batch range
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_words)
        batch_word_count = end_idx - start_idx
        
        # Prepare buffers
        words_buffer = np.zeros(batch_word_count * self.max_word_len, dtype=np.uint8)
        hashes_buffer = np.zeros(batch_word_count, dtype=np.uint32)
        
        # Process batch
        for i in range(batch_word_count):
            word_idx = start_idx + i
            word_start, word_length = self.word_positions[word_idx]
            
            # Copy word from memory map
            word_data = self.mmapped_file[word_start:word_start + word_length]
            word_start_offset = i * self.max_word_len
            words_buffer[word_start_offset:word_start_offset + word_length] = np.frombuffer(word_data, dtype=np.uint8)
            
            # Compute hash
            hash_val = 2166136261
            for byte in word_data:
                hash_val = (hash_val ^ byte) * 16777619 & 0xFFFFFFFF
            hashes_buffer[i] = hash_val
        
        self.current_batch += 1
        
        return words_buffer, hashes_buffer, batch_word_count
    
    def get_word_count(self):
        """Get total number of words"""
        return self.total_words
    
    def get_batch_count(self):
        """Get total number of batches"""
        return self.total_batches

def create_wordlist_batches(wordlist_path, max_word_len=32, batch_size=100000):
    """Generator function that yields word batches using memory mapping"""
    with MemoryMappedWordlist(wordlist_path, max_word_len, batch_size) as wordlist:
        total_batches = wordlist.get_batch_count()
        
        # Create progress bar for word batches
        with tqdm(total=total_batches, desc="Processing word batches", unit="batch") as pbar:
            for batch_idx, (words_buffer, hashes_buffer, word_count) in enumerate(wordlist):
                yield words_buffer, hashes_buffer, word_count
                
                # Update progress
                pbar.update(1)
                pbar.set_postfix({
                    'words': f'{word_count:,}',
                    'total': f'{(batch_idx + 1) * batch_size:,}'
                })

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

def calculate_optimal_parameters(available_vram, total_words, cracked_hashes_count, total_rules, reduction_factor=1.0):
    """
    Calculate optimal parameters with consideration for large rule sets
    """
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
        optimal_batch_size = max(MIN_BATCH_SIZE, optimal_batch_size // 2)
    
    return optimal_batch_size, global_bits, cracked_bits

def get_performance_profile_info():
    """Get information about available performance profiles"""
    print(f"{blue('🔧')} {bold('Available Performance Profiles:')}")
    for profile_name, config in PERFORMANCE_PROFILES.items():
        print(f"   {cyan(profile_name)}: {config['description']}")
        print(f"      Batch size: {config['batch_size']:,}, "
              f"Global map: {config['global_bits']} bits, "
              f"Cracked map: {config['cracked_bits']} bits")

# ====================================================================
# --- GPU INITIALIZATION ---
# ====================================================================

def initialize_gpu_environment(target_device="RTX 3060 Ti", force_nvidia=False, force_amd=False):
    """Initialize GPU environment optimized for specific hardware"""
    try:
        platforms = cl.get_platforms()
        
        # Platform selection logic
        selected_platform = None
        
        if force_nvidia:
            for platform in platforms:
                if "NVIDIA" in platform.name.upper():
                    selected_platform = platform
                    print(f"{green('🎮')} {bold('Forced NVIDIA platform:')} {platform.name}")
                    break
        
        elif force_amd:
            for platform in platforms:
                if "AMD" in platform.name.upper():
                    selected_platform = platform
                    print(f"{green('🎮')} {bold('Forced AMD platform:')} {platform.name}")
                    break
        
        else:
            # Auto-select platform
            for platform in platforms:
                if "NVIDIA" in platform.name.upper():
                    selected_platform = platform
                    print(f"{green('🎮')} {bold('Found NVIDIA platform:')} {platform.name}")
                    break
            
            if not selected_platform:
                selected_platform = platforms[0]
                print(f"{blue('🎮')} {bold('Using platform:')} {selected_platform.name}")
        
        platform = selected_platform
        
        # Find GPU devices
        devices = platform.get_devices(cl.device_type.GPU)
        if not devices:
            devices = platform.get_devices(cl.device_type.ALL)
        
        # Device selection logic
        target_device_lower = target_device.lower() if target_device else ""
        selected_device = None
        
        for device in devices:
            device_name = device.name.strip()
            if target_device_lower and target_device_lower in device_name.lower():
                selected_device = device
                print(f"{green('✅')} {bold('Found target device:')} {cyan(device_name)}")
                break
        
        if not selected_device:
            selected_device = devices[0]
            print(f"{blue('🎮')} {bold('Using device:')} {cyan(selected_device.name.strip())}")
        
        # Create context and queue with optimization flags
        context = cl.Context([selected_device])
        
        # Optimize queue for compute-intensive workloads
        queue = cl.CommandQueue(context, selected_device, 
                               properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        # Get device info
        device_name = selected_device.name.strip()
        compute_units = selected_device.max_compute_units
        clock_freq = selected_device.max_clock_frequency
        global_mem = selected_device.global_mem_size
        
        print(f"{blue('📊')} {bold('Device specs:')}")
        print(f"   {bold('Compute Units:')} {cyan(str(compute_units))}")
        print(f"   {bold('Clock Frequency:')} {cyan(f'{clock_freq} MHz')}")
        print(f"   {bold('Global Memory:')} {cyan(f'{global_mem / (1024**3):.1f} GB')}")
        
        return platform, selected_device, context, queue
        
    except Exception as e:
        print(f"{red('❌')} {bold('GPU initialization error:')} {e}")
        raise

# ====================================================================
# --- FULL OPTIMIZED KERNEL SOURCE ---
# ====================================================================

def get_optimized_kernel_source(config):
    """Generate fully optimized OpenCL kernel source """
    
    # Extract configuration
    max_word_len = config['max_word_len']
    max_output_len = config['max_output_len']
    max_rule_args = config['max_rule_args']
    local_work_size = config['local_work_size']
    global_hash_map_mask = config['global_hash_map_mask']
    cracked_hash_map_mask = config['cracked_hash_map_mask']
    
    # Rule ID constants
    start_id_simple = config['start_id_simple']
    num_simple_rules = config['num_simple_rules']
    start_id_td = config['start_id_td']
    num_td_rules = config['num_td_rules']
    start_id_s = config['start_id_s']
    num_s_rules = config['num_s_rules']
    start_id_a = config['start_id_a']
    num_a_rules = config['num_a_rules']
    start_id_groupb = config['start_id_groupb']
    num_groupb_rules = config['num_groupb_rules']
    start_id_new = config['start_id_new']
    num_new_rules = config['num_new_rules']
    
    # FULL OPTIMIZED KERNEL SOURCE WITH CORRECT CONST QUALIFIERS
    kernel_source = f"""
// ====================================================================
// ULTRA-OPTIMIZED RULE PROCESSING KERNEL 
// ====================================================================

// Fast FNV-1a hash using loop unrolling for maximum throughput
unsigned int fnv1a_hash_fast(const unsigned char* data, unsigned int len) {{
    unsigned int hash = 2166136261U;
    
    // Process 4 bytes at a time when possible
    unsigned int i = 0;
    for (; i + 3 < len; i += 4) {{
        hash ^= data[i];
        hash *= 16777619U;
        hash ^= data[i + 1];
        hash *= 16777619U;
        hash ^= data[i + 2];
        hash *= 16777619U;
        hash ^= data[i + 3];
        hash *= 16777619U;
    }}
    
    // Process remaining bytes
    for (; i < len; i++) {{
        hash ^= data[i];
        hash *= 16777619U;
    }}
    
    return hash;
}}

// Fast character to position conversion using lookup optimization
unsigned int char_to_pos_fast(unsigned char c) {{
    // Fast lookup for common cases
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    return 0xFFFFFFFF;
}}

// Main kernel - optimized compute architecture
__kernel __attribute__((reqd_work_group_size({local_work_size}, 1, 1)))
void bfs_kernel_optimized(
    __global const unsigned char* base_words_in,           // READ-ONLY: Input words
    __global const unsigned int* rules_in,                 // READ-ONLY: Input rules
    __global unsigned int* rule_uniqueness_counts,         // WRITABLE: Unique word counter
    __global unsigned int* rule_effectiveness_counts,      // WRITABLE: Cracked password counter
    __global unsigned int* global_hash_map,                // WRITABLE: Global uniqueness tracking
    __global const unsigned int* cracked_hash_map,         // READ-ONLY: Pre-populated cracked passwords
    const unsigned int num_words,
    const unsigned int num_rules_in_batch,
    const unsigned int max_word_len,
    const unsigned int max_output_len,
    const unsigned int global_map_mask,
    const unsigned int cracked_map_mask)
{{
    unsigned int global_id = get_global_id(0);
    
    // Early exit for threads beyond required work
    if (global_id >= num_words * num_rules_in_batch) return;
    
    // Calculate indices - optimized division
    unsigned int word_idx = global_id / num_rules_in_batch;
    unsigned int rule_batch_idx = global_id % num_rules_in_batch;
    
    // Pre-calculate memory offsets
    unsigned int word_offset = word_idx * max_word_len;
    unsigned int rule_offset = rule_batch_idx * {2 + max_rule_args};
    
    // Load word into local memory
    unsigned char word_buffer[{max_word_len}];
    unsigned int word_len = 0;
    
    __global const unsigned char* word_ptr = base_words_in + word_offset;
    for (unsigned int i = 0; i < max_word_len; i++) {{
        unsigned char c = word_ptr[i];
        if (c == 0) {{
            word_len = i;
            break;
        }}
        word_buffer[i] = c;
    }}
    
    // Early exit for empty words with non-append rules
    unsigned int rule_id = rules_in[rule_offset];
    if (word_len == 0 && rule_id < {start_id_a}) return;
    
    // Load rule arguments - optimized single load
    unsigned int rule_args_int = rules_in[rule_offset + 1];
    unsigned int rule_args_int2 = rules_in[rule_offset + 2];
    unsigned int rule_args_int3 = rules_in[rule_offset + 3];
    
    // Unpack arguments once
    unsigned char arg0 = (unsigned char)(rule_args_int & 0xFF);
    unsigned char arg1 = (unsigned char)((rule_args_int >> 8) & 0xFF);
    unsigned char arg2 = (unsigned char)((rule_args_int >> 16) & 0xFF);
    unsigned char arg3 = (unsigned char)(rule_args_int2 & 0xFF);
    unsigned char arg4 = (unsigned char)((rule_args_int2 >> 8) & 0xFF);
    
    // Result buffer in local memory
    unsigned char result[{max_output_len}];
    unsigned int out_len = word_len;
    bool changed_flag = false;
    
    // Copy word to result initially
    for (unsigned int i = 0; i < word_len; i++) {{
        result[i] = word_buffer[i];
    }}
    
    // ============================================================
    // OPTIMIZED RULE DISPATCH WITH REDUCED BRANCHING
    // ============================================================
    
    // Simple rules (most common - 70% of cases)
    if (rule_id >= {start_id_simple} && rule_id < {start_id_simple} + {num_simple_rules}) {{
        unsigned int rule_type = rule_id - {start_id_simple};
        
        switch(rule_type) {{
            case 0: {{ // 'l' lowercase - optimized with SIMD-like approach
                for (unsigned int i = 0; i < word_len; i++) {{
                    unsigned char c = word_buffer[i];
                    if (c >= 'A' && c <= 'Z') {{
                        result[i] = c + 32;
                        changed_flag = true;
                    }}
                }}
                break;
            }}
            case 1: {{ // 'u' uppercase
                for (unsigned int i = 0; i < word_len; i++) {{
                    unsigned char c = word_buffer[i];
                    if (c >= 'a' && c <= 'z') {{
                        result[i] = c - 32;
                        changed_flag = true;
                    }}
                }}
                break;
            }}
            case 2: {{ // 'c' capitalize
                if (word_len > 0) {{
                    unsigned char first = word_buffer[0];
                    if (first >= 'a' && first <= 'z') {{
                        result[0] = first - 32;
                        changed_flag = true;
                    }}
                    for (unsigned int i = 1; i < word_len; i++) {{
                        unsigned char c = word_buffer[i];
                        if (c >= 'A' && c <= 'Z') {{
                            result[i] = c + 32;
                            changed_flag = true;
                        }}
                    }}
                }}
                break;
            }}
            case 3: {{ // 'C' invert capitalize
                if (word_len > 0) {{
                    unsigned char first = word_buffer[0];
                    if (first >= 'A' && first <= 'Z') {{
                        result[0] = first + 32;
                        changed_flag = true;
                    }}
                    for (unsigned int i = 1; i < word_len; i++) {{
                        unsigned char c = word_buffer[i];
                        if (c >= 'a' && c <= 'z') {{
                            result[i] = c - 32;
                            changed_flag = true;
                        }}
                    }}
                }}
                break;
            }}
            case 4: {{ // 't' toggle case
                for (unsigned int i = 0; i < word_len; i++) {{
                    unsigned char c = word_buffer[i];
                    if (c >= 'a' && c <= 'z') {{
                        result[i] = c - 32;
                        changed_flag = true;
                    }} else if (c >= 'A' && c <= 'Z') {{
                        result[i] = c + 32;
                        changed_flag = true;
                    }}
                }}
                break;
            }}
            case 5: {{ // 'r' reverse
                if (word_len > 1) {{
                    for (unsigned int i = 0; i < word_len; i++) {{
                        result[i] = word_buffer[word_len - 1 - i];
                    }}
                    changed_flag = true;
                }}
                break;
            }}
            case 6: {{ // 'k' swap first two
                if (word_len >= 2) {{
                    result[0] = word_buffer[1];
                    result[1] = word_buffer[0];
                    changed_flag = true;
                }}
                break;
            }}
            case 7: {{ // ':' no change
                changed_flag = false;
                break;
            }}
            case 8: {{ // 'd' duplicate
                unsigned int new_len = word_len * 2;
                if (new_len <= max_output_len) {{
                    for (unsigned int i = 0; i < word_len; i++) {{
                        result[i] = word_buffer[i];
                        result[word_len + i] = word_buffer[i];
                    }}
                    out_len = new_len;
                    changed_flag = true;
                }}
                break;
            }}
            case 9: {{ // 'f' reflect
                unsigned int new_len = word_len * 2;
                if (new_len <= max_output_len) {{
                    for (unsigned int i = 0; i < word_len; i++) {{
                        result[i] = word_buffer[i];
                        result[word_len + i] = word_buffer[word_len - 1 - i];
                    }}
                    out_len = new_len;
                    changed_flag = true;
                }}
                break;
            }}
        }}
    }}
    
    // T/D rules (toggle/delete at position)
    else if (rule_id >= {start_id_td} && rule_id < {start_id_td} + {num_td_rules}) {{
        unsigned char operator_char = arg0;
        unsigned int pos = char_to_pos_fast(arg1);
        
        if (operator_char == 'T') {{ // Toggle case at position
            if (pos != 0xFFFFFFFF && pos < word_len) {{
                unsigned char c = word_buffer[pos];
                if (c >= 'a' && c <= 'z') {{
                    result[pos] = c - 32;
                    changed_flag = true;
                }} else if (c >= 'A' && c <= 'Z') {{
                    result[pos] = c + 32;
                    changed_flag = true;
                }}
            }}
        }}
        else if (operator_char == 'D') {{ // Delete at position
            if (pos != 0xFFFFFFFF && pos < word_len) {{
                unsigned int new_idx = 0;
                for (unsigned int i = 0; i < word_len; i++) {{
                    if (i != pos) {{
                        result[new_idx++] = word_buffer[i];
                    }}
                }}
                out_len = new_idx;
                changed_flag = true;
            }}
        }}
    }}
    
    // s/ substitution rules
    else if (rule_id >= {start_id_s} && rule_id < {start_id_s} + {num_s_rules}) {{
        unsigned char find_char = arg0;
        unsigned char replace_char = arg1;
        
        for (unsigned int i = 0; i < word_len; i++) {{
            if (word_buffer[i] == find_char) {{
                result[i] = replace_char;
                changed_flag = true;
            }}
        }}
    }}
    
    // A rules (append/prepend/delete)
    else if (rule_id >= {start_id_a} && rule_id < {start_id_a} + {num_a_rules}) {{
        unsigned char cmd = arg0;
        unsigned char arg_char = arg1;
        
        if (cmd == '^') {{ // Prepend
            if (word_len + 1 <= max_output_len) {{
                for (unsigned int i = word_len; i > 0; i--) {{
                    result[i] = result[i - 1];
                }}
                result[0] = arg_char;
                out_len = word_len + 1;
                changed_flag = true;
            }}
        }}
        else if (cmd == '$') {{ // Append
            if (word_len + 1 <= max_output_len) {{
                result[word_len] = arg_char;
                out_len = word_len + 1;
                changed_flag = true;
            }}
        }}
        else if (cmd == '@') {{ // Delete all occurrences
            unsigned int new_idx = 0;
            for (unsigned int i = 0; i < word_len; i++) {{
                if (word_buffer[i] != arg_char) {{
                    result[new_idx++] = word_buffer[i];
                }}
            }}
            if (new_idx != word_len) {{
                out_len = new_idx;
                changed_flag = true;
            }}
        }}
    }}
    
    // Group B rules
    else if (rule_id >= {start_id_groupb} && rule_id < {start_id_groupb} + {num_groupb_rules}) {{
        unsigned char cmd = arg0;
        unsigned int N = char_to_pos_fast(arg1);
        unsigned int M = char_to_pos_fast(arg2);
        unsigned char X = arg3;
        
        if (cmd == 'p') {{ // Duplicate N times
            if (N != 0xFFFFFFFF && N > 0) {{
                unsigned int total_len = word_len * (N + 1);
                if (total_len <= max_output_len) {{
                    for (unsigned int dup = 1; dup <= N; dup++) {{
                        unsigned int offset = word_len * dup;
                        for (unsigned int i = 0; i < word_len; i++) {{
                            result[offset + i] = word_buffer[i];
                        }}
                    }}
                    out_len = total_len;
                    changed_flag = true;
                }}
            }}
        }}
        else if (cmd == 'q') {{ // Duplicate all characters
            unsigned int total_len = word_len * 2;
            if (total_len <= max_output_len) {{
                for (unsigned int i = 0; i < word_len; i++) {{
                    result[i * 2] = word_buffer[i];
                    result[i * 2 + 1] = word_buffer[i];
                }}
                out_len = total_len;
                changed_flag = true;
            }}
        }}
        else if (cmd == '{{') {{ // Rotate left
            if (word_len > 0) {{
                unsigned char first = result[0];
                for (unsigned int i = 0; i < word_len - 1; i++) {{
                    result[i] = result[i + 1];
                }}
                result[word_len - 1] = first;
                changed_flag = true;
            }}
        }}
        else if (cmd == '}}') {{ // Rotate right
            if (word_len > 0) {{
                unsigned char last = result[word_len - 1];
                for (unsigned int i = word_len - 1; i > 0; i--) {{
                    result[i] = result[i - 1];
                }}
                result[0] = last;
                changed_flag = true;
            }}
        }}
        else if (cmd == '[') {{ // Truncate left
            if (word_len > 0) {{
                out_len = word_len - 1;
                for (unsigned int i = 0; i < out_len; i++) {{
                    result[i] = result[i + 1];
                }}
                changed_flag = true;
            }}
        }}
        else if (cmd == ']') {{ // Truncate right
            if (word_len > 0) {{
                out_len = word_len - 1;
                changed_flag = true;
            }}
        }}
        else if (cmd == 'x') {{ // Extract range
            if (N != 0xFFFFFFFF && M != 0xFFFFFFFF && N < word_len && M > 0) {{
                unsigned int end = min(N + M, word_len);
                out_len = end - N;
                for (unsigned int i = 0; i < out_len; i++) {{
                    result[i] = word_buffer[N + i];
                }}
                changed_flag = true;
            }}
        }}
        else if (cmd == 'i') {{ // Insert at position
            if (N != 0xFFFFFFFF && word_len + 1 <= max_output_len) {{
                unsigned int pos = min(N, word_len);
                for (unsigned int i = word_len; i > pos; i--) {{
                    result[i] = result[i - 1];
                }}
                result[pos] = X;
                out_len = word_len + 1;
                changed_flag = true;
            }}
        }}
        else if (cmd == 'o') {{ // Overwrite at position
            if (N != 0xFFFFFFFF && N < word_len) {{
                result[N] = X;
                changed_flag = true;
            }}
        }}
    }}
    
    // New comprehensive rules
    else if (rule_id >= {start_id_new} && rule_id < {start_id_new} + {num_new_rules}) {{
        unsigned char cmd = arg0;
        unsigned int N = char_to_pos_fast(arg1);
        unsigned int M = char_to_pos_fast(arg2);
        unsigned char X = arg3;
        
        if (cmd == 'K') {{ // Swap last two
            if (word_len >= 2) {{
                unsigned char temp = result[word_len - 1];
                result[word_len - 1] = result[word_len - 2];
                result[word_len - 2] = temp;
                changed_flag = true;
            }}
        }}
        else if (cmd == '*') {{ // Swap characters
            if (N != 0xFFFFFFFF && M != 0xFFFFFFFF && N < word_len && M < word_len && N != M) {{
                unsigned char temp = result[N];
                result[N] = result[M];
                result[M] = temp;
                changed_flag = true;
            }}
        }}
        else if (cmd == 'E') {{ // Title case
            bool capitalize_next = true;
            for (unsigned int i = 0; i < word_len; i++) {{
                unsigned char c = word_buffer[i];
                if (c >= 'A' && c <= 'Z') {{
                    c += 32;
                }}
                
                if (capitalize_next && c >= 'a' && c <= 'z') {{
                    result[i] = c - 32;
                    if (result[i] != word_buffer[i]) changed_flag = true;
                }} else {{
                    result[i] = c;
                }}
                
                capitalize_next = (result[i] == ' ');
            }}
        }}
    }}
    
    // ============================================================
    // DUAL-UNIQUENESS CHECK WITH OPTIMIZED MEMORY ACCESS
    // ============================================================
    
    if (changed_flag && out_len > 0) {{
        // Fast hash computation
        unsigned int word_hash = fnv1a_hash_fast(result, out_len);
        
        // Check global hash map (uniqueness)
        unsigned int global_map_index = (word_hash >> 5) & global_map_mask;
        unsigned int bit_index = word_hash & 31;
        unsigned int check_bit = (1U << bit_index);
        
        __global unsigned int* global_map_ptr = &global_hash_map[global_map_index];
        unsigned int global_word = *global_map_ptr;
        
        if (!(global_word & check_bit)) {{
            // Unique word found - atomically mark it in the global hash map
            // This prevents duplicate counting across batches
            atomic_or(global_map_ptr, check_bit);
            
            // Update uniqueness counter (WRITABLE buffer)
            atomic_inc(&rule_uniqueness_counts[rule_batch_idx]);
            
            // Check cracked hash map (effectiveness)
            unsigned int cracked_map_index = (word_hash >> 5) & cracked_map_mask;
            __global const unsigned int* cracked_map_ptr = &cracked_hash_map[cracked_map_index];
            unsigned int cracked_word = *cracked_map_ptr;
            
            if (cracked_word & check_bit) {{
                // Update effectiveness counter (WRITABLE buffer)
                atomic_inc(&rule_effectiveness_counts[rule_batch_idx]);
            }}
        }}
    }}
}}
"""
    return kernel_source

# ====================================================================
# --- BUFFER MANAGEMENT ---
# ====================================================================

def create_optimized_buffers(context, queue, config):
    """Create OpenCL buffers optimized for memory architecture"""
    mf = cl.mem_flags
    
    # Calculate buffer sizes based on configuration
    words_per_batch = config['batch_size']
    max_word_len = config['max_word_len']
    max_rules_per_batch = config['max_rules_per_batch']
    rule_size_in_int = config['rule_size_in_int']
    hash_map_size = config['hash_map_size']
    
    print(f"{blue('💾')} {bold('Optimizing buffer allocation...')}")
    
    # Calculate required memory
    word_buffer_size = words_per_batch * max_word_len * np.uint8().itemsize
    hash_buffer_size = words_per_batch * np.uint32().itemsize
    rule_buffer_size = max_rules_per_batch * rule_size_in_int * np.uint32().itemsize
    counter_size = max_rules_per_batch * np.uint32().itemsize
    
    print(f"{blue('📊')} {bold('Memory allocation:')}")
    print(f"   {bold('Word buffers:')} {cyan(f'{word_buffer_size * 2 / (1024**2):.1f} MB')}")
    print(f"   {bold('Hash buffers:')} {cyan(f'{hash_buffer_size * 2 / (1024**2):.1f} MB')}")
    print(f"   {bold('Rule buffer:')} {cyan(f'{rule_buffer_size / (1024**2):.1f} MB')}")
    print(f"   {bold('Counter buffers:')} {cyan(f'{counter_size * 2 / (1024**2):.1f} MB')}")
    print(f"   {bold('Hash maps:')} {cyan(f'{hash_map_size * 2 / (1024**2):.1f} MB')}")
    
    # Create buffers
    buffers = {}
    
    # Double buffering for words and hashes
    for i in range(2):
        buffers[f'base_words_in_{i}'] = cl.Buffer(context, mf.READ_ONLY, 
                                                 word_buffer_size)
        buffers[f'base_hashes_{i}'] = cl.Buffer(context, mf.READ_ONLY,
                                               hash_buffer_size)
    
    # Rule buffer (READ_ONLY - rules don't change during kernel execution)
    buffers['rules_in'] = cl.Buffer(context, mf.READ_ONLY, rule_buffer_size)
    
    # Counters (READ_WRITE - modified by kernel with atomic operations)
    buffers['rule_uniqueness_counts'] = cl.Buffer(context, mf.READ_WRITE, counter_size)
    buffers['rule_effectiveness_counts'] = cl.Buffer(context, mf.READ_WRITE, counter_size)
    
    # Hash maps
    buffers['global_hash_map'] = cl.Buffer(context, mf.READ_WRITE, hash_map_size)  # Modified by kernel
    buffers['cracked_hash_map'] = cl.Buffer(context, mf.READ_ONLY, hash_map_size)  # Read-only after init
    
    print(f"{green('✅')} {bold('Buffers created successfully')}")
    return buffers

# ====================================================================
# --- MAIN OPTIMIZED PIPELINE ---
# ====================================================================

def load_rules(path):
    """Load Hashcat rules from file"""
    print(f"{blue('📊')} {bold('Loading rules from:')} {path}")
    
    rules_list = []
    rule_id_counter = 0
    
    try:
        with open(path, 'r', encoding='latin-1') as f:
            for line in f:
                rule = line.strip()
                if not rule or rule.startswith('#'):
                    continue
                rules_list.append({
                    'rule_data': rule,
                    'rule_id': rule_id_counter,
                    'uniqueness_score': 0,
                    'effectiveness_score': 0
                })
                rule_id_counter += 1
    except FileNotFoundError:
        print(f"{red('❌')} {bold('Rules file not found:')} {path}")
        exit(1)
    
    print(f"{green('✅')} {bold('Loaded')} {cyan(f'{len(rules_list):,}')} {bold('rules')}")
    return rules_list

def load_cracked_hashes(path, max_len=32):
    """Load cracked password hashes"""
    print(f"{blue('📊')} {bold('Loading cracked hashes from:')} {path}")
    
    cracked_hashes = []
    
    try:
        with open(path, 'rb') as f:
            for line in f:
                line = line.strip()
                if 1 <= len(line) <= max_len:
                    # Fast FNV-1a hash
                    hash_val = 2166136261
                    for byte in line:
                        hash_val = (hash_val ^ byte) * 16777619 & 0xFFFFFFFF
                    cracked_hashes.append(hash_val)
    except FileNotFoundError:
        print(f"{yellow('⚠️')} {bold('Cracked file not found:')} {path}")
        return np.array([], dtype=np.uint32)
    
    unique_hashes = np.unique(np.array(cracked_hashes, dtype=np.uint32))
    print(f"{green('✅')} {bold('Loaded')} {cyan(f'{len(unique_hashes):,}')} {bold('unique hashes')}")
    return unique_hashes

def save_ranking_data(rules_list, output_path):
    """Save ranking results to CSV"""
    print(f"{blue('💾')} {bold('Saving ranking data to:')} {output_path}")
    
    # Calculate combined scores
    for rule in rules_list:
        rule['combined_score'] = rule['effectiveness_score'] * 10 + rule['uniqueness_score']
    
    # Sort by combined score
    ranked_rules = sorted(rules_list, key=lambda x: x['combined_score'], reverse=True)
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Rank', 'Combined_Score', 'Effectiveness_Score', 
                'Uniqueness_Score', 'Rule_Data'
            ])
            writer.writeheader()
            
            for rank, rule in enumerate(ranked_rules, 1):
                writer.writerow({
                    'Rank': rank,
                    'Combined_Score': rule['combined_score'],
                    'Effectiveness_Score': rule['effectiveness_score'],
                    'Uniqueness_Score': rule['uniqueness_score'],
                    'Rule_Data': rule['rule_data']
                })
        
        print(f"{green('✅')} {bold('Saved')} {cyan(f'{len(ranked_rules):,}')} {bold('rules')}")
        return output_path
        
    except Exception as e:
        print(f"{red('❌')} {bold('Error saving:')} {e}")
        return None

def run_optimized_pipeline(wordlist_path, rules_path, cracked_path, 
                          output_path, top_k=1000,
                          batch_size=None, global_bits=None, cracked_bits=None,
                          preset=None, target_device=None, 
                          force_nvidia=False, force_amd=False,
                          skip_rule_encoding=False, disable_double_buffering=False,
                          disable_progress_bars=False, benchmark_mode=False):
    """
    Main optimized pipeline for GPU rule ranking with memory mapping
    """
    print(f"{green('=' * 70)}")
    print(f"{bold('⚡ ULTRA-OPTIMIZED RULE RANKING PIPELINE ')}")
    print(f"{green('=' * 70)}{Colors.END}")
    
    # Show command line flags
    if benchmark_mode:
        print(f"{blue('⚡')} {bold('Benchmark mode enabled')}")
    if skip_rule_encoding:
        print(f"{blue('⚡')} {bold('Rule encoding optimization disabled')}")
    if disable_double_buffering:
        print(f"{blue('⚡')} {bold('Double buffering disabled')}")
    if disable_progress_bars:
        print(f"{blue('⚡')} {bold('Progress bars disabled')}")
    
    start_time = time()
    
    # 1. Load data
    print(f"{blue('📥')} {bold('Loading data...')}")
    rules_list = load_rules(rules_path)
    cracked_hashes = load_cracked_hashes(cracked_path)
    
    # 2. Initialize GPU
    print(f"{blue('🎮')} {bold('Initializing GPU ...')}")
    platform, device, context, queue = initialize_gpu_environment(
        target_device=target_device,
        force_nvidia=force_nvidia,
        force_amd=force_amd
    )
    
    # Get GPU memory info for parameter calculation
    total_vram, available_vram = get_gpu_memory_info(device)
    
    # 3. Configuration
    if preset:
        if preset in PERFORMANCE_PROFILES:
            config_profile = PERFORMANCE_PROFILES[preset]
            print(f"{green('🎯')} {bold('Using preset:')} {cyan(preset)} - {config_profile['description']}")
            batch_size = config_profile['batch_size']
            global_bits = config_profile['global_bits']
            cracked_bits = config_profile['cracked_bits']
        else:
            print(f"{red('❌')} {bold('Unknown preset:')} {cyan(preset)}")
            get_performance_profile_info()
            return 0
    
    # Get wordlist info using memory mapping
    print(f"{blue('🗺️')}  {bold('Analyzing wordlist...')}")
    with MemoryMappedWordlist(wordlist_path, MAX_WORD_LEN, batch_size or 100000) as wordlist:
        total_words = wordlist.get_word_count()
    
    # Auto-calculate parameters if not specified
    if batch_size is None or global_bits is None or cracked_bits is None:
        batch_size, global_bits, cracked_bits = calculate_optimal_parameters(
            available_vram, total_words, len(cracked_hashes), len(rules_list)
        )
    
    # Create final config
    config = {
        'batch_size': batch_size,
        'max_word_len': MAX_WORD_LEN,
        'max_output_len': MAX_OUTPUT_LEN,
        'max_rule_args': MAX_RULE_ARGS,
        'max_rules_per_batch': MAX_RULES_IN_BATCH,
        'local_work_size': LOCAL_WORK_SIZE,
        'rule_size_in_int': 2 + MAX_RULE_ARGS,
        'hash_map_size': 1 << 28,  # 256MB hash map
        'global_hash_map_mask': (1 << (global_bits - 5)) - 1,
        'cracked_hash_map_mask': (1 << (cracked_bits - 5)) - 1,
        
        # Rule ID constants
        'start_id_simple': START_ID_SIMPLE,
        'num_simple_rules': NUM_SIMPLE_RULES,
        'start_id_td': START_ID_TD,
        'num_td_rules': NUM_TD_RULES,
        'start_id_s': START_ID_S,
        'num_s_rules': NUM_S_RULES,
        'start_id_a': START_ID_A,
        'num_a_rules': NUM_A_RULES,
        'start_id_groupb': START_ID_GROUPB,
        'num_groupb_rules': NUM_GROUPB_RULES,
        'start_id_new': START_ID_NEW,
        'num_new_rules': NUM_NEW_RULES
    }
    
    print(f"{blue('📊')} {bold('Final configuration:')}")
    print(f"   {bold('Wordlist size:')} {cyan(f'{total_words:,}')} words")
    print(f"   {bold('Batch size:')} {cyan(f'{batch_size:,}')}")
    print(f"   {bold('Global hash map:')} {cyan(f'{global_bits} bits')}")
    print(f"   {bold('Cracked hash map:')} {cyan(f'{cracked_bits} bits')}")
    
    # 4. Precompute rule batches
    if not skip_rule_encoding:
        print(f"{blue('⚡')} {bold('Preparing rule data...')}")
        rule_batches, encoded_rule_batches = precompute_rule_batches(
            rules_list, 
            batch_size=config['max_rules_per_batch'],
            num_threads=4
        )
    else:
        print(f"{yellow('⚠️')} {bold('Skipping rule encoding optimization')}")
        rule_batches = create_rule_batches(rules_list, config['max_rules_per_batch'])
        encoded_rule_batches = []
        for start_idx, end_idx in rule_batches:
            encoded_rule_batches.append(encode_rules_batch(rules_list, start_idx, end_idx))
    
    # 5. Create buffers
    print(f"{blue('💾')} {bold('Allocating GPU memory...')}")
    buffers = create_optimized_buffers(context, queue, config)
    
    # 6. Compile optimized kernel
    print(f"{blue('🔧')} {bold('Compiling optimized kernel...')}")
    kernel_source = get_optimized_kernel_source(config)
    
    # Compile with NVIDIA-specific optimizations
    compile_options = [
        "-cl-fast-relaxed-math",
        "-cl-mad-enable",
        "-cl-no-signed-zeros",
        "-cl-unsafe-math-optimizations",
        "-cl-opt-disable",
        "-w"
    ]
    
    try:
        program = cl.Program(context, kernel_source).build(options=compile_options)
        kernel = program.bfs_kernel_optimized
        print(f"{green('✅')} {bold('Kernel compiled successfully')}")
    except cl.CompileError as e:
        print(f"{red('❌')} {bold('Kernel compilation error:')}")
        print(e)
        return 0
    
    # 7. Initialize hash maps
    print(f"{blue('🗺️')}  {bold('Initializing hash maps...')}")
    
    # Initialize global hash map
    global_map_size = config['hash_map_size'] // np.uint32().itemsize
    global_map_np = np.zeros(global_map_size, dtype=np.uint32)
    
    # Initialize cracked hash map
    if len(cracked_hashes) > 0:
        cracked_map_np = np.zeros(global_map_size, dtype=np.uint32)
        
        # Populate cracked hash map
        for hash_val in cracked_hashes:
            map_index = (hash_val >> 5) & config['cracked_hash_map_mask']
            bit_index = hash_val & 31
            cracked_map_np[map_index] |= (1 << bit_index)
        
        cl.enqueue_copy(queue, buffers['cracked_hash_map'], cracked_map_np).wait()
        print(f"{green('✅')} {bold('Cracked hash map populated:')} {cyan(f'{len(cracked_hashes):,}')} hashes")
    else:
        print(f"{yellow('⚠️')}  {bold('No cracked hashes loaded')}")
    
    # 8. Main processing loop with memory mapping
    print(f"{green('🚀')} {bold('Starting processing pipeline...')}")
    
    performance_stats = {
        'words_processed': 0,
        'unique_found': 0,
        'cracked_found': 0,
        'start_time': time(),
        'batch_speeds': [],
        'total_copy_time': 0.0,
        'total_kernel_time': 0.0
    }
    
    # Setup progress tracking
    total_rule_batches = len(rule_batches)
    total_word_batches = (total_words + batch_size - 1) // batch_size
    total_iterations = total_rule_batches * total_word_batches
    
    print(f"{blue('📊')} {bold('Processing plan:')}")
    print(f"   {bold('Rule batches:')} {cyan(f'{total_rule_batches:,}')}")
    print(f"   {bold('Word batches:')} {cyan(f'{total_word_batches:,}')}")
    print(f"   {bold('Total iterations:')} {cyan(f'{total_iterations:,}')}")
    
    # Initialize global hash map
    cl.enqueue_copy(queue, buffers['global_hash_map'], global_map_np).wait()
    
    # Create overall progress bar if enabled
    if not disable_progress_bars:
        overall_pbar = tqdm(total=total_rule_batches, desc="Processing rules", unit="batch", 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    else:
        overall_pbar = None
    
    # Process each rule batch with ALL word batches
    for rule_batch_idx, ((start_idx, end_idx), encoded_rules) in enumerate(zip(rule_batches, encoded_rule_batches)):
        num_rules = end_idx - start_idx
        
        # Copy rules to GPU once per rule batch
        cl.enqueue_copy(queue, buffers['rules_in'], encoded_rules).wait()
        
        # Reset counters for this rule batch
        counter_size = config['max_rules_per_batch'] * np.uint32().itemsize
        cl.enqueue_fill_buffer(queue, buffers['rule_uniqueness_counts'], 
                              np.uint32(0), 0, counter_size)
        cl.enqueue_fill_buffer(queue, buffers['rule_effectiveness_counts'],
                              np.uint32(0), 0, counter_size)
        
        # Create memory-mapped wordlist for this rule batch
        wordlist_gen = create_wordlist_batches(
            wordlist_path,
            max_word_len=config['max_word_len'],
            batch_size=config['batch_size']
        )
        
        # Process this rule batch against ALL word batches
        word_batch_idx = 0
        for words_np, hashes_np, word_count in wordlist_gen:
            # Copy word data to GPU
            if disable_double_buffering:
                # Use single buffer if double buffering disabled
                words_buffer = buffers['base_words_in_0']
                hashes_buffer = buffers['base_hashes_0']
            else:
                # Use double buffering for better performance
                words_buffer = buffers[f'base_words_in_{word_batch_idx % 2}']
                hashes_buffer = buffers[f'base_hashes_{word_batch_idx % 2}']
            
            copy_start = time()
            cl.enqueue_copy(queue, words_buffer, words_np[:word_count * config['max_word_len']])
            cl.enqueue_copy(queue, hashes_buffer, hashes_np[:word_count]).wait()
            copy_time = time() - copy_start
            
            # Execute kernel
            global_size = word_count * num_rules
            global_size_aligned = ((global_size + config['local_work_size'] - 1) // 
                                  config['local_work_size']) * config['local_work_size']
            
            kernel.set_args(
                words_buffer,
                buffers['rules_in'],
                buffers['rule_uniqueness_counts'],
                buffers['rule_effectiveness_counts'],
                buffers['global_hash_map'],
                buffers['cracked_hash_map'],
                np.uint32(word_count),
                np.uint32(num_rules),
                np.uint32(config['max_word_len']),
                np.uint32(config['max_output_len']),
                np.uint32(config['global_hash_map_mask']),
                np.uint32(config['cracked_hash_map_mask'])
            )
            
            kernel_start = time()
            event = cl.enqueue_nd_range_kernel(queue, kernel, 
                                              (global_size_aligned,), 
                                              (config['local_work_size'],))
            event.wait()
            kernel_time = time() - kernel_start
            
            # Update performance stats
            performance_stats['words_processed'] += word_count
            performance_stats['total_copy_time'] += copy_time
            performance_stats['total_kernel_time'] += kernel_time
            
            total_time = copy_time + kernel_time
            words_per_sec = word_count / total_time if total_time > 0 else 0
            performance_stats['batch_speeds'].append(words_per_sec)
            
            word_batch_idx += 1
            
            # Update overall progress - SIMPLIFIED DESCRIPTION
            if overall_pbar:
                avg_speed = sum(performance_stats['batch_speeds'][-5:]) / min(5, len(performance_stats['batch_speeds']))
                # Simplified description without word count
                overall_pbar.set_description(f"Batch {rule_batch_idx+1}/{total_rule_batches} | {avg_speed:,.0f} w/s")
        
        # After processing ALL word batches for this rule batch, read results
        uniqueness_counts = np.zeros(num_rules, dtype=np.uint32)
        effectiveness_counts = np.zeros(num_rules, dtype=np.uint32)
        
        cl.enqueue_copy(queue, uniqueness_counts, buffers['rule_uniqueness_counts'])
        cl.enqueue_copy(queue, effectiveness_counts, buffers['rule_effectiveness_counts']).wait()
        
        # Update rule scores
        for i in range(num_rules):
            rule_idx = start_idx + i
            rules_list[rule_idx]['uniqueness_score'] += int(uniqueness_counts[i])
            rules_list[rule_idx]['effectiveness_score'] += int(effectiveness_counts[i])
        
        # Calculate batch statistics
        batch_unique = int(uniqueness_counts.sum())
        batch_cracked = int(effectiveness_counts.sum())
        
        # Update performance stats
        performance_stats['unique_found'] += batch_unique
        performance_stats['cracked_found'] += batch_cracked
        
        # Update overall progress bar - NO POSTFIX
        if overall_pbar:
            overall_pbar.update(1)
        
        # Show batch completion message
        if not disable_progress_bars:
            print(f"{green('✅')} Rule batch {rule_batch_idx+1}: +{batch_unique:,} unique, +{batch_cracked:,} cracked")
    
    if overall_pbar:
        overall_pbar.close()
    
    # 9. Final results
    total_time = time() - start_time
    total_words_processed = performance_stats['words_processed']
    avg_speed = total_words_processed / total_time if total_time > 0 else 0
    
    print(f"\n{green('=' * 60)}")
    print(f"{bold('🎉 OPTIMIZED PIPELINE COMPLETE')}")
    print(f"{green('=' * 60)}")
    print(f"{blue('📊')} {bold('Performance Summary:')}")
    print(f"   {bold('Total Rule Applications:')} {cyan(f'{total_words_processed:,}')}")
    print(f"   {bold('Unique Rule Applications:')} {cyan(f'{total_words:,} words × {len(rules_list):,} rules = {total_words * len(rules_list):,}')}")
    print(f"   {bold('Total Rules Tested:')} {cyan(f'{len(rules_list):,}')}")
    print(f"   {bold('Total Time:')} {cyan(f'{total_time:.2f}s')}")
    print(f"   {bold('Average Speed:')} {cyan(f'{avg_speed:,.0f}')} rule applications/sec")
    
    if benchmark_mode:
        print(f"   {bold('Performance Metrics:')}")
        # Use the accumulated times
        total_execution_time = performance_stats['total_copy_time'] + performance_stats['total_kernel_time']
        if total_execution_time > 0:
            copy_ratio = performance_stats['total_copy_time']/total_execution_time*100
            gpu_util = performance_stats['total_kernel_time']/total_execution_time*100
        else:
            copy_ratio = 0
            gpu_util = 0
            
        print(f"      {bold('Copy Time/Execution Time Ratio:')} {cyan(f'{copy_ratio:.1f}%')}")
        print(f"      {bold('GPU Utilization:')} {cyan(f'{gpu_util:.1f}%')}")
        if performance_stats['batch_speeds']:
            max_speed = max(performance_stats['batch_speeds'])
            avg_batch_speed = sum(performance_stats['batch_speeds'])/len(performance_stats['batch_speeds'])
            print(f"      {bold('Peak Speed:')} {cyan(f'{max_speed:,.0f}')} words/sec")
            print(f"      {bold('Average Batch Speed:')} {cyan(f'{avg_batch_speed:,.0f}')} words/sec")
    
    # Extract values before formatting
    unique_found = performance_stats['unique_found']
    cracked_found = performance_stats['cracked_found']
    
    print(f"   {bold('Unique Words Found:')} {cyan(f'{unique_found:,}')}")
    print(f"   {bold('Cracked Passwords Found:')} {cyan(f'{cracked_found:,}')}")
    
    # Safe calculation for unique ratio
    if total_words_processed > 0:
        unique_ratio = unique_found/total_words_processed*100
    else:
        unique_ratio = 0
    print(f"   {bold('Unique Ratio:')} {cyan(f'{unique_ratio:.2f}%')} of all generated words were unique")
    
    print(f"{green('=' * 60)}{Colors.END}")
    
    # Save results
    csv_path = save_ranking_data(rules_list, output_path)
    
    # Save optimized rules if requested
    if top_k > 0 and csv_path:
        optimized_path = os.path.splitext(output_path)[0] + "_optimized.rule"
        print(f"{blue('💾')} {bold('Saving top')} {cyan(str(top_k))} {bold('rules to:')} {optimized_path}")
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rules = list(reader)
            
            rules.sort(key=lambda x: int(x['Combined_Score']), reverse=True)
            
            with open(optimized_path, 'w') as f:
                f.write(":\n")
                for rule in rules[:top_k]:
                    f.write(f"{rule['Rule_Data']}\n")
            
            print(f"{green('✅')} {bold('Saved')} {cyan(f'{min(top_k, len(rules)):,}')} {bold('optimized rules')}")
        except Exception as e:
            print(f"{red('❌')} {bold('Error saving optimized rules:')} {e}")
    
    return avg_speed

# ====================================================================
# --- MAIN FUNCTION ---
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ultra-optimized GPU rule ranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -w wordlist.txt -r rules.rule -c cracked.txt -o output.csv
  %(prog)s -w wordlist.txt -r rules.rule -c cracked.txt -o output.csv -k 5000
  %(prog)s -w wordlist.txt -r rules.rule -c cracked.txt -o output.csv --preset aggressive
  %(prog)s -w wordlist.txt -r rules.rule -c cracked.txt -o output.csv --batch-size 200000 --global-bits 34
        """
    )
    
    # Required arguments
    parser.add_argument('-w', '--wordlist', required=True, 
                       help='Path to the base wordlist file')
    parser.add_argument('-r', '--rules', required=True,
                       help='Path to the Hashcat rules file to rank')
    parser.add_argument('-c', '--cracked', required=True,
                       help='Path to a list of cracked passwords for effectiveness scoring')
    parser.add_argument('-o', '--output', default='ranker_output.csv',
                       help='Path to save the final ranking CSV (default: ranker_output.csv)')
    
    # Optional arguments
    parser.add_argument('-k', '--topk', type=int, default=1000,
                       help='Number of top rules to save to optimized .rule file (default: 1000)')
    
    # Performance tuning flags
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Number of words to process in each GPU batch (default: auto-calculate)')
    parser.add_argument('--global-bits', type=int, default=None,
                       help='Bits for global hash map size (default: auto-calculate)')
    parser.add_argument('--cracked-bits', type=int, default=None,
                       help='Bits for cracked hash map size (default: auto-calculate)')
    
    # Preset configuration
    parser.add_argument('--preset', type=str, default=None,
                       choices=['low_memory', 'medium_memory', 'high_memory', 'aggressive', 'balanced'],
                       help='Use preset configuration for easy setup')
    
    # GPU selection flags
    parser.add_argument('--target-device', type=str, default=None,
                       help='Target specific GPU device (e.g., "RTX 3060 Ti")')
    parser.add_argument('--force-nvidia', action='store_true',
                       help='Force use of NVIDIA platform')
    parser.add_argument('--force-amd', action='store_true',
                       help='Force use of AMD platform')
    
    # Optimization flags
    parser.add_argument('--skip-rule-encoding', action='store_true',
                       help='Skip parallel rule encoding optimization')
    parser.add_argument('--disable-double-buffering', action='store_true',
                       help='Disable double buffering for word loading')
    parser.add_argument('--disable-progress-bars', action='store_true',
                       help='Disable all progress bars for cleaner output')
    parser.add_argument('--benchmark-mode', action='store_true',
                       help='Enable detailed performance benchmarking')
    
    # List presets flag
    parser.add_argument('--list-presets', action='store_true',
                       help='List available performance presets and exit')
    
    args = parser.parse_args()
    
    # Handle list presets flag
    if args.list_presets:
        print(f"{green('=' * 60)}")
        print(f"{bold('📊 Available Performance Presets')}")
        print(f"{green('=' * 60)}")
        get_performance_profile_info()
        print(f"{green('=' * 60)}")
        return
    
    # Validate conflicting flags
    if args.force_nvidia and args.force_amd:
        print(f"{red('❌')} {bold('Cannot force both NVIDIA and AMD platforms')}")
        return
    
    # Run the optimized pipeline
    speed = run_optimized_pipeline(
        wordlist_path=args.wordlist,
        rules_path=args.rules,
        cracked_path=args.cracked,
        output_path=args.output,
        top_k=args.topk,
        batch_size=args.batch_size,
        global_bits=args.global_bits,
        cracked_bits=args.cracked_bits,
        preset=args.preset,
        target_device=args.target_device,
        force_nvidia=args.force_nvidia,
        force_amd=args.force_amd,
        skip_rule_encoding=args.skip_rule_encoding,
        disable_double_buffering=args.disable_double_buffering,
        disable_progress_bars=args.disable_progress_bars,
        benchmark_mode=args.benchmark_mode
    )
    
    print(f"\n{green('✅')} {bold('Processing complete!')}")
    print(f"   {bold('Final speed:')} {cyan(f'{speed:,.0f}')} rule applications/sec")

if __name__ == '__main__':
    main()
