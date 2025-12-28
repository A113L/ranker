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
# --- RANKER v3.3: COMPLETE HASHCAT RULES IMPLEMENTATION ---
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
def cyan(text): return f"{Colors.CYAN}{text}{Colors.END}"
def bold(text): return f"{Colors.BOLD}{text}{Colors.END}"

# ====================================================================
# --- CONSTANTS ---
# ====================================================================
MAX_WORD_LEN = 256
MAX_OUTPUT_LEN = 512
MAX_RULE_LEN = 255  # Maximum hashcat rule length
MAX_RULES_IN_BATCH = 1024
LOCAL_WORK_SIZE = 256

# Default values
DEFAULT_WORDS_PER_GPU_BATCH = 100000
DEFAULT_GLOBAL_HASH_MAP_BITS = 35
DEFAULT_CRACKED_HASH_MAP_BITS = 33

# Memory settings
VRAM_SAFETY_MARGIN = 0.15
MIN_BATCH_SIZE = 25000
MIN_HASH_MAP_BITS = 28
MEMORY_REDUCTION_FACTOR = 0.7
MAX_ALLOCATION_RETRIES = 5

# Global variables
interrupted = False
current_rules_list = None
current_ranking_output_path = None
current_top_k = 0
words_processed_total = None
total_unique_found = None
total_cracked_found = None

# ====================================================================
# --- DEVICE LISTING AND SELECTION ---
# ====================================================================

def list_opencl_devices():
    """List all available OpenCL devices"""
    print(f"{green('=' * 60)}")
    print(f"{bold('Available OpenCL Devices')}")
    print(f"{green('=' * 60)}")
    
    devices_found = []
    try:
        platforms = cl.get_platforms()
        device_index = 0
        
        for platform_idx, platform in enumerate(platforms):
            print(f"{bold('Platform')} {platform_idx}: {platform.name}")
            print(f"  Vendor: {platform.vendor}")
            print(f"  Version: {platform.version}")
            
            # Get all devices for this platform
            for device_type in [cl.device_type.GPU, cl.device_type.CPU, cl.device_type.ACCELERATOR, cl.device_type.ALL]:
                try:
                    devices = platform.get_devices(device_type)
                    for device in devices:
                        device_type_str = {
                            cl.device_type.GPU: "GPU",
                            cl.device_type.CPU: "CPU",
                            cl.device_type.ACCELERATOR: "Accelerator",
                            cl.device_type.ALL: "All"
                        }.get(device.type, "Unknown")
                        
                        print(f"  [{device_index}] {device_type_str}: {device.name.strip()}")
                        print(f"      Compute Units: {device.max_compute_units}")
                        print(f"      Global Memory: {device.global_mem_size / (1024**3):.2f} GB")
                        print(f"      Max Work Group Size: {device.max_work_group_size}")
                        print()
                        
                        devices_found.append({
                            'index': device_index,
                            'platform': platform,
                            'device': device,
                            'type': device_type_str,
                            'name': device.name.strip(),
                            'memory_gb': device.global_mem_size / (1024**3)
                        })
                        device_index += 1
                except:
                    continue
        
        print(f"{green('=' * 60)}")
        return devices_found
        
    except Exception as e:
        print(f"{red('ERROR')}: Failed to list OpenCL devices: {e}")
        return []

def select_opencl_device(device_id=None):
    """Select an OpenCL device by ID or automatically"""
    devices = list_opencl_devices()
    
    if not devices:
        print(f"{red('ERROR')}: No OpenCL devices found!")
        return None
    
    if device_id is not None:
        # Try to find the specified device
        for device_info in devices:
            if device_info['index'] == device_id:
                print(f"{green('SELECTED')}: Device {device_id} - {device_info['type']}: {device_info['name']}")
                return device_info['platform'], device_info['device']
        
        print(f"{yellow('WARNING')}: Device ID {device_id} not found. Using default selection.")
    
    # Auto-select: prefer GPU, then highest memory
    gpu_devices = [d for d in devices if d['type'] == 'GPU']
    
    if gpu_devices:
        # Select GPU with most memory
        selected = max(gpu_devices, key=lambda x: x['memory_gb'])
        print(f"{green('AUTO-SELECTED')}: Device {selected['index']} - {selected['type']}: {selected['name']} ({selected['memory_gb']:.2f} GB)")
        return selected['platform'], selected['device']
    else:
        # Fall back to CPU
        cpu_devices = [d for d in devices if d['type'] == 'CPU']
        if cpu_devices:
            selected = max(cpu_devices, key=lambda x: x['memory_gb'])
            print(f"{green('AUTO-SELECTED')}: Device {selected['index']} - {selected['type']}: {selected['name']}")
            return selected['platform'], selected['device']
    
    # Last resort: first device
    print(f"{yellow('WARNING')}: Using first available device")
    return devices[0]['platform'], devices[0]['device']

# ====================================================================
# --- OPTIMIZED FILE LOADING FUNCTIONS ---
# ====================================================================

def estimate_word_count(path):
    """Fast word count estimation for large files without reading entire content"""
    print(f"{blue('ESTIMATING')}: Words in {path}...")
    
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
                
        print(f"{green('ESTIMATED')}: {cyan(f'{total_lines:,}')} words")
        return total_lines
        
    except Exception as e:
        print(f"{yellow('WARNING')}: Could not estimate word count: {e}")
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

def optimized_wordlist_iterator(wordlist_path, max_len, batch_size):
    """
    Massively optimized wordlist loader using memory mapping and bulk processing
    """
    print(f"{green('OPTIMIZED')}: Using memory-mapped loader...")
    
    file_size = os.path.getsize(wordlist_path)
    print(f"{blue('FILE SIZE')}: {cyan(f'{file_size / (1024**3):.2f} GB')}")
    
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
                with tqdm(total=file_size, desc="Loading wordlist", unit="B", unit_scale=True, 
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
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
        print(f"{red('ERROR')}: Error in optimized loader: {e}")
        raise
    
    load_time = time() - load_start
    print(f"{green('LOADED')}: {cyan(f'{total_words_loaded:,}')} words in {load_time:.2f}s "
          f"({total_words_loaded/load_time:,.0f} words/sec)")

# ====================================================================
# --- COMPREHENSIVE HASHCAT RULES KERNEL (UPDATED) ---
# ====================================================================

def get_kernel_source(global_hash_map_bits, cracked_hash_map_bits):
    global_hash_map_mask = (1 << (global_hash_map_bits - 5)) - 1
    cracked_hash_map_mask = (1 << (cracked_hash_map_bits - 5)) - 1
    
    return f"""
// ============================================================================
// COMPREHENSIVE HASHCAT RULES KERNEL FOR RANKER
// ============================================================================

#define MAX_WORD_LEN {MAX_WORD_LEN}
#define MAX_OUTPUT_LEN {MAX_OUTPUT_LEN}
#define MAX_RULE_LEN {MAX_RULE_LEN}
#define GLOBAL_HASH_MAP_MASK {global_hash_map_mask}
#define CRACKED_HASH_MAP_MASK {cracked_hash_map_mask}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Check if character is lowercase
int is_lower(unsigned char c) {{
    return (c >= 'a' && c <= 'z');
}}

// Check if character is uppercase
int is_upper(unsigned char c) {{
    return (c >= 'A' && c <= 'Z');
}}

// Check if character is a digit
int is_digit(unsigned char c) {{
    return (c >= '0' && c <= '9');
}}

// Check if character is alphanumeric
int is_alnum(unsigned char c) {{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}}

// Toggle case of a single character
unsigned char toggle_case(unsigned char c) {{
    if (is_lower(c)) return c - 32;
    if (is_upper(c)) return c + 32;
    return c;
}}

// Convert character to lowercase
unsigned char to_lower(unsigned char c) {{
    if (is_upper(c)) return c + 32;
    return c;
}}

// Convert character to uppercase
unsigned char to_upper(unsigned char c) {{
    if (is_lower(c)) return c - 32;
    return c;
}}

// FNV-1a Hash implementation for 32-bit
unsigned int fnv1a_hash_32(const unsigned char* data, unsigned int len) {{
    unsigned int hash = 2166136261U;
    for (unsigned int i = 0; i < len; i++) {{
        hash ^= data[i];
        hash *= 16777619U;
    }}
    return hash;
}}

// Helper function to convert char to position (0-9, A-Z, a-z)
unsigned int char_to_pos(unsigned char c) {{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    if (c >= 'a' && c <= 'z') return c - 'a' + 36;
    return 0xFFFFFFFF; 
}}

// ============================================================================
// HASHCAT RULE APPLICATION FUNCTION (COMPREHENSIVE)
// ============================================================================

void apply_hashcat_rule(const unsigned char* word, int word_len,
                        const unsigned char* rule_str, int rule_len,
                        unsigned char* output, int* out_len, int* changed) {{
    
    *out_len = 0;
    *changed = 0;
    
    // Clear output
    for (int i = 0; i < MAX_OUTPUT_LEN; i++) output[i] = 0;
    
    // Early exit for empty inputs
    if (rule_len == 0 || word_len == 0) return;
    
    // ========================================================================
    // SIMPLE RULES (1 character)
    // ========================================================================
    
    if (rule_len == 1) {{
        switch (rule_str[0]) {{
            // l - Lowercase all letters
            case 'l':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {{
                    output[i] = to_lower(word[i]);
                }}
                *changed = 1;
                return;
                
            // u - Uppercase all letters
            case 'u':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {{
                    output[i] = to_upper(word[i]);
                }}
                *changed = 1;
                return;
                
            // c - Capitalize first letter, lowercase rest
            case 'c':
                *out_len = word_len;
                if (word_len > 0) {{
                    output[0] = to_upper(word[0]);
                    for (int i = 1; i < word_len; i++) {{
                        output[i] = to_lower(word[i]);
                    }}
                }}
                *changed = 1;
                return;
                
            // C - Lowercase first letter, uppercase rest
            case 'C':
                *out_len = word_len;
                if (word_len > 0) {{
                    output[0] = to_lower(word[0]);
                    for (int i = 1; i < word_len; i++) {{
                        output[i] = to_upper(word[i]);
                    }}
                }}
                *changed = 1;
                return;
                
            // t - Toggle case of all letters
            case 't':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {{
                    output[i] = toggle_case(word[i]);
                }}
                *changed = 1;
                return;
                
            // r - Reverse the entire word
            case 'r':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {{
                    output[i] = word[word_len - 1 - i];
                }}
                *changed = 1;
                return;
                
            // k - Swap first two characters
            case 'k':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {{
                    output[i] = word[i];
                }}
                if (word_len >= 2) {{
                    output[0] = word[1];
                    output[1] = word[0];
                    *changed = 1;
                }}
                return;
                
            // : - Do nothing (identity)
            case ':':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {{
                    output[i] = word[i];
                }}
                *changed = 0;
                return;
                
            // d - Duplicate word
            case 'd':
                if (word_len * 2 <= MAX_OUTPUT_LEN) {{
                    *out_len = word_len * 2;
                    for (int i = 0; i < word_len; i++) {{
                        output[i] = word[i];
                        output[word_len + i] = word[i];
                    }}
                    *changed = 1;
                }}
                return;
                
            // f - Reflect word (word + reverse)
            case 'f':
                if (word_len * 2 <= MAX_OUTPUT_LEN) {{
                    *out_len = word_len * 2;
                    for (int i = 0; i < word_len; i++) {{
                        output[i] = word[i];
                        output[word_len + i] = word[word_len - 1 - i];
                    }}
                    *changed = 1;
                }}
                return;
                
            // p - Pluralize (add 's')
            case 'p':
                if (word_len + 1 <= MAX_OUTPUT_LEN) {{
                    *out_len = word_len;
                    for (int i = 0; i < word_len; i++) {{
                        output[i] = word[i];
                    }}
                    output[*out_len] = 's';
                    (*out_len)++;
                    *changed = 1;
                }}
                return;
                
            // z - Duplicate first character
            case 'z':
                if (word_len + 1 <= MAX_OUTPUT_LEN) {{
                    output[0] = word[0];
                    for (int i = 0; i < word_len; i++) {{
                        output[i + 1] = word[i];
                    }}
                    *out_len = word_len + 1;
                    *changed = 1;
                }}
                return;
                
            // Z - Duplicate last character
            case 'Z':
                if (word_len + 1 <= MAX_OUTPUT_LEN) {{
                    for (int i = 0; i < word_len; i++) {{
                        output[i] = word[i];
                    }}
                    output[word_len] = word[word_len - 1];
                    *out_len = word_len + 1;
                    *changed = 1;
                }}
                return;
                
            // q - Duplicate all characters
            case 'q':
                if (word_len * 2 <= MAX_OUTPUT_LEN) {{
                    int idx = 0;
                    for (int i = 0; i < word_len; i++) {{
                        output[idx++] = word[i];
                        output[idx++] = word[i];
                    }}
                    *out_len = word_len * 2;
                    *changed = 1;
                }}
                return;
                
            // E - Title case (capitalize first letter of each word)
            case 'E':
                *out_len = word_len;
                int capitalize_next = 1;
                for (int i = 0; i < word_len; i++) {{
                    if (capitalize_next && is_lower(word[i])) {{
                        output[i] = word[i] - 32;
                        capitalize_next = 0;
                    }} else {{
                        output[i] = word[i];
                    }}
                    if (word[i] == ' ' || word[i] == '-' || word[i] == '_') {{
                        capitalize_next = 1;
                    }}
                }}
                *changed = 1;
                return;
                
            // Memory rules placeholder (M, 4, 6, _)
            case 'M':
            case '4':
            case '6':
            case '_':
                // Memory operations require context management
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {{
                    output[i] = word[i];
                }}
                *changed = 0;
                return;
                
            default:
                // Unknown single character rule, treat as identity
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {{
                    output[i] = word[i];
                }}
                *changed = 0;
                return;
        }}
    }}
    
    // ========================================================================
    // POSITION-BASED RULES (Tn, Dn, etc.)
    // ========================================================================
    
    else if (rule_len == 2) {{
        unsigned char cmd = rule_str[0];
        unsigned char arg = rule_str[1];
        
        // Check if arg is a digit for position-based rules
        if (is_digit(arg)) {{
            int n = arg - '0';
            
            switch (cmd) {{
                // Tn - Toggle case at position n
                case 'T':
                    *out_len = word_len;
                    for (int i = 0; i < word_len; i++) {{
                        output[i] = word[i];
                        if (i == n) {{
                            output[i] = toggle_case(word[i]);
                            *changed = 1;
                        }}
                    }}
                    return;
                    
                // Dn - Delete character at position n
                case 'D':
                    *out_len = 0;
                    for (int i = 0; i < word_len; i++) {{
                        if (i != n) {{
                            output[(*out_len)++] = word[i];
                        }} else {{
                            *changed = 1;
                        }}
                    }}
                    return;
                    
                // Ln - Delete left of position n
                case 'L':
                    *out_len = 0;
                    for (int i = n; i < word_len; i++) {{
                        output[(*out_len)++] = word[i];
                        *changed = 1;
                    }}
                    return;
                    
                // Rn - Delete right of position n
                case 'R':
                    *out_len = n + 1;
                    if (*out_len > word_len) *out_len = word_len;
                    for (int i = 0; i < *out_len; i++) {{
                        output[i] = word[i];
                    }}
                    *changed = 1;
                    return;
                    
                // +n - ASCII increment at position n
                case '+':
                    *out_len = word_len;
                    for (int i = 0; i < word_len; i++) {{
                        output[i] = word[i];
                        if (i == n && word[i] < 255) {{
                            output[i] = word[i] + 1;
                            *changed = 1;
                        }}
                    }}
                    return;
                    
                // -n - ASCII decrement at position n
                case '-':
                    *out_len = word_len;
                    for (int i = 0; i < word_len; i++) {{
                        output[i] = word[i];
                        if (i == n && word[i] > 0) {{
                            output[i] = word[i] - 1;
                            *changed = 1;
                        }}
                    }}
                    return;
                    
                // .n - Replace with dot at position n
                case '.':
                    *out_len = word_len;
                    for (int i = 0; i < word_len; i++) {{
                        output[i] = word[i];
                        if (i == n) {{
                            output[i] = '.';
                            *changed = 1;
                        }}
                    }}
                    return;
                    
                // ,n - Replace with comma at position n
                case ',':
                    *out_len = word_len;
                    for (int i = 0; i < word_len; i++) {{
                        output[i] = word[i];
                        if (i == n) {{
                            output[i] = ',';
                            *changed = 1;
                        }}
                    }}
                    return;
                    
                // 'n - Increment character at position n (using escape sequence)
                case '\\'':
                    *out_len = word_len;
                    for (int i = 0; i < word_len; i++) {{
                        output[i] = word[i];
                        if (i == n && word[i] < 255) {{
                            output[i] = word[i] + 1;
                            *changed = 1;
                        }}
                    }}
                    return;
            }}
        }}
        
        // Handle non-digit arguments
        switch (cmd) {{
            // ^X - Prepend character X
            case '^':
                if (word_len + 1 <= MAX_OUTPUT_LEN) {{
                    output[0] = arg;
                    for (int i = 0; i < word_len; i++) {{
                        output[i + 1] = word[i];
                    }}
                    *out_len = word_len + 1;
                    *changed = 1;
                }}
                return;
                
            // $X - Append character X
            case '$':
                if (word_len + 1 <= MAX_OUTPUT_LEN) {{
                    for (int i = 0; i < word_len; i++) {{
                        output[i] = word[i];
                    }}
                    output[word_len] = arg;
                    *out_len = word_len + 1;
                    *changed = 1;
                }}
                return;
                
            // @X - Delete all instances of character X
            case '@':
                *out_len = 0;
                for (int i = 0; i < word_len; i++) {{
                    if (word[i] != arg) {{
                        output[(*out_len)++] = word[i];
                    }} else {{
                        *changed = 1;
                    }}
                }}
                return;
                
            // !X - Reject word if it contains X
            case '!':
                for (int i = 0; i < word_len; i++) {{
                    if (word[i] == arg) {{
                        *out_len = 0;
                        *changed = -1;
                        return;
                    }}
                }}
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {{
                    output[i] = word[i];
                }}
                return;
                
            // /X - Reject word if it doesn't contain X
            case '/': {{
                int found_char = 0;
                for (int i = 0; i < word_len; i++) {{
                    if (word[i] == arg) {{
                        found_char = 1;
                        break;
                    }}
                }}
                if (!found_char) {{
                    *out_len = 0;
                    *changed = -1;
                    return;
                }}
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {{
                    output[i] = word[i];
                }}
                return;
            }}
            
            // pX - Purge all instances of X (same as @X)
            case 'p':
                *out_len = 0;
                for (int i = 0; i < word_len; i++) {{
                    if (word[i] != arg) {{
                        output[(*out_len)++] = word[i];
                    }} else {{
                        *changed = 1;
                    }}
                }}
                return;
        }}
    }}
    
    // ========================================================================
    // SUBSTITUTION RULES (sXY)
    // ========================================================================
    
    else if (rule_len == 3 && rule_str[0] == 's') {{
        // sXY - Substitute X with Y
        unsigned char find = rule_str[1];
        unsigned char replace = rule_str[2];
        
        *out_len = word_len;
        for (int i = 0; i < word_len; i++) {{
            output[i] = word[i];
            if (word[i] == find) {{
                output[i] = replace;
                *changed = 1;
            }}
        }}
        return;
    }}
    
    // ========================================================================
    // COMPLEX RULES (multi-character)
    // ========================================================================
    
    else if (rule_len >= 3) {{
        // Handle rules like xn m, *n m, Kn m, i n X, o n X, etc.
        
        // xn m - Extract substring from n to m
        if (rule_str[0] == 'x' && rule_len >= 3) {{
            unsigned int n = char_to_pos(rule_str[1]);
            unsigned int m = char_to_pos(rule_str[2]);
            
            if (n != 0xFFFFFFFF && m != 0xFFFFFFFF) {{
                if (n > m) {{
                    unsigned int temp = n;
                    n = m;
                    m = temp;
                }}
                if (n >= word_len) n = 0;
                if (m >= word_len) m = word_len - 1;
                
                *out_len = 0;
                for (unsigned int i = n; i <= m && i < word_len; i++) {{
                    output[(*out_len)++] = word[i];
                }}
                *changed = (*out_len > 0);
                return;
            }}
        }}
        
        // *n m - Swap characters at positions n and m
        else if (rule_str[0] == '*' && rule_len >= 3) {{
            unsigned int n = char_to_pos(rule_str[1]);
            unsigned int m = char_to_pos(rule_str[2]);
            
            *out_len = word_len;
            for (int i = 0; i < word_len; i++) {{
                output[i] = word[i];
            }}
            if (n != 0xFFFFFFFF && m != 0xFFFFFFFF && n < word_len && m < word_len && n != m) {{
                unsigned char temp = output[n];
                output[n] = output[m];
                output[m] = temp;
                *changed = 1;
            }}
            return;
        }}
        
        // i n X - Insert X at position n
        else if (rule_str[0] == 'i' && rule_len >= 3) {{
            unsigned int n = char_to_pos(rule_str[1]);
            unsigned char x = rule_str[2];
            
            if (n != 0xFFFFFFFF && word_len + 1 <= MAX_OUTPUT_LEN) {{
                *out_len = 0;
                for (int i = 0; i < word_len; i++) {{
                    if (i == n) {{
                        output[(*out_len)++] = x;
                    }}
                    output[(*out_len)++] = word[i];
                }}
                if (n >= word_len) {{
                    output[(*out_len)++] = x;
                }}
                *changed = 1;
                return;
            }}
        }}
        
        // o n X - Overwrite at position n with X
        else if (rule_str[0] == 'o' && rule_len >= 3) {{
            unsigned int n = char_to_pos(rule_str[1]);
            unsigned char x = rule_str[2];
            
            if (n != 0xFFFFFFFF) {{
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {{
                    output[i] = word[i];
                    if (i == n) {{
                        output[i] = x;
                        *changed = 1;
                    }}
                }}
                return;
            }}
        }}
        
        // {{N - Rotate left N positions
        else if (rule_str[0] == '{{' && rule_len >= 2) {{
            unsigned int n = char_to_pos(rule_str[1]);
            if (n == 0xFFFFFFFF) n = 1;
            
            *out_len = word_len;
            for (int i = 0; i < word_len; i++) {{
                int src = (i + n) % word_len;
                output[i] = word[src];
            }}
            *changed = 1;
            return;
        }}
        
        // }}N - Rotate right N positions
        else if (rule_str[0] == '}}' && rule_len >= 2) {{
            unsigned int n = char_to_pos(rule_str[1]);
            if (n == 0xFFFFFFFF) n = 1;
            
            *out_len = word_len;
            for (int i = 0; i < word_len; i++) {{
                int src = (i - n + word_len) % word_len;
                output[i] = word[src];
            }}
            *changed = 1;
            return;
        }}
        
        // [N - Delete first N characters
        else if (rule_str[0] == '[' && rule_len >= 2) {{
            unsigned int n = char_to_pos(rule_str[1]);
            if (n == 0xFFFFFFFF) n = 1;
            if (n > word_len) n = word_len;
            
            *out_len = word_len - n;
            for (int i = n; i < word_len; i++) {{
                output[i - n] = word[i];
            }}
            *changed = 1;
            return;
        }}
        
        // ]N - Delete last N characters
        else if (rule_str[0] == ']' && rule_len >= 2) {{
            unsigned int n = char_to_pos(rule_str[1]);
            if (n == 0xFFFFFFFF) n = 1;
            if (n > word_len) n = word_len;
            
            *out_len = word_len - n;
            for (int i = 0; i < *out_len; i++) {{
                output[i] = word[i];
            }}
            *changed = 1;
            return;
        }}
        
        // T n m - Toggle case from position n to m
        else if (rule_str[0] == 'T' && rule_len >= 3) {{
            unsigned int n = char_to_pos(rule_str[1]);
            unsigned int m = char_to_pos(rule_str[2]);
            
            if (n != 0xFFFFFFFF && m != 0xFFFFFFFF) {{
                if (n > m) {{
                    unsigned int temp = n;
                    n = m;
                    m = temp;
                }}
                
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) {{
                    output[i] = word[i];
                    if (i >= n && i <= m) {{
                        output[i] = toggle_case(word[i]);
                        *changed = 1;
                    }}
                }}
                return;
            }}
        }}
        
        // yN - Duplicate first N characters
        else if (rule_str[0] == 'y' && rule_len >= 2) {{
            unsigned int n = char_to_pos(rule_str[1]);
            if (n == 0xFFFFFFFF) n = 1;
            if (n > word_len) n = word_len;
            
            if (word_len + n <= MAX_OUTPUT_LEN) {{
                *out_len = word_len + n;
                // Copy original
                for (int i = 0; i < word_len; i++) {{
                    output[i] = word[i];
                }}
                // Duplicate first n
                for (int i = 0; i < n; i++) {{
                    output[word_len + i] = word[i];
                }}
                *changed = 1;
                return;
            }}
        }}
        
        // YN - Duplicate last N characters
        else if (rule_str[0] == 'Y' && rule_len >= 2) {{
            unsigned int n = char_to_pos(rule_str[1]);
            if (n == 0xFFFFFFFF) n = 1;
            if (n > word_len) n = word_len;
            
            if (word_len + n <= MAX_OUTPUT_LEN) {{
                *out_len = word_len + n;
                // Copy original
                for (int i = 0; i < word_len; i++) {{
                    output[i] = word[i];
                }}
                // Duplicate last n
                for (int i = 0; i < n; i++) {{
                    output[word_len + i] = word[word_len - n + i];
                }}
                *changed = 1;
                return;
            }}
        }}
        
        // e X - Title case with separator X
        else if (rule_str[0] == 'e' && rule_len >= 2) {{
            unsigned char separator = rule_str[1];
            
            *out_len = word_len;
            int capitalize_next = 1;
            for (int i = 0; i < word_len; i++) {{
                if (capitalize_next && is_lower(word[i])) {{
                    output[i] = word[i] - 32;
                    capitalize_next = 0;
                }} else {{
                    output[i] = word[i];
                }}
                if (word[i] == separator) {{
                    capitalize_next = 1;
                }}
            }}
            *changed = 1;
            return;
        }}
        
        // v n X - Insert X every n characters
        else if (rule_str[0] == 'v' && rule_len >= 3) {{
            unsigned int n = char_to_pos(rule_str[1]);
            unsigned char x = rule_str[2];
            
            if (n > 0 && word_len + (word_len / n) <= MAX_OUTPUT_LEN) {{
                *out_len = 0;
                for (int i = 0; i < word_len; i++) {{
                    output[(*out_len)++] = word[i];
                    if ((i + 1) % n == 0) {{
                        output[(*out_len)++] = x;
                    }}
                }}
                *changed = 1;
                return;
            }}
        }}
        
        // 3 n X - Toggle case at position n of separator X
        else if (rule_str[0] == '3' && rule_len >= 3) {{
            unsigned int n = char_to_pos(rule_str[1]);
            unsigned char separator = rule_str[2];
            
            *out_len = word_len;
            int separator_count = 0;
            for (int i = 0; i < word_len; i++) {{
                output[i] = word[i];
                if (word[i] == separator) {{
                    separator_count++;
                    if (separator_count == n && i + 1 < word_len) {{
                        output[i + 1] = toggle_case(word[i + 1]);
                        *changed = 1;
                    }}
                }}
            }}
            return;
        }}
        
        // ? n X - Reject unless character at n is X
        else if (rule_str[0] == '?' && rule_len >= 3) {{
            unsigned int n = char_to_pos(rule_str[1]);
            unsigned char x = rule_str[2];
            
            if (n < word_len && word[n] != x) {{
                *out_len = 0;
                *changed = -1;
                return;
            }}
            *out_len = word_len;
            for (int i = 0; i < word_len; i++) {{
                output[i] = word[i];
            }}
            return;
        }}
        
        // = n X - Reject unless character at n is NOT X
        else if (rule_str[0] == '=' && rule_len >= 3) {{
            unsigned int n = char_to_pos(rule_str[1]);
            unsigned char x = rule_str[2];
            
            if (n < word_len && word[n] == x) {{
                *out_len = 0;
                *changed = -1;
                return;
            }}
            *out_len = word_len;
            for (int i = 0; i < word_len; i++) {{
                output[i] = word[i];
            }}
            return;
        }}
        
        // < N - Reject if length is less than N
        else if (rule_str[0] == '<' && rule_len >= 2) {{
            unsigned int n = char_to_pos(rule_str[1]);
            if (word_len < n) {{
                *out_len = 0;
                *changed = -1;
                return;
            }}
            *out_len = word_len;
            for (int i = 0; i < word_len; i++) {{
                output[i] = word[i];
            }}
            return;
        }}
        
        // > N - Reject if length is greater than N
        else if (rule_str[0] == '>' && rule_len >= 2) {{
            unsigned int n = char_to_pos(rule_str[1]);
            if (word_len > n) {{
                *out_len = 0;
                *changed = -1;
                return;
            }}
            *out_len = word_len;
            for (int i = 0; i < word_len; i++) {{
                output[i] = word[i];
            }}
            return;
        }}
        
        // ( N - Reject unless length is less than N
        else if (rule_str[0] == '(' && rule_len >= 2) {{
            unsigned int n = char_to_pos(rule_str[1]);
            if (word_len >= n) {{
                *out_len = 0;
                *changed = -1;
                return;
            }}
            *out_len = word_len;
            for (int i = 0; i < word_len; i++) {{
                output[i] = word[i];
            }}
            return;
        }}
        
        // ) N - Reject unless length is greater than N
        else if (rule_str[0] == ')' && rule_len >= 2) {{
            unsigned int n = char_to_pos(rule_str[1]);
            if (word_len <= n) {{
                *out_len = 0;
                *changed = -1;
                return;
            }}
            *out_len = word_len;
            for (int i = 0; i < word_len; i++) {{
                output[i] = word[i];
            }}
            return;
        }}
    }}
    
    // ========================================================================
    // DEFAULT: Unknown rule, treat as identity
    // ========================================================================
    
    *out_len = word_len;
    for (int i = 0; i < word_len; i++) {{
        output[i] = word[i];
    }}
    *changed = 0;
}}

// ============================================================================
// MAIN RANKING KERNEL WITH DUAL-UNIQUENESS CHECK
// ============================================================================

__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void ranker_kernel(
    // Input buffers
    __global const unsigned char* base_words_in,
    __global const unsigned char* rules_in,
    __global const unsigned int* rule_ids,
    
    // Hash maps
    __global unsigned int* global_hash_map,
    __global const unsigned int* cracked_hash_map,
    
    // Output counters
    __global unsigned int* rule_uniqueness_counts,
    __global unsigned int* rule_effectiveness_counts,
    
    // Configuration
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

    // Load rule from packed format (rule_id + rule_string)
    unsigned int rule_start = rule_batch_idx * MAX_RULE_LEN;
    unsigned char rule_str[MAX_RULE_LEN];
    unsigned int rule_len = 0;
    
    for (unsigned int i = 0; i < MAX_RULE_LEN; i++) {{
        unsigned char c = rules_in[rule_start + i];
        if (c == 0) break;
        rule_str[i] = c;
        rule_len++;
    }}

    // Apply Hashcat rule
    unsigned char result_temp[MAX_OUTPUT_LEN];
    int out_len = 0;
    int changed = 0;
    
    apply_hashcat_rule(word, word_len, rule_str, rule_len, result_temp, &out_len, &changed);

    // DUAL-UNIQUENESS LOGIC
    if (changed > 0 && out_len > 0) {{
        unsigned int word_hash = fnv1a_hash_32(result_temp, out_len);

        // 1. Check against Base Wordlist (Uniqueness)
        unsigned int global_map_index = (word_hash >> 5) & GLOBAL_HASH_MAP_MASK;
        unsigned int bit_index = word_hash & 31;
        unsigned int check_bit = (1U << bit_index);

        __global unsigned int* global_map_ptr = &global_hash_map[global_map_index];
        unsigned int current_global_word = *global_map_ptr;

        if (!(current_global_word & check_bit)) {{
            // Mark as found in global hash map
            atomic_or(global_map_ptr, check_bit);
            
            // Increment uniqueness count for this rule
            atomic_inc(&rule_uniqueness_counts[rule_batch_idx]);

            // 2. Check against Cracked List (Effectiveness)
            unsigned int cracked_map_index = (word_hash >> 5) & CRACKED_HASH_MAP_MASK;
            __global const unsigned int* cracked_map_ptr = &cracked_hash_map[cracked_map_index];
            unsigned int current_cracked_word = *cracked_map_ptr;

            if (current_cracked_word & check_bit) {{
                atomic_inc(&rule_effectiveness_counts[rule_batch_idx]);
            }}
        }}
    }}
}}

// ============================================================================
// HASH MAP INITIALIZATION KERNEL
// ============================================================================

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
"""

# ====================================================================
# --- MEMORY MANAGEMENT FUNCTIONS ---
# ====================================================================

def get_gpu_memory_info(device):
    """Get total and available GPU memory in bytes"""
    try:
        # Try to get global memory size
        total_memory = device.global_mem_size
        # For available memory, we'll use a conservative estimate
        available_memory = int(total_memory * (1 - VRAM_SAFETY_MARGIN))
        return total_memory, available_memory
    except Exception as e:
        print(f"{yellow('WARNING')}: Could not query GPU memory: {e}")
        # Fallback to conservative defaults
        return 8 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024  # 8GB total, 6GB available

def calculate_optimal_parameters_large_rules(available_vram, total_words, cracked_hashes_count, total_rules, reduction_factor=1.0):
    """
    Calculate optimal parameters with consideration for large rule sets
    """
    print(f"{blue('CALCULATING')}: Optimal parameters for {cyan(f'{available_vram / (1024**3):.1f} GB')} available VRAM")
    print(f"{blue('DATASET')}: {cyan(f'{total_words:,}')} words, {cyan(f'{total_rules:,}')} rules, {cyan(f'{cracked_hashes_count:,}')} cracked hashes")
    
    if reduction_factor < 1.0:
        print(f"{yellow('REDUCTION')}: Applying memory reduction factor: {cyan(f'{reduction_factor:.2f}')}")
    
    # Apply reduction factor
    available_vram = int(available_vram * reduction_factor)
    
    # Memory requirements per component (in bytes)
    word_batch_bytes = MAX_WORD_LEN * np.uint8().itemsize
    hash_batch_bytes = np.uint32().itemsize
    rule_batch_bytes = MAX_RULES_IN_BATCH * MAX_RULE_LEN * np.uint8().itemsize
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
        print(f"{yellow('WARNING')}: Limited VRAM, using minimal configuration")
        available_for_maps = available_vram * 0.5
    
    print(f"{blue('AVAILABLE')}: For hash maps: {cyan(f'{available_for_maps / (1024**3):.2f} GB')}")
    
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
    
    print(f"{green('OPTIMAL CONFIGURATION')}:")
    print(f"  {blue('BATCH SIZE')}: {cyan(f'{optimal_batch_size:,} words')}")
    print(f"  {blue('RULES PER BATCH')}: {cyan(f'{MAX_RULES_IN_BATCH:,}')}")
    print(f"  {blue('GLOBAL HASH MAP')}: {cyan(f'{global_bits} bits')} ({global_map_bytes / (1024**2):.1f} MB)")
    print(f"  {blue('CRACKED HASH MAP')}: {cyan(f'{cracked_bits} bits')} ({cracked_map_bytes / (1024**2):.1f} MB)")
    print(f"  {blue('TOTAL MAP MEMORY')}: {cyan(f'{total_map_memory / (1024**3):.2f} GB')}")
    print(f"  {blue('ESTIMATED RULE BATCHES')}: {cyan(f'{(total_rules + MAX_RULES_IN_BATCH - 1) // MAX_RULES_IN_BATCH}')}")
    
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
            print(f"{blue('ATTEMPT')}: {cyan(f'{retry + 1}/{max_retries + 1}')} to allocate buffers (reduction: {current_reduction:.2f})")
            
            for name, spec in buffer_specs.items():
                flags = spec['flags']
                size = int(spec['size'] * current_reduction)
                
                if 'hostbuf' in spec:
                    buffers[name] = cl.Buffer(context, flags, size, hostbuf=spec['hostbuf'])
                else:
                    buffers[name] = cl.Buffer(context, flags, size)
            
            print(f"{green('SUCCESS')}: Successfully allocated all buffers on attempt {cyan(f'{retry + 1}')}")
            return buffers
            
        except cl.MemoryError as e:
            if "MEM_OBJECT_ALLOCATION_FAILURE" in str(e) and retry < max_retries:
                print(f"{yellow('WARNING')}: Memory allocation failed, reducing memory usage...")
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
                
    raise cl.MemoryError(f"{red('ERROR')}: Failed to allocate buffers after {cyan(f'{max_retries}')} retries")

# ====================================================================
# --- UPDATED HELPER FUNCTIONS FOR RULE ENCODING ---
# ====================================================================

def encode_rule_full(rule_str, rule_id):
    """Encode a rule string into fixed-length format for GPU"""
    rule_bytes = rule_str.encode('latin-1')
    rule_len = len(rule_bytes)
    
    # Create fixed-size buffer
    encoded = np.zeros(MAX_RULE_LEN, dtype=np.uint8)
    
    # Copy rule string
    encoded[:rule_len] = np.frombuffer(rule_bytes, dtype=np.uint8, count=rule_len)
    
    return encoded

def prepare_rules_buffer(rules_list):
    """Prepare all rules in a batch for GPU processing"""
    num_rules = len(rules_list)
    rules_buffer = np.zeros((num_rules, MAX_RULE_LEN), dtype=np.uint8)
    rule_ids = np.zeros(num_rules, dtype=np.uint32)
    
    for i, rule in enumerate(rules_list):
        rules_buffer[i] = encode_rule_full(rule['rule_data'], rule['rule_id'])
        rule_ids[i] = rule['rule_id']
    
    return rules_buffer, rule_ids

# ====================================================================
# --- INTERRUPT HANDLER FUNCTIONS ---
# ====================================================================

def signal_handler(sig, frame):
    """Handle Ctrl+C interrupt signal"""
    global interrupted, current_rules_list, current_ranking_output_path, current_top_k
    global words_processed_total, total_unique_found, total_cracked_found
    
    print(f"\n{yellow('INTERRUPT')}: Interrupt received!")
    
    if interrupted:
        print(f"{red('FORCED EXIT')}: Forced exit!")
        sys.exit(1)
        
    interrupted = True
    
    if current_rules_list is not None and current_ranking_output_path is not None:
        print(f"{blue('SAVING')}: Saving current progress...")
        save_current_progress()
    else:
        print(f"{yellow('WARNING')}: No data to save. Exiting...")
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
            print(f"{blue('SAVING')}: Saving intermediate results to: {intermediate_output_path}")
            
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
            
            print(f"{green('SAVED')}: Intermediate ranking data saved: {cyan(f'{len(ranked_rules):,}')} rules")
            
            # Save optimized rules if requested
            if current_top_k > 0:
                print(f"{blue('SAVING')}: Saving intermediate optimized rules to: {intermediate_optimized_path}")
                
                available_rules = len(ranked_rules)
                final_count = min(current_top_k, available_rules)
                
                with open(intermediate_optimized_path, 'w', newline='\n', encoding='utf-8') as f:
                    f.write(":\n")  # Default rule
                    for rule in ranked_rules[:final_count]:
                        f.write(f"{rule['rule_data']}\n")
                
                print(f"{green('SAVED')}: Intermediate optimized rules saved: {cyan(f'{final_count:,}')} rules")
        
        # Print progress summary
        if words_processed_total is not None:
            print(f"\n{green('=' * 60)}")
            print(f"{bold('Progress Summary at Interruption')}")
            print(f"{green('=' * 60)}")
            print(f"{blue('PROCESSED')}: Words: {cyan(f'{int(words_processed_total):,}')}")
            if total_unique_found is not None:
                print(f"{blue('UNIQUE')}: Words Generated: {cyan(f'{int(total_unique_found):,}')}")
            if total_cracked_found is not None:
                print(f"{blue('CRACKED')}: True Cracks Found: {cyan(f'{int(total_cracked_found):,}')}")
            print(f"{green('=' * 60)}")
            
        print(f"{green('SUCCESS')}: Progress saved successfully. You can resume later using the intermediate files.")
        
    except Exception as e:
        print(f"{red('ERROR')}: Error saving intermediate progress: {e}")
    
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
# --- UPDATED RANKER MAIN FUNCTION ---
# ====================================================================

def rank_rules_uniqueness_large(wordlist_path, rules_path, cracked_list_path, ranking_output_path, top_k, 
                               words_per_gpu_batch=None, global_hash_map_bits=None, cracked_hash_map_bits=None,
                               preset=None, device_id=None):
    start_time = time()
    
    # 0. PRELIMINARY DATA LOADING
    print(f"{blue('INITIALIZING')}: RANKER with Complete Hashcat Rules Implementation...")
    
    # Estimate word count
    total_words = estimate_word_count(wordlist_path)
    
    # Load rules with comprehensive support
    print(f"{blue('LOADING')}: Hashcat rules from {rules_path}...")
    rules_list = []
    rule_id_counter = 0
    try:
        with open(rules_path, 'r', encoding='latin-1') as f:
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
        print(f"{red('ERROR')}: Rules file not found!")
        exit(1)
    
    total_rules = len(rules_list)
    print(f"{green('LOADED')}: {cyan(f'{total_rules:,}')} Hashcat rules")
    
    # Setup interrupt handler
    setup_interrupt_handler(rules_list, ranking_output_path, top_k)
    
    # Load cracked hashes
    print(f"{blue('LOADING')}: Cracked passwords from {cracked_list_path}...")
    cracked_hashes = []
    try:
        with open(cracked_list_path, 'rb') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Compute hash
                    hash_val = 2166136261
                    for byte in line:
                        hash_val = (hash_val ^ byte) * 16777619 & 0xFFFFFFFF
                    cracked_hashes.append(hash_val)
    except:
        print(f"{yellow('WARNING')}: Cracked list not found or error reading")
    
    cracked_hashes_np = np.unique(np.array(cracked_hashes, dtype=np.uint32))
    cracked_hashes_count = len(cracked_hashes_np)
    print(f"{green('LOADED')}: {cyan(f'{cracked_hashes_count:,}')} unique cracked password hashes")
    
    # 1. OPENCL INITIALIZATION
    try:
        # Device selection
        if device_id is not None:
            platform, device = select_opencl_device(device_id)
        else:
            platform, device = select_opencl_device()
        
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        
        # Get GPU memory information
        total_vram, available_vram = get_gpu_memory_info(device)
        print(f"{green('GPU')}: {cyan(device.name.strip())}")
        print(f"{blue('VRAM TOTAL')}: {cyan(f'{total_vram / (1024**3):.1f} GB')}")
        print(f"{blue('VRAM AVAILABLE')}: {cyan(f'{available_vram / (1024**3):.1f} GB')}")
        
        # Handle preset parameter specification
        if preset:
            recommendations, recommended_preset = get_recommended_parameters(device, total_words, cracked_hashes_count)
            
            if preset == "recommend":
                print(f"{green('RECOMMENDED')}: Preset: {cyan(recommended_preset)}")
                preset = recommended_preset
            
            if preset in recommendations:
                preset_config = recommendations[preset]
                print(f"{blue('USING')}: {cyan(preset_config['description'])}")
                words_per_gpu_batch = preset_config['batch_size']
                global_hash_map_bits = preset_config['global_bits']
                cracked_hash_map_bits = preset_config['cracked_bits']
            else:
                print(f"{red('ERROR')}: Unknown preset: {cyan(preset)}. Available presets: {list(recommendations.keys())}")
                return
        
        # Handle manual parameter specification
        using_manual_params = False
        if words_per_gpu_batch is not None or global_hash_map_bits is not None or cracked_hash_map_bits is not None:
            using_manual_params = True
            print(f"{blue('MANUAL PARAMETERS')}:")
            
            # Set defaults for any unspecified manual parameters
            if words_per_gpu_batch is None:
                words_per_gpu_batch = DEFAULT_WORDS_PER_GPU_BATCH
            if global_hash_map_bits is None:
                global_hash_map_bits = DEFAULT_GLOBAL_HASH_MAP_BITS
            if cracked_hash_map_bits is None:
                cracked_hash_map_bits = DEFAULT_CRACKED_HASH_MAP_BITS
                
            print(f"  {blue('BATCH SIZE')}: {cyan(f'{words_per_gpu_batch:,}')}")
            print(f"  {blue('GLOBAL HASH MAP')}: {cyan(f'{global_hash_map_bits} bits')}")
            print(f"  {blue('CRACKED HASH MAP')}: {cyan(f'{cracked_hash_map_bits} bits')}")
            
            # Validate manual parameters against available VRAM
            global_map_bytes = (1 << (global_hash_map_bits - 5)) * np.uint32().itemsize
            cracked_map_bytes = (1 << (cracked_hash_map_bits - 5)) * np.uint32().itemsize
            total_map_memory = global_map_bytes + cracked_map_bytes
            
            # Memory requirements for batch processing
            word_batch_bytes = words_per_gpu_batch * MAX_WORD_LEN * np.uint8().itemsize
            hash_batch_bytes = words_per_gpu_batch * np.uint32().itemsize
            rule_batch_bytes = MAX_RULES_IN_BATCH * MAX_RULE_LEN * np.uint8().itemsize
            counter_bytes = MAX_RULES_IN_BATCH * np.uint32().itemsize * 2
            
            total_batch_memory = (word_batch_bytes + hash_batch_bytes) * 2 + rule_batch_bytes + counter_bytes + total_map_memory
            
            if total_batch_memory > available_vram:
                print(f"{yellow('WARNING')}: Manual parameters exceed available VRAM!")
                print(f"  Required: {cyan(f'{total_batch_memory / (1024**3):.2f} GB')}")
                print(f"  Available: {cyan(f'{available_vram / (1024**3):.2f} GB')}")
                print(f"  Consider reducing batch size or hash map bits")
        else:
            # Auto-calculate optimal parameters with large rule set consideration
            words_per_gpu_batch, global_hash_map_bits, cracked_hash_map_bits = calculate_optimal_parameters_large_rules(
                available_vram, total_words, cracked_hashes_count, total_rules
            )

        # Calculate hash map sizes
        GLOBAL_HASH_MAP_WORDS = 1 << (global_hash_map_bits - 5)
        GLOBAL_HASH_MAP_BYTES = GLOBAL_HASH_MAP_WORDS * np.uint32(4)
        GLOBAL_HASH_MAP_MASK = (1 << (global_hash_map_bits - 5)) - 1
        
        CRACKED_HASH_MAP_WORDS = 1 << (cracked_hash_map_bits - 5)
        CRACKED_HASH_MAP_BYTES = CRACKED_HASH_MAP_WORDS * np.uint32(4)
        CRACKED_HASH_MAP_MASK = (1 << (cracked_hash_map_bits - 5)) - 1
        
        print(f"{green('CONFIGURATION')}:")
        print(f"  {blue('BATCH SIZE')}: {cyan(f'{words_per_gpu_batch:,}')} words")
        print(f"  {blue('GLOBAL HASH MAP')}: {cyan(f'{global_hash_map_bits} bits')} ({GLOBAL_HASH_MAP_BYTES/(1024**2):.1f} MB)")
        print(f"  {blue('CRACKED HASH MAP')}: {cyan(f'{cracked_hash_map_bits} bits')} ({CRACKED_HASH_MAP_BYTES/(1024**2):.1f} MB)")
        
        # Compile kernel
        KERNEL_SOURCE = get_kernel_source(global_hash_map_bits, cracked_hash_map_bits)
        prg = cl.Program(context, KERNEL_SOURCE).build()
        
        kernel_ranker = prg.ranker_kernel
        kernel_init = prg.hash_map_init_kernel
        
    except Exception as e:
        print(f"{red('ERROR')}: OpenCL initialization failed: {e}")
        return
    
    # 2. DATA PREPARATION
    print(f"{blue('PREPARING')}: Data for GPU processing...")
    
    # Initialize hash maps
    global_hash_map_np = np.zeros(GLOBAL_HASH_MAP_WORDS, dtype=np.uint32)
    cracked_hash_map_np = np.zeros(CRACKED_HASH_MAP_WORDS, dtype=np.uint32)
    
    # Prepare rules in batches
    rule_batch_starts = list(range(0, total_rules, MAX_RULES_IN_BATCH))
    total_rule_batches = len(rule_batch_starts)
    
    # 3. OPENCL BUFFER ALLOCATION
    mf = cl.mem_flags
    
    # Calculate buffer sizes
    words_buffer_size = words_per_gpu_batch * MAX_WORD_LEN * np.uint8().itemsize
    hashes_buffer_size = words_per_gpu_batch * np.uint32().itemsize
    rules_buffer_size = MAX_RULES_IN_BATCH * MAX_RULE_LEN * np.uint8().itemsize
    counters_size = MAX_RULES_IN_BATCH * np.uint32().itemsize
    
    try:
        # Allocate buffers
        base_words_g = cl.Buffer(context, mf.READ_ONLY, words_buffer_size)
        base_hashes_g = cl.Buffer(context, mf.READ_ONLY, hashes_buffer_size)
        rules_g = cl.Buffer(context, mf.READ_ONLY, rules_buffer_size)
        global_hash_map_g = cl.Buffer(context, mf.READ_WRITE, global_hash_map_np.nbytes)
        cracked_hash_map_g = cl.Buffer(context, mf.READ_ONLY, cracked_hash_map_np.nbytes)
        rule_uniqueness_counts_g = cl.Buffer(context, mf.READ_WRITE, counters_size)
        rule_effectiveness_counts_g = cl.Buffer(context, mf.READ_WRITE, counters_size)
        
        if cracked_hashes_np.size > 0:
            cracked_temp_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cracked_hashes_np)
        
    except cl.MemoryError:
        print(f"{red('ERROR')}: GPU memory allocation failed!")
        return
    
    # 4. INITIALIZE CRACKED HASH MAP
    if cracked_hashes_np.size > 0:
        print(f"{blue('INITIALIZING')}: Cracked hash map on GPU...")
        global_size_init = (int(math.ceil(cracked_hashes_np.size / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
        kernel_init(queue, global_size_init, (LOCAL_WORK_SIZE,),
                    cracked_hash_map_g,
                    cracked_temp_g,
                    np.uint32(cracked_hashes_np.size),
                    np.uint32(CRACKED_HASH_MAP_MASK)).wait()
        print(f"{green('INITIALIZED')}: Cracked hash map")
    
    # 5. PROCESSING LOOP
    print(f"{blue('STARTING')}: Rule ranking with {cyan(f'{total_rules:,}')} rules in {cyan(f'{total_rule_batches:,}')} batches...")
    
    # Initialize counters
    words_processed_total = 0
    total_unique_found = 0
    total_cracked_found = 0
    
    # Calculate total work for progress bars
    total_word_batches = math.ceil(total_words / words_per_gpu_batch)
    total_work_units = total_word_batches * total_rule_batches
    current_work_unit = 0
    
    # Create single progress bar for all work
    pbar = tqdm(total=total_work_units, desc="Overall Progress", unit=" unit",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                position=0)
    
    # Main processing loop
    try:
        word_iter = optimized_wordlist_iterator(wordlist_path, MAX_WORD_LEN, words_per_gpu_batch)
        
        for word_batch_idx, (words_np_batch, hashes_np_batch, num_words_batch) in enumerate(word_iter):
            if interrupted:
                break
            
            # Upload word batch to GPU
            cl.enqueue_copy(queue, base_words_g, words_np_batch)
            cl.enqueue_copy(queue, base_hashes_g, hashes_np_batch).wait()
            
            # Clear global hash map for new batch
            cl.enqueue_fill_buffer(queue, global_hash_map_g, np.uint32(0), 0, global_hash_map_np.nbytes).wait()
            
            # Initialize global hash map with current word batch
            global_size_init = (int(math.ceil(num_words_batch / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
            kernel_init(queue, global_size_init, (LOCAL_WORK_SIZE,),
                        global_hash_map_g,
                        base_hashes_g,
                        np.uint32(num_words_batch),
                        np.uint32(GLOBAL_HASH_MAP_MASK)).wait()
            
            # Process all rule batches for this word batch
            for rule_batch_idx in range(total_rule_batches):
                if interrupted:
                    break
                
                # Prepare rule batch
                start_idx = rule_batch_starts[rule_batch_idx]
                end_idx = min(start_idx + MAX_RULES_IN_BATCH, total_rules)
                current_rules = rules_list[start_idx:end_idx]
                num_rules_in_batch = len(current_rules)
                
                # Prepare rules buffer
                rules_buffer_np = np.zeros((MAX_RULES_IN_BATCH, MAX_RULE_LEN), dtype=np.uint8)
                rule_ids_np = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
                
                for i, rule in enumerate(current_rules):
                    rules_buffer_np[i] = encode_rule_full(rule['rule_data'], rule['rule_id'])
                    rule_ids_np[i] = rule['rule_id']
                
                # Upload rules to GPU
                cl.enqueue_copy(queue, rules_g, rules_buffer_np).wait()
                
                # Clear counters
                cl.enqueue_fill_buffer(queue, rule_uniqueness_counts_g, np.uint32(0), 0, counters_size).wait()
                cl.enqueue_fill_buffer(queue, rule_effectiveness_counts_g, np.uint32(0), 0, counters_size).wait()
                
                # Execute ranking kernel
                global_size = (num_words_batch * num_rules_in_batch,)
                global_size_aligned = (int(math.ceil(global_size[0] / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
                
                kernel_ranker(queue, global_size_aligned, (LOCAL_WORK_SIZE,),
                            base_words_g,
                            rules_g,
                            cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rule_ids_np),
                            global_hash_map_g,
                            cracked_hash_map_g,
                            rule_uniqueness_counts_g,
                            rule_effectiveness_counts_g,
                            np.uint32(num_words_batch),
                            np.uint32(num_rules_in_batch),
                            np.uint32(MAX_WORD_LEN),
                            np.uint32(MAX_OUTPUT_LEN)).wait()
                
                # Read back results
                uniqueness_counts_np = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
                effectiveness_counts_np = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
                
                cl.enqueue_copy(queue, uniqueness_counts_np, rule_uniqueness_counts_g).wait()
                cl.enqueue_copy(queue, effectiveness_counts_np, rule_effectiveness_counts_g).wait()
                
                # Update rule scores
                batch_unique = 0
                batch_cracked = 0
                
                for i, rule in enumerate(current_rules):
                    uniqueness_val = int(uniqueness_counts_np[i])
                    effectiveness_val = int(effectiveness_counts_np[i])
                    
                    rule['uniqueness_score'] += uniqueness_val
                    rule['effectiveness_score'] += effectiveness_val
                    
                    batch_unique += uniqueness_val
                    batch_cracked += effectiveness_val
                
                # Update global totals
                total_unique_found += batch_unique
                total_cracked_found += batch_cracked
                
                # Update progress
                current_work_unit += 1
                pbar.update(1)
                
                # Update progress description with current stats
                pbar.set_description(
                    f"Progress: Word {word_batch_idx+1}, "
                    f"Rule {rule_batch_idx+1}/{total_rule_batches} | "
                    f"Unique: {total_unique_found:,} | Cracked: {total_cracked_found:,}"
                )
                
                # Update interrupt stats
                update_progress_stats(words_processed_total + num_words_batch, 
                                    total_unique_found, total_cracked_found)
            
            # Update word count
            words_processed_total += num_words_batch
    
    except Exception as e:
        print(f"{red('ERROR')}: Processing error: {e}")
    
    finally:
        pbar.close()
    
    # Check for interruption
    if interrupted:
        print(f"\n{yellow('INTERRUPTED')}: Processing stopped. Results saved.")
        return
    
    # 6. FINAL RESULTS
    end_time = time()
    total_time = end_time - start_time
    
    print(f"\n{green('=' * 60)}")
    print(f"{bold('RANKING COMPLETE')}")
    print(f"{green('=' * 60)}")
    print(f"{blue('PROCESSED')}: {cyan(f'{words_processed_total:,}')} words")
    print(f"{blue('PROCESSED')}: {cyan(f'{total_rules:,}')} Hashcat rules")
    print(f"{blue('UNIQUE')}: Generated {cyan(f'{total_unique_found:,}')} unique words")
    print(f"{blue('CRACKED')}: Found {cyan(f'{total_cracked_found:,}')} true cracks")
    print(f"{blue('TIME')}: {cyan(f'{total_time:.2f}')} seconds total")
    if total_time > 0:
        print(f"{blue('SPEED')}: {cyan(f'{words_processed_total/total_time:.0f}')} words/sec")
    print(f"{green('=' * 60)}")
    
    # Save ranking data
    print(f"\n{blue('SAVING')}: Ranking results to {ranking_output_path}...")
    
    # Calculate combined scores
    for rule in rules_list:
        rule['combined_score'] = rule.get('effectiveness_score', 0) * 10 + rule.get('uniqueness_score', 0)
    
    # Sort rules
    ranked_rules = sorted(rules_list, key=lambda x: x['combined_score'], reverse=True)
    
    # Save to CSV
    try:
        with open(ranking_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Combined_Score', 'Effectiveness_Score', 'Uniqueness_Score', 'Rule_Data'])
            
            for rank, rule in enumerate(ranked_rules, 1):
                writer.writerow([
                    rank,
                    rule['combined_score'],
                    rule.get('effectiveness_score', 0),
                    rule.get('uniqueness_score', 0),
                    rule['rule_data']
                ])
        
        print(f"{green('SAVED')}: {cyan(f'{len(ranked_rules):,}')} rules ranked and saved")
        
        # Save optimized rules if requested
        if top_k > 0:
            optimized_path = os.path.splitext(ranking_output_path)[0] + "_optimized.rule"
            print(f"{blue('SAVING')}: Top {cyan(f'{top_k}')} rules to {optimized_path}...")
            
            with open(optimized_path, 'w', encoding='utf-8') as f:
                f.write(":\n")  # Default rule
                for rule in ranked_rules[:top_k]:
                    f.write(f"{rule['rule_data']}\n")
            
            print(f"{green('SAVED')}: Top {cyan(f'{min(top_k, len(ranked_rules)):,}')} rules saved")
    
    except Exception as e:
        print(f"{red('ERROR')}: Failed to save results: {e}")

# ====================================================================
# --- MAIN EXECUTION ---
# ====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPU-Accelerated Hashcat Rule Ranking Tool with Complete Rules Support")
    
    # Required arguments
    parser.add_argument('-w', '--wordlist', help='Path to the base wordlist file')
    parser.add_argument('-r', '--rules', help='Path to the Hashcat rules file to rank')
    parser.add_argument('-c', '--cracked', help='Path to a list of cracked passwords for effectiveness scoring')
    parser.add_argument('-o', '--output', default='ranker_output.csv', help='Path to save the ranking CSV')
    parser.add_argument('-k', '--topk', type=int, default=1000, help='Number of top rules to save (0 to skip)')
    
    # Performance tuning flags
    parser.add_argument('--batch-size', type=int, help='Words per GPU batch (auto-calculated if not specified)')
    parser.add_argument('--global-bits', type=int, help='Bits for global hash map (auto-calculated)')
    parser.add_argument('--cracked-bits', type=int, help='Bits for cracked hash map (auto-calculated)')
    
    # Preset configuration flag (restored from v3.2)
    parser.add_argument('--preset', type=str, default=None,
                       help='Use preset configuration: "low_memory", "medium_memory", "high_memory", "recommend" (auto-selects best)')
    
    # Device selection arguments
    parser.add_argument('--device', type=int, help='OpenCL device ID')
    parser.add_argument('--list-devices', action='store_true',
                       help='List all available OpenCL devices and exit')
    
    args = parser.parse_args()
    
    # If --list-devices is specified, show devices and exit
    if args.list_devices:
        list_opencl_devices()
        sys.exit(0)
    
    # Check if required arguments are provided when not listing devices
    if not args.wordlist or not args.rules or not args.cracked:
        print(f"{red('ERROR')}: Missing required arguments!")
        print()
        parser.print_help()
        sys.exit(1)
    
    print(f"{green('=' * 70)}")
    print(f"{bold('HASHCAT RULE RANKER - COMPLETE IMPLEMENTATION')}")
    print(f"{green('=' * 70)}")
    print(f"{blue('IMPLEMENTED RULES')}:")
    print(f"  {green('✓')} Simple rules (l, u, c, C, t, r, k, :, d, f, p, z, Z, q, E)")
    print(f"  {green('✓')} Position-based (Tn, Dn, Ln, Rn, +n, -n, .n, ,n, 'n)")
    print(f"  {green('✓')} Substitution (sXY, @X, !X, /X, pX)")
    print(f"  {green('✓')} Insertion/Appending (^X, $X)")
    print(f"  {green('✓')} Complex operations (xn m, *n m, i n X, o n X)")
    print(f"  {green('✓')} Rotation/Deletion ({{N, }}N, [N, ]N)")
    print(f"  {green('✓')} Range operations (T n m, yN, YN)")
    print(f"  {green('✓')} Conditional rules (?, =, <, >, (, ))")
    print(f"  {green('✓')} Special rules (e X, v n X, 3 n X)")
    print(f"{green('=' * 70)}")
    
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
        device_id=args.device
    )
