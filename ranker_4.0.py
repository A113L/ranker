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
# --- RANKER v4.0: MULTI-PASS MAB                                  ---
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
# --- FIXED MULTI-ARMED BANDIT (MAB) MODULE WITH PROPER UPDATES ---
# ====================================================================

class RuleBandit:
    """Fixed Multi-Armed Bandit with Thompson Sampling for rule selection"""
    
    def __init__(self, rules_list, exploration_factor=2.0, min_trials=50):
        """
        Initialize the bandit with all rules.
        
        Args:
            rules_list: List of all rules with their metadata
            exploration_factor: Controls exploration vs exploitation (higher = more exploration)
            min_trials: Minimum number of trials before considering a rule for pruning
        """
        self.all_rules = rules_list
        self.num_rules = len(rules_list)
        
        # Thompson Sampling parameters: Beta(alpha, beta) distribution
        self.successes = np.ones(self.num_rules, dtype=np.float32)
        self.failures = np.ones(self.num_rules, dtype=np.float32)
        
        # Performance tracking
        self.trials = np.zeros(self.num_rules, dtype=np.uint32)  # Number of TIMES selected, not words processed
        self.words_processed = np.zeros(self.num_rules, dtype=np.uint64)  # Total words processed with this rule
        self.performance_history = np.zeros((self.num_rules, 20), dtype=np.float32)
        self.performance_index = np.zeros(self.num_rules, dtype=np.uint32)
        
        # Configuration
        self.exploration_factor = exploration_factor
        self.min_trials = min_trials
        
        # Selection tracking
        self.selection_count = np.zeros(self.num_rules, dtype=np.uint32)
        self.last_selected_iteration = np.zeros(self.num_rules, dtype=np.uint32)
        self.current_iteration = 0
        
        # Statistics
        self.total_selections = 0
        self.total_updates = 0
        self.pruned_rules = set()
        self.active_rules = set(range(self.num_rules))
        self.last_selected_batch = []
        
        # Track which rules have been tested at least once
        self.tested_rules = set()
        self.tested_rules_count = 0
        
        # Track pruning statistics
        self.pruning_threshold = 0
        self.last_pruning_check = 0
        
        print(f"{green('MAB INIT')}: Thompson Sampling Bandit with {cyan(f'{self.num_rules:,}')} rules")
        print(f"{blue('MAB CONFIG')}: Exploration factor: {cyan(f'{exploration_factor}')}, Min trials: {cyan(f'{min_trials}')}")
    
    def select_rules(self, batch_size, iteration=0):
        """
        Select rules using Thompson Sampling with proper exploration.
        Ensures all rules get tested multiple times.
        """
        self.current_iteration = iteration
        
        # CRITICAL: First, identify all rules that still need more trials
        rules_needing_trials = []
        for idx in self.active_rules:
            if self.trials[idx] < self.min_trials:
                rules_needing_trials.append(idx)
        
        # Sort by trials (ascending) to prioritize least-tested rules
        rules_needing_trials.sort(key=lambda x: self.trials[x])
        
        # Always select rules that need more trials first - this is crucial
        selected = []
        
        # Take as many rules needing trials as possible, up to batch_size
        if rules_needing_trials:
            num_to_take = min(batch_size, len(rules_needing_trials))
            selected = rules_needing_trials[:num_to_take]
        
        # If we still have slots, fill with Thompson Sampling
        remaining_slots = batch_size - len(selected)
        
        if remaining_slots > 0 and len(self.active_rules) > 0:
            # Get active rules not already selected
            available = list(self.active_rules - set(selected))
            
            if available:
                # Prioritize rules that are closer to reaching min_trials
                available.sort(key=lambda x: (
                    self.trials[x] < self.min_trials,  # Rules needing trials first
                    self.trials[x],  # Then by fewest trials
                    -self.selection_count[x]  # Then by least selected
                ))
                
                # Take some from available
                additional = min(remaining_slots, len(available))
                selected.extend(available[:additional])
        
        # Update statistics
        self.total_selections += 1
        for idx in selected:
            self.trials[idx] += 1  # Increment trial count (number of times selected)
            self.selection_count[idx] += 1
            self.last_selected_iteration[idx] = self.current_iteration
            
            # Mark as tested
            if idx not in self.tested_rules:
                self.tested_rules.add(idx)
                self.tested_rules_count = len(self.tested_rules)
        
        self.last_selected_batch = selected
        return selected
    
    def update(self, selected_indices, successes_array, words_tested):
        """
        Update bandit parameters based on rule performance.
        """
        self.total_updates += 1
        
        for idx, success_count in zip(selected_indices, successes_array):
            if idx >= len(self.successes):
                continue
            
            # Ensure success_count is not negative
            success_count = max(0, success_count)
            
            # Calculate failures (words where rule didn't find a crack)
            failure_count = max(0, words_tested - success_count)
            
            # Track total words processed with this rule
            self.words_processed[idx] += words_tested
            
            # Scale down counts for very large numbers to prevent numerical overflow
            # This only affects the Beta distribution, not the actual counts
            scaled_success = success_count
            scaled_failure = failure_count
            
            if words_tested > 1000000:
                scale = 1000000.0 / words_tested
                scaled_success = int(success_count * scale)
                scaled_failure = int(failure_count * scale)
            
            # Update Beta distribution parameters
            self.successes[idx] += scaled_success
            self.failures[idx] += scaled_failure
            
            # Update performance history (success rate per batch)
            if words_tested > 0:
                success_rate = success_count / words_tested
                hist_idx = int(self.performance_index[idx])
                self.performance_history[idx, hist_idx] = success_rate
                self.performance_index[idx] = (hist_idx + 1) % 20
        
        # Periodically check for pruning (every 50 iterations)
        if self.total_updates % 50 == 0 and self.total_updates > self.last_pruning_check:
            self._prune_worst_rules()
            self.last_pruning_check = self.total_updates
        
        # Decay exploration factor slowly
        self.exploration_factor *= 0.9999
    
    def _prune_worst_rules(self):
        """
        Prune rules that are 1000x worse than the average of the top 100 rules.
        Only considers rules that have been selected at least min_trials times.
        """
        if len(self.active_rules) < 200:  # Need enough rules to make pruning meaningful
            return
        
        active_indices = np.array(list(self.active_rules))
        
        # Only consider rules with sufficient trials
        sufficient_trials_mask = self.trials[active_indices] >= self.min_trials
        
        if not np.any(sufficient_trials_mask):
            return
        
        # Calculate success rates for rules with sufficient trials
        valid_indices = active_indices[sufficient_trials_mask]
        
        success_rates = []
        success_rates_dict = {}
        for idx in valid_indices:
            total_s = self.successes[idx] - 1
            total_f = self.failures[idx] - 1
            total_t = total_s + total_f
            if total_t > 0:
                rate = total_s / total_t
                success_rates.append((idx, rate))
                success_rates_dict[idx] = rate
        
        if len(success_rates) < 100:
            return
        
        # Sort by success rate (descending)
        success_rates.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate average of top 100
        top_100_avg = np.mean([rate for _, rate in success_rates[:100]])
        
        # Set pruning threshold to 1000x worse than top 100 average
        self.pruning_threshold = top_100_avg / 1000
        
        # Find rules to prune
        to_prune = []
        for idx, rate in success_rates:
            if rate < self.pruning_threshold and idx in self.active_rules:
                to_prune.append(idx)
        
        # Prune the worst rules (limit per check to avoid massive changes)
        max_prune_per_check = min(100, len(to_prune) // 10)
        if max_prune_per_check > 0:
            # Take the worst ones first
            to_prune.sort(key=lambda x: success_rates_dict.get(x, 0))
            for idx in to_prune[:max_prune_per_check]:
                if idx in self.active_rules:
                    self.pruned_rules.add(idx)
                    self.active_rules.remove(idx)
                    if idx in self.tested_rules:
                        self.tested_rules.remove(idx)
                        self.tested_rules_count = len(self.tested_rules)
            
            if max_prune_per_check > 0:
                print(f"{yellow('PRUNE')}: Removed {cyan(max_prune_per_check)} rules "
                      f"(threshold: {self.pruning_threshold:.8f}, top 100 avg: {top_100_avg:.8f})")
    
    def get_statistics(self):
        """Get current bandit statistics"""
        active_count = len(self.active_rules)
        tested_count = len(self.tested_rules)
        
        if active_count > 0:
            active_indices = np.array(list(self.active_rules))
            avg_trials = np.mean(self.trials[active_indices])  # Average number of TIMES selected
            avg_selections = np.mean(self.selection_count[active_indices])
            avg_words_processed = np.mean(self.words_processed[active_indices]) if active_count > 0 else 0
            min_trials_active = np.min(self.trials[active_indices]) if len(active_indices) > 0 else 0
            max_trials_active = np.max(self.trials[active_indices]) if len(active_indices) > 0 else 0
            
            # FIXED: Count rules that still need more trials (trials < min_trials)
            rules_needing_trials = 0
            for idx in active_indices:
                if self.trials[idx] < self.min_trials:
                    rules_needing_trials += 1
            
            # Calculate success rates for active rules
            total_successes = self.successes[active_indices] - 1
            total_failures = self.failures[active_indices] - 1
            total_trials_words = total_successes + total_failures
            
            mask = total_trials_words > 0
            if np.any(mask):
                success_rates = total_successes[mask] / total_trials_words[mask]
                avg_success_rate = np.mean(success_rates)
            else:
                avg_success_rate = 0
        else:
            avg_trials = 0
            avg_selections = 0
            avg_words_processed = 0
            avg_success_rate = 0
            min_trials_active = 0
            max_trials_active = 0
            rules_needing_trials = 0
        
        return {
            'total_rules': self.num_rules,
            'active_rules': active_count,
            'tested_rules': tested_count,
            'pruned_rules': len(self.pruned_rules),
            'rules_needing_trials': rules_needing_trials,  # This will now decrease over time
            'avg_trials_per_rule': float(avg_trials),  # This is now number of selections, not words
            'avg_words_processed_per_rule': float(avg_words_processed),
            'avg_selections_per_rule': float(avg_selections),
            'avg_success_rate': float(avg_success_rate),
            'min_trials_active': int(min_trials_active),
            'max_trials_active': int(max_trials_active),
            'exploration_factor': float(self.exploration_factor),
            'total_selections': int(self.total_selections),
            'total_updates': int(self.total_updates),
            'current_iteration': int(self.current_iteration),
            'pruning_threshold': float(self.pruning_threshold)
        }
    
    def get_top_rules(self, n=100):
        """Get top n rules based on estimated success probability"""
        if len(self.active_rules) == 0:
            return []
        
        active_indices = np.array(list(self.active_rules))
        
        # Calculate success probability for each active rule
        total_successes = self.successes[active_indices] - 1
        total_failures = self.failures[active_indices] - 1
        total_trials_words = total_successes + total_failures
        
        # Only consider rules with sufficient trials (min_trials selections)
        sufficient_trials_mask = self.trials[active_indices] >= self.min_trials
        
        success_probs = np.zeros(len(active_indices))
        if np.any(sufficient_trials_mask):
            success_probs[sufficient_trials_mask] = total_successes[sufficient_trials_mask] / total_trials_words[sufficient_trials_mask]
        else:
            return []
        
        # Get indices of top rules
        valid_indices = np.where(sufficient_trials_mask)[0]
        if len(valid_indices) == 0:
            return []
        
        top_k = min(n, len(valid_indices))
        top_local_indices = valid_indices[np.argpartition(-success_probs[valid_indices], top_k)[:top_k]]
        top_indices = active_indices[top_local_indices]
        
        # Sort by success probability
        sort_order = np.argsort(-success_probs[top_local_indices])
        top_indices = top_indices[sort_order]
        
        # Return rules with their probabilities
        results = []
        for idx in top_indices[:n]:
            total_s = self.successes[idx] - 1
            total_f = self.failures[idx] - 1
            total_t = total_s + total_f
            success_prob = total_s / total_t if total_t > 0 else 0.0
            
            results.append({
                'rule_id': idx,
                'rule_data': self.all_rules[idx]['rule_data'],
                'success_probability': float(success_prob),
                'trials': int(self.trials[idx]),  # Number of times selected
                'words_processed': int(self.words_processed[idx]),  # Total words processed
                'selections': int(self.selection_count[idx]),
                'successes': int(total_s),
                'failures': int(total_f),
                'total_tested': int(total_t)
            })
        
        return results

# ====================================================================
# --- CONSTANTS ---
# ====================================================================
MAX_WORD_LEN = 256
MAX_OUTPUT_LEN = 512
MAX_RULE_LEN = 255
MAX_RULES_IN_BATCH = 1024
LOCAL_WORK_SIZE = 256

# Default values
DEFAULT_WORDS_PER_GPU_BATCH = 100000
DEFAULT_GLOBAL_HASH_MAP_BITS = 35
DEFAULT_CRACKED_HASH_MAP_BITS = 33

# MAB Configuration
DEFAULT_MAB_EXPLORATION_FACTOR = 2.0
DEFAULT_MAB_MIN_TRIALS = 50  # Increased default for better statistical significance

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
        for device_info in devices:
            if device_info['index'] == device_id:
                print(f"{green('SELECTED')}: Device {device_id} - {device_info['type']}: {device_info['name']}")
                return device_info['platform'], device_info['device']
        
        print(f"{yellow('WARNING')}: Device ID {device_id} not found. Using default selection.")
    
    gpu_devices = [d for d in devices if d['type'] == 'GPU']
    
    if gpu_devices:
        selected = max(gpu_devices, key=lambda x: x['memory_gb'])
        print(f"{green('AUTO-SELECTED')}: Device {selected['index']} - {selected['type']}: {selected['name']} ({selected['memory_gb']:.2f} GB)")
        return selected['platform'], selected['device']
    else:
        cpu_devices = [d for d in devices if d['type'] == 'CPU']
        if cpu_devices:
            selected = max(cpu_devices, key=lambda x: x['memory_gb'])
            print(f"{green('AUTO-SELECTED')}: Device {selected['index']} - {selected['type']}: {selected['name']}")
            return selected['platform'], selected['device']
    
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
        sample_size = min(10 * 1024 * 1024, file_size)
        
        with open(path, 'rb') as f:
            sample = f.read(sample_size)
            lines = sample.count(b'\n')
            
            if file_size <= sample_size:
                total_lines = lines
            else:
                avg_line_length = sample_size / max(lines, 1)
                total_lines = int(file_size / avg_line_length)
                
        print(f"{green('ESTIMATED')}: {cyan(f'{total_lines:,}')} words")
        return total_lines
        
    except Exception as e:
        print(f"{yellow('WARNING')}: Could not estimate word count: {e}")
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
                
                with tqdm(total=file_size, desc="Loading wordlist", unit="B", unit_scale=True) as pbar:
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
        print(f"{red('ERROR')}: Error in optimized loader: {e}")
        raise
    
    load_time = time() - load_start
    print(f"{green('LOADED')}: {cyan(f'{total_words_loaded:,}')} words in {load_time:.2f}s "
          f"({total_words_loaded/load_time:,.0f} words/sec)")

# ====================================================================
# --- COMPREHENSIVE HASHCAT RULES KERNEL ---
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

int is_lower(unsigned char c) {{
    return (c >= 'a' && c <= 'z');
}}

int is_upper(unsigned char c) {{
    return (c >= 'A' && c <= 'Z');
}}

int is_digit(unsigned char c) {{
    return (c >= '0' && c <= '9');
}}

int is_alnum(unsigned char c) {{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}}

unsigned char toggle_case(unsigned char c) {{
    if (is_lower(c)) return c - 32;
    if (is_upper(c)) return c + 32;
    return c;
}}

unsigned char to_lower(unsigned char c) {{
    if (is_upper(c)) return c + 32;
    return c;
}}

unsigned char to_upper(unsigned char c) {{
    if (is_lower(c)) return c - 32;
    return c;
}}

unsigned int fnv1a_hash_32(const unsigned char* data, unsigned int len) {{
    unsigned int hash = 2166136261U;
    for (unsigned int i = 0; i < len; i++) {{
        hash ^= data[i];
        hash *= 16777619U;
    }}
    return hash;
}}

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
    
    for (int i = 0; i < MAX_OUTPUT_LEN; i++) output[i] = 0;
    
    if (rule_len == 0 || word_len == 0) return;
    
    // ========================================================================
    // SIMPLE RULES (1 character)
    // ========================================================================
    
    if (rule_len == 1) {{
        switch (rule_str[0]) {{
            case 'l':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) output[i] = to_lower(word[i]);
                *changed = 1;
                return;
            case 'u':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) output[i] = to_upper(word[i]);
                *changed = 1;
                return;
            case 'c':
                *out_len = word_len;
                if (word_len > 0) {{
                    output[0] = to_upper(word[0]);
                    for (int i = 1; i < word_len; i++) output[i] = to_lower(word[i]);
                }}
                *changed = 1;
                return;
            case 'C':
                *out_len = word_len;
                if (word_len > 0) {{
                    output[0] = to_lower(word[0]);
                    for (int i = 1; i < word_len; i++) output[i] = to_upper(word[i]);
                }}
                *changed = 1;
                return;
            case 't':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) output[i] = toggle_case(word[i]);
                *changed = 1;
                return;
            case 'r':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) output[i] = word[word_len - 1 - i];
                *changed = 1;
                return;
            case 'k':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) output[i] = word[i];
                if (word_len >= 2) {{
                    output[0] = word[1];
                    output[1] = word[0];
                    *changed = 1;
                }}
                return;
            case ':':
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) output[i] = word[i];
                *changed = 0;
                return;
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
            case 'p':
                if (word_len + 1 <= MAX_OUTPUT_LEN) {{
                    *out_len = word_len;
                    for (int i = 0; i < word_len; i++) output[i] = word[i];
                    output[*out_len] = 's';
                    (*out_len)++;
                    *changed = 1;
                }}
                return;
            case 'z':
                if (word_len + 1 <= MAX_OUTPUT_LEN) {{
                    output[0] = word[0];
                    for (int i = 0; i < word_len; i++) output[i + 1] = word[i];
                    *out_len = word_len + 1;
                    *changed = 1;
                }}
                return;
            case 'Z':
                if (word_len + 1 <= MAX_OUTPUT_LEN) {{
                    for (int i = 0; i < word_len; i++) output[i] = word[i];
                    output[word_len] = word[word_len - 1];
                    *out_len = word_len + 1;
                    *changed = 1;
                }}
                return;
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
                    if (word[i] == ' ' || word[i] == '-' || word[i] == '_') capitalize_next = 1;
                }}
                *changed = 1;
                return;
            default:
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) output[i] = word[i];
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
        
        if (is_digit(arg)) {{
            int n = arg - '0';
            
            switch (cmd) {{
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
                case 'D':
                    *out_len = 0;
                    for (int i = 0; i < word_len; i++) {{
                        if (i != n) output[(*out_len)++] = word[i];
                        else *changed = 1;
                    }}
                    return;
                case 'L':
                    *out_len = 0;
                    for (int i = n; i < word_len; i++) output[(*out_len)++] = word[i];
                    *changed = 1;
                    return;
                case 'R':
                    *out_len = n + 1;
                    if (*out_len > word_len) *out_len = word_len;
                    for (int i = 0; i < *out_len; i++) output[i] = word[i];
                    *changed = 1;
                    return;
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
        
        switch (cmd) {{
            case '^':
                if (word_len + 1 <= MAX_OUTPUT_LEN) {{
                    output[0] = arg;
                    for (int i = 0; i < word_len; i++) output[i + 1] = word[i];
                    *out_len = word_len + 1;
                    *changed = 1;
                }}
                return;
            case '$':
                if (word_len + 1 <= MAX_OUTPUT_LEN) {{
                    for (int i = 0; i < word_len; i++) output[i] = word[i];
                    output[word_len] = arg;
                    *out_len = word_len + 1;
                    *changed = 1;
                }}
                return;
            case '@':
                *out_len = 0;
                for (int i = 0; i < word_len; i++) {{
                    if (word[i] != arg) output[(*out_len)++] = word[i];
                    else *changed = 1;
                }}
                return;
            case '!':
                for (int i = 0; i < word_len; i++) {{
                    if (word[i] == arg) {{
                        *out_len = 0;
                        *changed = -1;
                        return;
                    }}
                }}
                *out_len = word_len;
                for (int i = 0; i < word_len; i++) output[i] = word[i];
                return;
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
                for (int i = 0; i < word_len; i++) output[i] = word[i];
                return;
            }}
            case 'p':
                *out_len = 0;
                for (int i = 0; i < word_len; i++) {{
                    if (word[i] != arg) output[(*out_len)++] = word[i];
                    else *changed = 1;
                }}
                return;
        }}
    }}
    
    // ========================================================================
    // SUBSTITUTION RULES (sXY)
    // ========================================================================
    
    else if (rule_len == 3 && rule_str[0] == 's') {{
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
                for (unsigned int i = n; i <= m && i < word_len; i++) output[(*out_len)++] = word[i];
                *changed = (*out_len > 0);
                return;
            }}
        }}
        else if (rule_str[0] == '*' && rule_len >= 3) {{
            unsigned int n = char_to_pos(rule_str[1]);
            unsigned int m = char_to_pos(rule_str[2]);
            
            *out_len = word_len;
            for (int i = 0; i < word_len; i++) output[i] = word[i];
            if (n != 0xFFFFFFFF && m != 0xFFFFFFFF && n < word_len && m < word_len && n != m) {{
                unsigned char temp = output[n];
                output[n] = output[m];
                output[m] = temp;
                *changed = 1;
            }}
            return;
        }}
        else if (rule_str[0] == 'i' && rule_len >= 3) {{
            unsigned int n = char_to_pos(rule_str[1]);
            unsigned char x = rule_str[2];
            
            if (n != 0xFFFFFFFF && word_len + 1 <= MAX_OUTPUT_LEN) {{
                *out_len = 0;
                for (int i = 0; i < word_len; i++) {{
                    if (i == n) output[(*out_len)++] = x;
                    output[(*out_len)++] = word[i];
                }}
                if (n >= word_len) output[(*out_len)++] = x;
                *changed = 1;
                return;
            }}
        }}
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
        else if (rule_str[0] == '[' && rule_len >= 2) {{
            unsigned int n = char_to_pos(rule_str[1]);
            if (n == 0xFFFFFFFF) n = 1;
            if (n > word_len) n = word_len;
            
            *out_len = word_len - n;
            for (int i = n; i < word_len; i++) output[i - n] = word[i];
            *changed = 1;
            return;
        }}
        else if (rule_str[0] == ']' && rule_len >= 2) {{
            unsigned int n = char_to_pos(rule_str[1]);
            if (n == 0xFFFFFFFF) n = 1;
            if (n > word_len) n = word_len;
            
            *out_len = word_len - n;
            for (int i = 0; i < *out_len; i++) output[i] = word[i];
            *changed = 1;
            return;
        }}
    }}
    
    *out_len = word_len;
    for (int i = 0; i < word_len; i++) output[i] = word[i];
    *changed = 0;
}}

// ============================================================================
// MAIN RANKING KERNEL
// ============================================================================

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

    unsigned char word[MAX_WORD_LEN];
    unsigned int word_len = 0;
    for (unsigned int i = 0; i < max_word_len; i++) {{
        unsigned char c = base_words_in[word_idx * max_word_len + i];
        if (c == 0) break;
        word[i] = c;
        word_len++;
    }}

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

// ============================================================================
// HASH MAP INITIALIZATION KERNEL
// ============================================================================

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
# --- MEMORY MANAGEMENT FUNCTIONS ---
# ====================================================================

def get_gpu_memory_info(device):
    """Get total and available GPU memory in bytes"""
    try:
        total_memory = device.global_mem_size
        available_memory = int(total_memory * (1 - VRAM_SAFETY_MARGIN))
        return total_memory, available_memory
    except Exception as e:
        print(f"{yellow('WARNING')}: Could not query GPU memory: {e}")
        return 8 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024

def calculate_optimal_parameters_large_rules(available_vram, total_words, cracked_hashes_count, total_rules, reduction_factor=1.0):
    """Calculate optimal parameters with consideration for large rule sets"""
    print(f"{blue('CALCULATING')}: Optimal parameters for {cyan(f'{available_vram / (1024**3):.1f} GB')} available VRAM")
    print(f"{blue('DATASET')}: {cyan(f'{total_words:,}')} words, {cyan(f'{total_rules:,}')} rules, {cyan(f'{cracked_hashes_count:,}')} cracked hashes")
    
    if reduction_factor < 1.0:
        print(f"{yellow('REDUCTION')}: Applying memory reduction factor: {cyan(f'{reduction_factor:.2f}')}")
    
    available_vram = int(available_vram * reduction_factor)
    
    word_batch_bytes = MAX_WORD_LEN * np.uint8().itemsize
    hash_batch_bytes = np.uint32().itemsize
    rule_batch_bytes = MAX_RULES_IN_BATCH * MAX_RULE_LEN * np.uint8().itemsize
    counter_bytes = MAX_RULES_IN_BATCH * np.uint32().itemsize * 2
    
    base_memory = (
        (word_batch_bytes + hash_batch_bytes) * 2 +
        rule_batch_bytes + counter_bytes
    )
    
    if total_rules > 100000:
        suggested_batch_size = min(DEFAULT_WORDS_PER_GPU_BATCH, 50000)
    else:
        suggested_batch_size = DEFAULT_WORDS_PER_GPU_BATCH
    
    available_for_maps = available_vram - base_memory
    if available_for_maps <= 0:
        print(f"{yellow('WARNING')}: Limited VRAM, using minimal configuration")
        available_for_maps = available_vram * 0.5
    
    print(f"{blue('AVAILABLE')}: For hash maps: {cyan(f'{available_for_maps / (1024**3):.2f} GB')}")
    
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
    optimal_batch_size = (optimal_batch_size // LOCAL_WORK_SIZE) * LOCAL_WORK_SIZE
    
    if total_rules > 50000:
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
    """Get recommended parameter values based on GPU capabilities and dataset size"""
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

# ====================================================================
# --- HELPER FUNCTIONS FOR RULE ENCODING ---
# ====================================================================

def encode_rule_full(rule_str, rule_id):
    """Encode a rule string into fixed-length format for GPU"""
    rule_bytes = rule_str.encode('latin-1')
    rule_len = len(rule_bytes)
    
    encoded = np.zeros(MAX_RULE_LEN, dtype=np.uint8)
    encoded[:rule_len] = np.frombuffer(rule_bytes, dtype=np.uint8, count=rule_len)
    
    return encoded

def prepare_rules_buffer(rules_list, selected_indices):
    """Prepare selected rules in a batch for GPU processing"""
    num_rules = len(selected_indices)
    rules_buffer = np.zeros((MAX_RULES_IN_BATCH, MAX_RULE_LEN), dtype=np.uint8)
    rule_ids = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
    
    for i, rule_idx in enumerate(selected_indices):
        if i >= MAX_RULES_IN_BATCH:
            break
        rule = rules_list[rule_idx]
        rules_buffer[i] = encode_rule_full(rule['rule_data'], rule['rule_id'])
        rule_ids[i] = rule['rule_id']
    
    return rules_buffer, rule_ids, num_rules

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
        base_path = os.path.splitext(current_ranking_output_path)[0]
        intermediate_output_path = f"{base_path}_INTERRUPTED.csv"
        intermediate_optimized_path = f"{base_path}_INTERRUPTED.rule"
        
        if current_rules_list:
            print(f"{blue('SAVING')}: Saving intermediate results to: {intermediate_output_path}")
            
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
            
            print(f"{green('SAVED')}: Intermediate ranking data saved: {cyan(f'{len(ranked_rules):,}')} rules")
            
            if current_top_k > 0:
                print(f"{blue('SAVING')}: Saving intermediate optimized rules to: {intermediate_optimized_path}")
                
                available_rules = len(ranked_rules)
                final_count = min(current_top_k, available_rules)
                
                with open(intermediate_optimized_path, 'w', newline='\n', encoding='utf-8') as f:
                    f.write(":\n")
                    for rule in ranked_rules[:final_count]:
                        f.write(f"{rule['rule_data']}\n")
                
                print(f"{green('SAVED')}: Intermediate optimized rules saved: {cyan(f'{final_count:,}')} rules")
        
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
    
    signal.signal(signal.SIGINT, signal_handler)

def update_progress_stats(words_processed, unique_found, cracked_found):
    """Update progress statistics for interrupt handler"""
    global words_processed_total, total_unique_found, total_cracked_found
    words_processed_total = words_processed
    total_unique_found = unique_found
    total_cracked_found = cracked_found

# ====================================================================
# --- MAIN RANKER FUNCTION WITH MULTI-PASS MAB ---
# ====================================================================

def rank_rules_uniqueness_large(wordlist_path, rules_path, cracked_list_path, ranking_output_path, top_k, 
                               words_per_gpu_batch=None, global_hash_map_bits=None, cracked_hash_map_bits=None,
                               preset=None, device_id=None, mab_exploration_factor=None, mab_min_trials=None):
    start_time = time()
    
    # 0. PRELIMINARY DATA LOADING
    print(f"{blue('INITIALIZING')}: RANKER v4.3 with Multi-Pass Multi-Armed Bandits...")
    
    # Estimate word count
    total_words = estimate_word_count(wordlist_path)
    
    # Load rules
    print(f"{blue('LOADING')}: Hashcat rules from {rules_path}...")
    rules_list = []
    rule_id_counter = 0
    try:
        with open(rules_path, 'r', encoding='latin-1') as f:
            for line in tqdm(f, desc="Loading rules", unit=" rules"):
                rule = line.strip()
                if not rule or rule.startswith('#'):
                    continue
                rules_list.append({
                    'rule_data': rule, 
                    'rule_id': rule_id_counter, 
                    'uniqueness_score': 0, 
                    'effectiveness_score': 0,
                    'times_tested': 0,
                    'total_successes': 0,
                    'total_trials': 0
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
                    hash_val = 2166136261
                    for byte in line:
                        hash_val = (hash_val ^ byte) * 16777619 & 0xFFFFFFFF
                    cracked_hashes.append(hash_val)
    except:
        print(f"{yellow('WARNING')}: Cracked list not found or error reading")
    
    cracked_hashes_np = np.unique(np.array(cracked_hashes, dtype=np.uint32))
    cracked_hashes_count = len(cracked_hashes_np)
    print(f"{green('LOADED')}: {cyan(f'{cracked_hashes_count:,}')} unique cracked password hashes")
    
    # 1. INITIALIZE MULTI-ARMED BANDIT
    exploration_factor = mab_exploration_factor if mab_exploration_factor else DEFAULT_MAB_EXPLORATION_FACTOR
    min_trials = mab_min_trials if mab_min_trials else DEFAULT_MAB_MIN_TRIALS
    
    print(f"{green('MAB')}: Initializing Thompson Sampling Bandit...")
    rule_bandit = RuleBandit(
        rules_list, 
        exploration_factor=exploration_factor,
        min_trials=min_trials
    )
    
    # 2. OPENCL INITIALIZATION
    try:
        if device_id is not None:
            platform, device = select_opencl_device(device_id)
        else:
            platform, device = select_opencl_device()
        
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        
        total_vram, available_vram = get_gpu_memory_info(device)
        print(f"{green('GPU')}: {cyan(device.name.strip())}")
        print(f"{blue('VRAM TOTAL')}: {cyan(f'{total_vram / (1024**3):.1f} GB')}")
        print(f"{blue('VRAM AVAILABLE')}: {cyan(f'{available_vram / (1024**3):.1f} GB')}")
        
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
        
        using_manual_params = False
        if words_per_gpu_batch is not None or global_hash_map_bits is not None or cracked_hash_map_bits is not None:
            using_manual_params = True
            print(f"{blue('MANUAL PARAMETERS')}:")
            
            if words_per_gpu_batch is None:
                words_per_gpu_batch = DEFAULT_WORDS_PER_GPU_BATCH
            if global_hash_map_bits is None:
                global_hash_map_bits = DEFAULT_GLOBAL_HASH_MAP_BITS
            if cracked_hash_map_bits is None:
                cracked_hash_map_bits = DEFAULT_CRACKED_HASH_MAP_BITS
                
            print(f"  {blue('BATCH SIZE')}: {cyan(f'{words_per_gpu_batch:,}')}")
            print(f"  {blue('GLOBAL HASH MAP')}: {cyan(f'{global_hash_map_bits} bits')}")
            print(f"  {blue('CRACKED HASH MAP')}: {cyan(f'{cracked_hash_map_bits} bits')}")
            
            global_map_bytes = (1 << (global_hash_map_bits - 5)) * np.uint32().itemsize
            cracked_map_bytes = (1 << (cracked_hash_map_bits - 5)) * np.uint32().itemsize
            total_map_memory = global_map_bytes + cracked_map_bytes
            
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
            words_per_gpu_batch, global_hash_map_bits, cracked_hash_map_bits = calculate_optimal_parameters_large_rules(
                available_vram, total_words, cracked_hashes_count, total_rules
            )

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
        
        KERNEL_SOURCE = get_kernel_source(global_hash_map_bits, cracked_hash_map_bits)
        prg = cl.Program(context, KERNEL_SOURCE).build()
        
        kernel_ranker = prg.ranker_kernel
        kernel_init = prg.hash_map_init_kernel
        
    except Exception as e:
        print(f"{red('ERROR')}: OpenCL initialization failed: {e}")
        return
    
    # 3. LOAD ALL WORD BATCHES INTO MEMORY FOR MULTIPLE PASSES
    print(f"{blue('LOADING')}: Loading wordlist into memory for multiple passes...")
    all_word_batches = []
    all_hash_batches = []
    all_batch_sizes = []
    
    word_iter = optimized_wordlist_iterator(wordlist_path, MAX_WORD_LEN, words_per_gpu_batch)
    for words_np_batch, hashes_np_batch, num_words_batch in word_iter:
        all_word_batches.append(words_np_batch)
        all_hash_batches.append(hashes_np_batch)
        all_batch_sizes.append(num_words_batch)
    
    total_word_batches = len(all_word_batches)
    total_words_loaded = sum(all_batch_sizes)
    print(f"{green('LOADED')}: {cyan(f'{total_word_batches}')} word batches, {cyan(f'{total_words_loaded:,}')} total words")
    
    # 4. OPENCL BUFFER ALLOCATION
    mf = cl.mem_flags
    
    words_buffer_size = words_per_gpu_batch * MAX_WORD_LEN * np.uint8().itemsize
    hashes_buffer_size = words_per_gpu_batch * np.uint32().itemsize
    rules_buffer_size = MAX_RULES_IN_BATCH * MAX_RULE_LEN * np.uint8().itemsize
    counters_size = MAX_RULES_IN_BATCH * np.uint32().itemsize
    
    try:
        base_words_g = cl.Buffer(context, mf.READ_ONLY, words_buffer_size)
        base_hashes_g = cl.Buffer(context, mf.READ_ONLY, hashes_buffer_size)
        rules_g = cl.Buffer(context, mf.READ_ONLY, rules_buffer_size)
        global_hash_map_g = cl.Buffer(context, mf.READ_WRITE, GLOBAL_HASH_MAP_WORDS * np.uint32(4))
        cracked_hash_map_g = cl.Buffer(context, mf.READ_ONLY, CRACKED_HASH_MAP_WORDS * np.uint32(4))
        rule_uniqueness_counts_g = cl.Buffer(context, mf.READ_WRITE, counters_size)
        rule_effectiveness_counts_g = cl.Buffer(context, mf.READ_WRITE, counters_size)
        
        if cracked_hashes_np.size > 0:
            cracked_temp_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cracked_hashes_np)
        
    except cl.MemoryError:
        print(f"{red('ERROR')}: GPU memory allocation failed!")
        return
    
    # 5. INITIALIZE CRACKED HASH MAP
    if cracked_hashes_np.size > 0:
        print(f"{blue('INITIALIZING')}: Cracked hash map on GPU...")
        global_size_init = (int(math.ceil(cracked_hashes_np.size / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
        kernel_init(queue, global_size_init, (LOCAL_WORK_SIZE,),
                    cracked_hash_map_g,
                    cracked_temp_g,
                    np.uint32(cracked_hashes_np.size),
                    np.uint32(CRACKED_HASH_MAP_MASK)).wait()
        print(f"{green('INITIALIZED')}: Cracked hash map")
    
    # 6. MULTI-PASS MAB PROCESSING LOOP
    print(f"{blue('STARTING')}: Multi-pass rule ranking with MAB optimization...")
    
    # Reset counters
    words_processed_total = 0
    total_unique_found = 0
    total_cracked_found = 0
    
    # Statistics tracking
    mab_stats_history = []
    batch_statistics = []
    
    # Calculate total iterations needed - CRITICAL FIX
    # We need to select each rule at least min_trials times
    total_rule_selections_needed = total_rules * min_trials
    selections_per_batch = MAX_RULES_IN_BATCH
    estimated_iterations = int(math.ceil(total_rule_selections_needed / selections_per_batch))
    estimated_word_passes = int(math.ceil(estimated_iterations / total_word_batches))
    
    print(f"{blue('MAB LOOP')}: Need {cyan(f'{total_rule_selections_needed:,}')} total rule selections")
    print(f"{blue('MAB LOOP')}: Can select {cyan(f'{selections_per_batch:,}')} rules per batch")
    print(f"{blue('MAB LOOP')}: Will process approximately {cyan(f'{estimated_iterations:,}')} batches")
    print(f"{blue('MAB LOOP')}: Estimated {cyan(f'{estimated_word_passes}')} passes through wordlist")
    print(f"{blue('MAB LOOP')}: Target: {cyan(f'{min_trials}')} selections per rule")
    
    # Initialize iteration counter and pass counter
    iteration = 0
    pass_count = 0
    
    # Progress bar for total iterations
    iter_pbar = tqdm(total=estimated_iterations, desc="MAB Iterations", unit="batch",
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                     position=0)
    
    try:
        # Continue until ALL active rules have been selected at least min_trials times
        while not interrupted:
            # Check if all active rules have reached min_trials
            if len(rule_bandit.active_rules) > 0:
                active_array = np.array(list(rule_bandit.active_rules))
                min_trials_achieved = np.min(rule_bandit.trials[active_array]) >= min_trials
                
                if min_trials_achieved:
                    print(f"\n{green('COMPLETE')}: All {len(active_array):,} active rules have been selected at least {min_trials} times!")
                    break
            else:
                print(f"\n{yellow('WARNING')}: No active rules remaining!")
                break
            
            # Start a new pass through the wordlist
            pass_count += 1
            mab_stats = rule_bandit.get_statistics()
            print(f"\n{green('PASS')} {pass_count}: Starting pass through wordlist (Rules needing trials: {mab_stats['rules_needing_trials']:,})")
            
            # Cycle through all word batches
            for batch_idx in range(total_word_batches):
                if interrupted:
                    break
                
                # Get current batch
                words_np_batch = all_word_batches[batch_idx]
                hashes_np_batch = all_hash_batches[batch_idx]
                num_words_batch = all_batch_sizes[batch_idx]
                
                # Upload word batch to GPU
                cl.enqueue_copy(queue, base_words_g, words_np_batch)
                cl.enqueue_copy(queue, base_hashes_g, hashes_np_batch).wait()
                
                # Clear global hash map for new batch
                cl.enqueue_fill_buffer(queue, global_hash_map_g, np.uint32(0), 0, GLOBAL_HASH_MAP_WORDS * np.uint32(4)).wait()
                
                # Initialize global hash map with current word batch
                global_size_init = (int(math.ceil(num_words_batch / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
                kernel_init(queue, global_size_init, (LOCAL_WORK_SIZE,),
                            global_hash_map_g,
                            base_hashes_g,
                            np.uint32(num_words_batch),
                            np.uint32(GLOBAL_HASH_MAP_MASK)).wait()
                
                # MAB: Select rules for this batch - prioritizes rules needing more trials
                selected_indices = rule_bandit.select_rules(
                    batch_size=MAX_RULES_IN_BATCH,
                    iteration=iteration
                )
                
                num_rules_in_batch = len(selected_indices)
                
                if num_rules_in_batch == 0:
                    # Force select from rules with lowest trials
                    if len(rule_bandit.active_rules) > 0:
                        active_list = list(rule_bandit.active_rules)
                        # Sort by trials (ascending) to prioritize least-tested rules
                        active_list.sort(key=lambda x: rule_bandit.trials[x])
                        num_to_select = min(MAX_RULES_IN_BATCH, len(active_list))
                        selected_indices = active_list[:num_to_select]
                    else:
                        print(f"{yellow('WARNING')}: No active rules remaining!")
                        break
                    
                    num_rules_in_batch = len(selected_indices)
                
                # Prepare rules buffer
                rules_buffer_np, rule_ids_np, actual_num_rules = prepare_rules_buffer(rules_list, selected_indices)
                
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
                
                # MAB: Update bandit parameters
                effectiveness_counts_selected = effectiveness_counts_np[:num_rules_in_batch]
                rule_bandit.update(selected_indices, effectiveness_counts_selected, num_words_batch)
                
                # Update rule scores and track statistics
                batch_unique = 0
                batch_cracked = 0
                
                for i, rule_idx in enumerate(selected_indices):
                    if i >= num_rules_in_batch:
                        break
                    
                    rule = rules_list[rule_idx]
                    uniqueness_val = int(uniqueness_counts_np[i])
                    effectiveness_val = int(effectiveness_counts_np[i])
                    
                    rule['uniqueness_score'] += uniqueness_val
                    rule['effectiveness_score'] += effectiveness_val
                    rule['times_tested'] += 1
                    rule['total_successes'] += effectiveness_val
                    rule['total_trials'] += num_words_batch
                    
                    batch_unique += uniqueness_val
                    batch_cracked += effectiveness_val
                
                # Update global totals
                total_unique_found += batch_unique
                total_cracked_found += batch_cracked
                words_processed_total += num_words_batch
                
                # Update progress bar
                iter_pbar.update(1)
                iteration += 1
                
                # Get MAB statistics
                mab_stats = rule_bandit.get_statistics()
                mab_stats_history.append(mab_stats)
                
                # Record batch statistics
                batch_statistics.append({
                    'batch': iteration,
                    'pass': pass_count,
                    'words_processed': num_words_batch,
                    'rules_tested': num_rules_in_batch,
                    'unique_found': batch_unique,
                    'cracked_found': batch_cracked,
                    'active_rules': mab_stats['active_rules'],
                    'tested_rules': mab_stats['tested_rules'],
                    'pruned_rules': mab_stats['pruned_rules'],
                    'avg_trials': mab_stats['avg_trials_per_rule'],
                    'rules_needing_trials': mab_stats['rules_needing_trials']
                })
                
                # Extract values for f-strings to avoid nested quote issues
                tested_rules_val = mab_stats['tested_rules']
                avg_trials_val = mab_stats['avg_trials_per_rule']
                success_rate_val = mab_stats['avg_success_rate'] * 100
                active_rules_val = mab_stats['active_rules']
                pruned_rules_val = mab_stats['pruned_rules']
                rules_needing_trials_val = mab_stats['rules_needing_trials']
                min_trials_active_val = mab_stats['min_trials_active']
                
                # Update progress description - NOW WITH CORRECT COUNTING
                iter_pbar.set_description(f"Pass {pass_count}, Iter {iteration} | "
                                        f"Need: {rules_needing_trials_val:,}/{active_rules_val:,} | "
                                        f"Min Trials: {min_trials_active_val}/{min_trials} | "
                                        f"Avg: {avg_trials_val:.1f} | "
                                        f"Success: {success_rate_val:.3f}%")
                
                # Update interrupt stats
                update_progress_stats(words_processed_total, total_unique_found, total_cracked_found)
                
                # Print detailed statistics every 50 iterations
                if iteration % 50 == 0:
                    print(f"\n{green('STATS')} Pass {pass_count}, Iter {iteration}:")
                    print(f"  {blue('WORDS')}: Processed {cyan(f'{words_processed_total:,}')} total words")
                    print(f"  {blue('UNIQUE')}: Found {cyan(f'{total_unique_found:,}')} unique words")
                    print(f"  {blue('CRACKED')}: Found {cyan(f'{total_cracked_found:,}')} true cracks")
                    print(f"  {blue('MAB')}: Rules needing trials: {cyan(f'{rules_needing_trials_val:,}/{active_rules_val:,}')}, "
                          f"Min trials: {cyan(f'{min_trials_active_val}/{min_trials}')}, "
                          f"Avg trials: {cyan(f'{avg_trials_val:.1f}')}, "
                          f"Active: {cyan(f'{active_rules_val:,}')}, "
                          f"Pruned: {cyan(f'{pruned_rules_val:,}')}")
                
                # Early break if all active rules have reached min_trials
                if rules_needing_trials_val == 0 and active_rules_val > 0:
                    print(f"\n{green('COMPLETE')}: All active rules have reached {min_trials} trials!")
                    break
            
            # Check if we should continue
            if len(rule_bandit.active_rules) > 0:
                active_array = np.array(list(rule_bandit.active_rules))
                if np.min(rule_bandit.trials[active_array]) >= min_trials:
                    break
                
    except Exception as e:
        print(f"{red('ERROR')}: Processing error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        iter_pbar.close()
    
    # Check for interruption
    if interrupted:
        print(f"\n{yellow('INTERRUPTED')}: Processing stopped. Results saved.")
        return
    
    # 7. FINAL RESULTS
    end_time = time()
    total_time = end_time - start_time
    
    final_mab_stats = rule_bandit.get_statistics()
    top_mab_rules = rule_bandit.get_top_rules(20)
    
    print(f"\n{green('=' * 80)}")
    print(f"{bold('MULTI-PASS MAB RANKING COMPLETE')}")
    print(f"{green('=' * 80)}")
    print(f"{blue('PROCESSED')}: {cyan(f'{words_processed_total:,}')} words total")
    print(f"{blue('PROCESSED')}: {cyan(f'{total_rules:,}')} Hashcat rules")
    print(f"{blue('PASSES')}: {cyan(f'{pass_count}')} passes through wordlist")
    print(f"{blue('ITERATIONS')}: {cyan(f'{iteration}')} MAB iterations")
    print(f"{blue('UNIQUE')}: Generated {cyan(f'{total_unique_found:,}')} unique words")
    print(f"{blue('CRACKED')}: Found {cyan(f'{total_cracked_found:,}')} true cracks")
    print(f"{blue('TIME')}: {cyan(f'{total_time:.2f}')} seconds total")
    
    if total_time > 0:
        speed = words_processed_total / total_time
        print(f"{blue('SPEED')}: {cyan(f'{speed:.0f}')} words/sec")
    
    print(f"\n{green('FIXED MAB FINAL STATISTICS')}:")
    active_rules = final_mab_stats['active_rules']
    total_rules_count = final_mab_stats['total_rules']
    tested_rules = final_mab_stats['tested_rules']
    pruned_rules = final_mab_stats['pruned_rules']
    pruned_percentage = pruned_rules / total_rules_count * 100 if total_rules_count > 0 else 0
    avg_trials = final_mab_stats['avg_trials_per_rule']
    avg_selections = final_mab_stats['avg_selections_per_rule']
    success_rate = final_mab_stats['avg_success_rate'] * 100
    avg_words_processed_val = final_mab_stats['avg_words_processed_per_rule']
    min_trials_active_val = final_mab_stats['min_trials_active']
    pruning_threshold_val = final_mab_stats['pruning_threshold']
    
    print(f"  {blue('ACTIVE RULES')}: {cyan(f'{active_rules:,}/{total_rules_count:,}')}")
    print(f"  {blue('TESTED RULES')}: {cyan(f'{tested_rules:,}/{total_rules_count:,}')}")
    print(f"  {blue('PRUNED RULES')}: {cyan(f'{pruned_rules:,}')} ({pruned_percentage:.1f}%)")
    print(f"  {blue('MIN TRIALS ACTIVE')}: {cyan(f'{min_trials_active_val}/{min_trials}')}")
    print(f"  {blue('AVG TRIALS PER RULE')}: {cyan(f'{avg_trials:.1f}')} (each trial = 1 batch of words)")
    print(f"  {blue('AVG WORDS PROCESSED PER RULE')}: {cyan(f'{avg_words_processed_val:.0f}')}")
    print(f"  {blue('AVG SELECTIONS PER RULE')}: {cyan(f'{avg_selections:.1f}')}")
    print(f"  {blue('AVG SUCCESS RATE')}: {cyan(f'{success_rate:.6f}%')}")
    if pruning_threshold_val > 0:
        print(f"  {blue('PRUNING THRESHOLD')}: {cyan(f'{pruning_threshold_val:.8f}')}")
    
    print(f"\n{green('TOP 10 RULES BY MAB SUCCESS PROBABILITY')}:")
    if top_mab_rules:
        for i, rule in enumerate(top_mab_rules[:10], 1):
            rule_data = rule['rule_data']
            success_prob = rule['success_probability']
            trials = rule['trials']  # Number of times selected
            words_processed = rule['words_processed']  # Total words processed
            selections = rule['selections']
            successes = rule['successes']
            total_tested = rule['total_tested']
            success_rate_percent = (successes / total_tested * 100) if total_tested > 0 else 0
            
            print(f"  {blue(f'{i:2}.')} {cyan(rule_data):30} "
                  f"Success: {success_prob:.6f} "
                  f"Trials: {trials:,} "
                  f"Words: {words_processed:,} "
                  f"Successes: {successes:,} "
                  f"Rate: {success_rate_percent:.4f}%")
    else:
        print(f"  {yellow('No rules with sufficient trials yet - run longer to see top rules!')}")
    
    print(f"{green('=' * 80)}")
    
    # Save ranking data
    print(f"\n{blue('SAVING')}: Ranking results to {ranking_output_path}...")
    
    for rule in rules_list:
        rule_idx = rule['rule_id']
        total_successes = rule['total_successes']
        total_trials = rule['total_trials']
        
        if total_trials > 0:
            actual_success_rate = total_successes / total_trials
        else:
            actual_success_rate = 0
        
        rule['mab_success_prob'] = actual_success_rate
        rule['selections'] = rule_bandit.selection_count[rule_idx] if rule_idx < len(rule_bandit.selection_count) else 0
        rule['mab_trials'] = rule_bandit.trials[rule_idx] if rule_idx < len(rule_bandit.trials) else 0
        rule['combined_score'] = (
            rule.get('effectiveness_score', 0) * 10 + 
            rule.get('uniqueness_score', 0) +
            actual_success_rate * 1000
        )
    
    ranked_rules = sorted(rules_list, key=lambda x: x['combined_score'], reverse=True)
    
    try:
        with open(ranking_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Combined_Score', 'Effectiveness_Score', 'Uniqueness_Score', 
                           'MAB_Success_Prob', 'Times_Tested', 'MAB_Trials', 'Selections', 
                           'Total_Successes', 'Total_Trials', 'Rule_Data'])
            
            for rank, rule in enumerate(ranked_rules, 1):
                writer.writerow([
                    rank,
                    f"{rule['combined_score']:.2f}",
                    rule.get('effectiveness_score', 0),
                    rule.get('uniqueness_score', 0),
                    f"{rule.get('mab_success_prob', 0):.8f}",
                    rule.get('times_tested', 0),
                    rule.get('mab_trials', 0),
                    rule.get('selections', 0),
                    rule.get('total_successes', 0),
                    rule.get('total_trials', 0),
                    rule['rule_data']
                ])
        
        print(f"{green('SAVED')}: {cyan(f'{len(ranked_rules):,}')} rules ranked and saved")
        
        if top_k > 0:
            optimized_path = os.path.splitext(ranking_output_path)[0] + "_optimized.rule"
            print(f"{blue('SAVING')}: Top {cyan(f'{top_k}')} rules to {optimized_path}...")
            
            with open(optimized_path, 'w', encoding='utf-8') as f:
                f.write(":\n")
                for rule in ranked_rules[:top_k]:
                    f.write(f"{rule['rule_data']}\n")
            
            print(f"{green('SAVED')}: Top {cyan(f'{min(top_k, len(ranked_rules)):,}')} rules saved")
        
        # Save MAB statistics
        mab_stats_path = os.path.splitext(ranking_output_path)[0] + "_mab_stats.csv"
        with open(mab_stats_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration', 'Pass', 'Active_Rules', 'Tested_Rules', 'Pruned_Rules', 
                           'Rules_Needing_Trials', 'Min_Trials_Active', 'Avg_Trials', 'Avg_Words_Processed', 
                           'Avg_Selections', 'Avg_Success_Rate', 'Total_Selections', 'Total_Updates'])
            
            for i, stats in enumerate(mab_stats_history):
                pass_val = batch_statistics[i]['pass'] if i < len(batch_statistics) else 0
                writer.writerow([
                    i,
                    pass_val,
                    stats['active_rules'],
                    stats['tested_rules'],
                    stats['pruned_rules'],
                    stats['rules_needing_trials'],
                    stats['min_trials_active'],
                    f"{stats['avg_trials_per_rule']:.2f}",
                    f"{stats['avg_words_processed_per_rule']:.0f}",
                    f"{stats['avg_selections_per_rule']:.2f}",
                    f"{stats['avg_success_rate']:.8f}",
                    stats['total_selections'],
                    stats['total_updates']
                ])
        
        print(f"{green('SAVED')}: MAB statistics to {cyan(mab_stats_path)}")
        
        # Save batch statistics
        batch_stats_path = os.path.splitext(ranking_output_path)[0] + "_batch_stats.csv"
        with open(batch_stats_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration', 'Pass', 'Words_Processed', 'Rules_Tested', 'Unique_Found', 
                           'Cracked_Found', 'Active_Rules', 'Tested_Rules', 'Pruned_Rules', 
                           'Avg_Trials', 'Rules_Needing_Trials'])
            
            for stats in batch_statistics:
                writer.writerow([
                    stats['batch'],
                    stats['pass'],
                    stats['words_processed'],
                    stats['rules_tested'],
                    stats['unique_found'],
                    stats['cracked_found'],
                    stats['active_rules'],
                    stats['tested_rules'],
                    stats['pruned_rules'],
                    f"{stats['avg_trials']:.2f}",
                    stats.get('rules_needing_trials', 0)
                ])
        
        print(f"{green('SAVED')}: Batch statistics to {cyan(batch_stats_path)}")
        
    except Exception as e:
        print(f"{red('ERROR')}: Failed to save results: {e}")

# ====================================================================
# --- MAIN EXECUTION ---
# ====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPU-Accelerated Hashcat Rule Ranking Tool with Multi-Pass Multi-Armed Bandits")
    
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
    
    # MAB configuration flags - CRITICAL: min_trials is now number of TIMES each rule is selected
    parser.add_argument('--mab-exploration', type=float, default=DEFAULT_MAB_EXPLORATION_FACTOR,
                       help=f'Multi-Armed Bandit exploration factor (default: {DEFAULT_MAB_EXPLORATION_FACTOR})')
    parser.add_argument('--mab-min-trials', type=int, default=DEFAULT_MAB_MIN_TRIALS,
                       help=f'MAB minimum number of TIMES each rule must be selected (default: {DEFAULT_MAB_MIN_TRIALS})')
    
    # Preset configuration flag
    parser.add_argument('--preset', type=str, default=None,
                       help='Use preset configuration: "low_memory", "medium_memory", "high_memory", "recommend" (auto-selects best)')
    
    # Device selection arguments
    parser.add_argument('--device', type=int, help='OpenCL device ID')
    parser.add_argument('--list-devices', action='store_true',
                       help='List all available OpenCL devices and exit')
    
    # Additional options
    parser.add_argument('--max-passes', type=int, default=0,
                       help='Maximum number of passes through wordlist (0 = unlimited until all rules tested)')
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_opencl_devices()
        sys.exit(0)
    
    if not args.wordlist or not args.rules or not args.cracked:
        print(f"{red('ERROR')}: Missing required arguments!")
        print()
        parser.print_help()
        sys.exit(1)
    
    print(f"{green('=' * 80)}")
    print(f"{bold('HASHCAT RULE RANKER v4.0 - WITH MULTI-PASS MULTI-ARMED BANDITS')}")
    print(f"{green('=' * 80)}")
    print(f"{bold('🎓 MINIMUM VIABLE PRODUCT (MVP) PROJECT')}")
    print(f"{bold('   GPU-Accelerated Rule Ranking with Multi-Armed Bandits')}")
    print(f"{green('=' * 80)}")
    print(f"{blue('MVP FEATURES')}:")
    print(f"  {green('✓')} Multi-pass wordlist processing for repeated rule testing")
    print(f"  {green('✓')} Thompson Sampling Multi-Armed Bandit for intelligent rule selection")
    print(f"  {green('✓')} Proper trial counting (each selection = 1 trial, not words processed)")
    print(f"  {green('✓')} Continues until ALL rules reach min_trials threshold")
    print(f"  {green('✓')} Prunes rules that are 1000x worse than top 100 average")
    print(f"  {green('✓')} Separate tracking of selections, words processed, and success rates")
    print(f"{green('=' * 80)}")
    print(f"{blue('MAB CONFIG')}: min_trials = {cyan(f'{args.mab_min_trials}')} selections per rule")
    print(f"{blue('MAB CONFIG')}: exploration = {cyan(f'{args.mab_exploration}')}")
    print(f"{blue('PROJECT STATUS')}: {cyan('MVP - Core functionality complete, testing in progress')}")
    print(f"{green('=' * 80)}")
    
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
        device_id=args.device,
        mab_exploration_factor=args.mab_exploration,
        mab_min_trials=args.mab_min_trials
    )
