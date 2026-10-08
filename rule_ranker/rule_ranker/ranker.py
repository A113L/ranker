#!/usr/bin/env python3
"""
Ranker v6.0 – GPU-Accelerated Hashcat Rule Ranking
===================================================
Multi-Pass MAB with Early Elimination for large rule sets.
Optionally runs in legacy exhaustive mode (v3.2).
All GPU‑compatible Hashcat rules are implemented.
MAX_RULE_LEN = 255, comprehensive rule application.

Changelog v6.0 (architecture/correctness fixes):
- Independent per-rule GPU fingerprint tables; no shared uniqueness bitmap or race-order credit.
- FNV-1a-64 fingerprints and versioned cracked-list caches.
- MAB trials use stratified wordlist samples instead of full wordlist passes.

Changelog v5.2 (historical bug fixes):
- BUG FIX #1: Operator precedence error in select_rules sort_key.
  `<< 32 - x` was parsed as `<< (32 - x)` instead of `(<< 32) - x`,
  causing completely wrong screening-phase sort order.
- BUG FIX #2: np.uint32 silent overflow when computing GPU hash-map byte
  sizes in both rank_rules_exhaustive and rank_rules_mab. For 35-bit maps
  (4 GB), multiplying by np.uint32(4) wraps to 0, causing wrong buffer
  allocation. Fixed by using plain Python int arithmetic.
- BUG FIX #3: should_exclude_rule did not exclude bare '!' (single char).
  '!' is a reject operator even without an argument; added to the
  single-char exclusion set.
- BUG FIX #4: duplicate_front (OpenCL C, 'y' operator) appended the
  duplicated front chars to the *back* of the word instead of the *front*,
  reversing the effect of 'y' vs 'Y'. Fixed to prepend correctly.
- BUG FIX #5: load_cracked_hashes progress bar advanced 1 byte past EOF
  for files without a trailing newline, causing a >100% display glitch.
  Clamped advance to remaining bytes.

Changelog v5.1:
- Added rule validation identical to rulest_v2.py (HashcatRuleValidator).
- Rules containing banned operators (M 4 6 X < > ! / ( ) = % Q) are now
  rejected at load time, matching rulest's behaviour.
- Removed possibility of processing rules that rulest would discard.
"""

import pyopencl as cl
import numpy as np
import argparse
import csv
import json
import heapq
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

from .hashing import (
    build_open_addressing_table_uint64,
    fast_fnv1a_hash_32,
    fast_fnv1a_hash_64,
    load_cached_hashes,
    save_cached_hashes,
    table_size_for_count,
)

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
# --- RULE VALIDATOR (identical to rulest_v2.py) ---
# ====================================================================
MAX_GPU_RULES = 255

def should_exclude_rule(rule):
    """Return True if the rule uses an operator that is permanently excluded."""
    if not rule:
        return False
    # Single-character reject/memory ops
    # BUG FIX: added '!' to single-char exclusion list; it is a reject op even alone
    if len(rule) == 1 and rule in ('_', 'M', '4', '6', 'Q', '!'):
        return True
    # Two-character reject ops (some have a digit, but we check only first char)
    if len(rule) == 2 and rule[0] in ('!', '/', '(', ')', '<', '>', '_'):
        return True
    # Three-character reject ops (e.g. '=0', '%0', 'Q0')
    if len(rule) == 3 and rule[0] in ('?', '=', 'v'):
        return True
    return False

class HashcatRuleValidator:
    MAX_GPU_RULES = MAX_GPU_RULES
    @staticmethod
    def is_digit(c): return '0' <= c <= '9'
    @staticmethod
    def is_pos(c):
        # BUG FIX #7: Hashcat position arguments are not decimal-only. Per the
        # hashcat rule reference, positions >9 are encoded as 'A'-'Z' (A=10 ... Z=35).
        # This applies to T,D,L,R,+,-,.,,,',y,Y,z,Z and the N/M parts of i,o,x,O,*,3.
        # The old digit-only check rejected every valid rule targeting position >=10
        # (e.g. 'TA', ''C', 'x0A'), which was the main source of false GPU-incompatible
        # rejections.
        return ('0' <= c <= '9') or ('A' <= c <= 'Z')
    @staticmethod
    def validate_rule_for_gpu(rule_str):
        if should_exclude_rule(rule_str): return False
        pos = cnt = 0
        n = len(rule_str)
        isd = HashcatRuleValidator.is_digit
        isp = HashcatRuleValidator.is_pos
        while pos < n:
            c = rule_str[pos]
            if c == ' ': pos+=1; continue
            if c == 'p':
                # BUG FIX #8: hashcat's rule table marks 'pN' (Duplicate N) with the
                # same '*' annotation as T/D/z/Z/etc, meaning N is a MANDATORY,
                # position-encoded argument (0-9A-Z), not an optional decimal digit.
                # The old code let bare 'p' pass and only consumed a following digit,
                # so arguments like 'pV' (duplicate 31 times) left 'V' unconsumed and
                # broke parsing of the rest of the rule.
                pos+=1
                if pos>=n or not isp(rule_str[pos]): return False
                pos+=1; cnt+=1; continue
            if c in ('z','Z'):
                # BUG FIX #7: zN/ZN take a mandatory position-encoded argument
                # (0-9A-Z), not an optional decimal digit. Previously an argument
                # like 'zA' left the 'A' unconsumed, causing the whole rule to be
                # rejected on the next loop iteration.
                pos+=1
                if pos>=n or not isp(rule_str[pos]): return False
                pos+=1; cnt+=1; continue
            if c in (':','l','u','c','C','t','r','d','f','a','q','k','K','E','{','}','[',']'):
                pos+=1; cnt+=1; continue
            if c in ('T','D','L','R','+','-','.',',',"'",'y','Y'):
                pos+=1
                if pos>=n or not isp(rule_str[pos]): return False
                pos+=1; cnt+=1; continue
            if c in ('i','o','3'):
                pos+=1
                if pos>=n or not isp(rule_str[pos]): return False
                pos+=1
                if pos>=n: return False
                pos+=1; cnt+=1; continue
            if c in ('x','*','O'):
                pos+=1
                if pos>=n or not isp(rule_str[pos]): return False
                pos+=1
                if pos>=n or not isp(rule_str[pos]): return False
                pos+=1; cnt+=1; continue
            if c == 's':
                pos+=1
                if pos+1>=n: return False
                pos+=2; cnt+=1; continue
            if c in ('@','e','$','^'):
                pos+=1
                if pos>=n: return False
                pos+=1; cnt+=1; continue
            return False
        return cnt <= MAX_GPU_RULES

    @staticmethod
    def validate_rules_for_gpu(rules):
        valid = []
        for r in rules:
            r = r.strip('\n\r')
            if r and HashcatRuleValidator.validate_rule_for_gpu(r):
                valid.append(r)
        return valid

# ----------------------------------------------------------------------
# Colors & helpers
# ----------------------------------------------------------------------
class C:
    RED = '\033[91m'; GREEN = '\033[92m'; YELLOW = '\033[93m'
    BLUE = '\033[94m'; CYAN = '\033[96m'; MAGENTA= '\033[95m'
    BOLD = '\033[1m';  DIM = '\033[2m';   END = '\033[0m'

def red(t): return f"{C.RED}{t}{C.END}"
def green(t): return f"{C.GREEN}{t}{C.END}"
def yellow(t): return f"{C.YELLOW}{t}{C.END}"
def blue(t): return f"{C.BLUE}{t}{C.END}"
def cyan(t): return f"{C.CYAN}{t}{C.END}"
def bold(t): return f"{C.BOLD}{t}{C.END}"
def dim(t): return f"{C.DIM}{t}{C.END}"

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

def _fnv_cache_paths(source_path, max_len):
    """Backward-compatible helper retained for callers of older releases."""
    from .hashing import cache_paths
    return cache_paths(source_path, max_len)


def _load_fnv_cache(source_path, max_len):
    return load_cached_hashes(source_path, max_len)


def _save_fnv_cache(source_path, max_len, hashes, n_skipped):
    return save_cached_hashes(source_path, max_len, hashes, n_skipped)


def optimized_wordlist_iterator(wordlist_path, max_len, batch_size):
    """Memory‑mapped iterator over words, returning batches of words and hashes"""
    print(f"{green('Using optimized memory-mapped loader...')}")
    file_size = os.path.getsize(wordlist_path)
    print(f"{blue('File size:')} {cyan(f'{file_size / (1024**3):.2f} GB')}")
    batch_elements = batch_size * max_len
    words_buffer = np.zeros(batch_elements, dtype=np.uint8)
    hashes_buffer = np.zeros(batch_size, dtype=np.uint64)
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
                    hashes_buffer[batch_count] = fast_fnv1a_hash_64(line)
                    batch_count += 1
                    total_words_loaded += 1
                    if batch_count >= batch_size:
                        # Public batch contract: (N, max_len) uint8 matrix.
                        # Keeping the shape explicit prevents legacy callers from
                        # accidentally slicing N bytes out of an N*max_len buffer.
                        yield (
                            words_buffer.reshape(batch_size, max_len).copy(),
                            hashes_buffer.copy(),
                            batch_count,
                        )
                        batch_count = 0
                        words_buffer.fill(0)
                        hashes_buffer.fill(0)
                if batch_count > 0 and not interrupted:
                    yield (
                        words_buffer.reshape(batch_size, max_len)[:batch_count].copy(),
                        hashes_buffer[:batch_count].copy(),
                        batch_count,
                    )
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
            print(f"{green('=' * 60)}{C.END}\n")
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
    """Loads Hashcat rules from file, filtering out rules invalid according to rulest."""
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
    total_lines = 0
    invalid_count = 0
    overlong_count = 0
    try:
        with open(path, 'r', encoding='latin-1') as f:
            for line in f:
                total_lines += 1
                rule = line.strip()
                if not rule or rule.startswith('#'):
                    continue
                # Keep the GPU rule buffer contract explicit: never silently
                # truncate a rule that would not fit in MAX_RULE_LEN bytes.
                if len(rule.encode('latin-1', errors='ignore')) > MAX_RULE_LEN:
                    overlong_count += 1
                    continue
                # Validate using rulest's validator
                if not HashcatRuleValidator.validate_rule_for_gpu(rule):
                    invalid_count += 1
                    continue
                rules_list.append({'rule_data': rule, 'rule_id': rule_id_counter,
                                   'uniqueness_score': 0, 'effectiveness_score': 0})
                rule_id_counter += 1
    except FileNotFoundError:
        print(f"{red('Error:')} Rules file not found at: {path}")
        exit(1)
    if invalid_count > 0:
        print(f"{yellow('Warning:')} {cyan(f'{invalid_count:,}')} rules were skipped because they are not GPU-compatible (rulest validator).")
    if overlong_count > 0:
        print(f"{yellow('Warning:')} {cyan(f'{overlong_count:,}')} rules longer than {MAX_RULE_LEN} characters were skipped (no truncation).")
    print(f"{green('Loaded')} {cyan(f'{len(rules_list):,}')} {bold('valid rules.')}")
    return rules_list

def load_cracked_hashes(path, max_len):
    """Load cracked passwords and return sorted unique FNV-1a-64 fingerprints."""
    from array import array

    print(f"{blue('Loading cracked list for effectiveness check from:')} {path}...")
    cached = _load_fnv_cache(path, max_len)
    if cached is not None:
        hashes, _n_skipped = cached
        print(f"{green('Loaded')} {cyan(f'{len(hashes):,}')} {bold('unique cracked password 64-bit fingerprints from cache.')}")
        return hashes

    cracked_hashes = array('Q')
    n_skipped = 0
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
                        advance = min((end_pos + 1) - pos, file_size - pos)
                        pos = end_pos + 1
                        pbar.update(advance)
                        if 1 <= len(line) <= max_len:
                            cracked_hashes.append(fast_fnv1a_hash_64(line))
                        elif len(line) > max_len:
                            n_skipped += 1
    except FileNotFoundError:
        print(f"{yellow('Warning:')} Cracked list file not found at: {path}. Effectiveness scores will be zero.")
        return np.array([], dtype=np.uint64)

    raw_hashes = np.frombuffer(cracked_hashes, dtype=np.uint64).copy() if cracked_hashes.itemsize == 8 else np.asarray(cracked_hashes, dtype=np.uint64)
    unique_hashes = np.unique(raw_hashes)
    _save_fnv_cache(path, max_len, unique_hashes, n_skipped)
    print(f"{green('Loaded')} {cyan(f'{len(unique_hashes):,}')} {bold('unique cracked password 64-bit fingerprints.')}")
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
    """Load ranking data and save Top K without materializing the full CSV.

    The bounded heap preserves the old stable ``reverse=True`` sort semantics:
    higher Combined_Score wins, and equal scores keep their original CSV order.
    """
    if not csv_path:
        print(f"{yellow('Optimization skipped: Ranking CSV path is missing.')}")
        return
    print(f"{blue('Loading ranking from CSV:')} {csv_path} {bold('and saving Top')} {cyan(f'{top_k}')} {bold('Optimized Rules to:')} {output_path}...")
    valid_count = 0
    ranked_data = []
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if top_k > 0:
                heap = []
                for seq, row in enumerate(reader):
                    try:
                        score = int(row['Combined_Score'])
                    except (KeyError, TypeError, ValueError):
                        continue
                    valid_count += 1
                    entry = (score, -seq, row)
                    if len(heap) < top_k:
                        heapq.heappush(heap, entry)
                    elif entry[:2] > heap[0][:2]:
                        heapq.heapreplace(heap, entry)
                ranked_data = [row for _score, _neg_seq, row in
                               sorted(heap, key=lambda item: (-item[0], -item[1]))]
            else:
                # Preserve the old top_k<=0 behaviour exactly.
                for row in reader:
                    try:
                        row['Combined_Score'] = int(row['Combined_Score'])
                        ranked_data.append(row)
                    except (KeyError, TypeError, ValueError):
                        continue
                valid_count = len(ranked_data)
                ranked_data.sort(key=lambda row: row['Combined_Score'], reverse=True)
    except FileNotFoundError:
        print(f"{red('Error: Ranking CSV file not found at:')} {csv_path}")
        return
    except Exception as e:
        print(f"{red('Error while reading CSV:')} {e}")
        return

    print(f"{blue('Loaded')} {cyan(f'{valid_count:,}')} {bold('total valid rules from CSV')}")
    available_rules = valid_count
    if top_k > available_rules:
        print(f"{yellow('Warning: Requested')} {cyan(f'{top_k:,}')} {bold('rules but only')} {cyan(f'{available_rules:,}')} {bold('available. Saving')} {cyan(f'{available_rules:,}')} {bold('rules.')}")
        final_optimized_list = ranked_data
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


def save_top_k_rules(ranking_list, output_path, top_k):
    """Write Top K directly from the already-ranked in-memory list.

    ``save_ranking_data()`` sorts ``ranking_list`` in place immediately before
    this helper is called.  Re-reading and re-sorting the CSV would therefore
    do the same work twice and is especially wasteful for large ``-k`` runs.
    """
    available_rules = len(ranking_list)
    final_count = min(max(int(top_k), 0), available_rules)
    if top_k > available_rules:
        print(f"{yellow('Warning: Requested')} {cyan(f'{top_k:,}')} {bold('rules but only')} {cyan(f'{available_rules:,}')} {bold('available. Saving')} {cyan(f'{available_rules:,}')} {bold('rules.')}")
    if final_count <= 0:
        print(f"{red('No rules available for optimized output.')}")
        return
    try:
        with open(output_path, 'w', newline='\n', encoding='utf-8') as f:
            f.write(":\n")
            for rule in ranking_list[:final_count]:
                f.write(f"{rule['rule_data']}\n")
        print(f"{green('Top')} {cyan(f'{final_count:,}')} {bold('optimized rules saved successfully to')} {output_path}.")
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
def get_kernel_source(rule_hash_table_bits, cracked_hash_table_bits):
    rule_hash_table_mask = (1 << int(rule_hash_table_bits)) - 1
    cracked_hash_table_mask = (1 << int(cracked_hash_table_bits)) - 1
    return f"""
// ============================================================================
// COMPREHENSIVE HASHCAT RULES KERNEL – WITH RULE CHAIN SUPPORT
// ============================================================================

#define MAX_WORD_LEN {MAX_WORD_LEN}
#define MAX_OUTPUT_LEN {MAX_OUTPUT_LEN}
#define MAX_RULE_LEN {MAX_RULE_LEN}
#define RULE_HASH_TABLE_MASK {rule_hash_table_mask}
#define RULE_HASH_TABLE_SIZE (RULE_HASH_TABLE_MASK + 1U)
#define CRACKED_HASH_TABLE_MASK {cracked_hash_table_mask}

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

ulong fnv1a_hash_64(const unsigned char* data, unsigned int len) {{
    ulong hash = 14695981039346656037UL;
    for (unsigned int i = 0; i < len; i++) {{
        hash ^= (ulong)data[i];
        hash *= 1099511628211UL;
    }}
    return hash;
}}

ulong mix64(ulong x) {{
    x += 0x9E3779B97F4A7C15UL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
    return x ^ (x >> 31);
}}

// Per-rule exact-fingerprint insertion. states: 0=empty, 1=writer, 2=ready.
// A separate state array keeps all uint64 fingerprints, including 0, valid.
inline int insert_rule_fingerprint(__global ulong* hashes, __global uint* states,
                                   uint table_base, ulong key) {{
    uint slot = (uint)(mix64(key) & (ulong)RULE_HASH_TABLE_MASK);
    for (uint probe = 0; probe <= RULE_HASH_TABLE_MASK; probe++) {{
        uint absolute = table_base + slot;
        __global volatile uint* state_ptr = (__global volatile uint*)&states[absolute];
        uint state = atomic_cmpxchg(state_ptr, 0U, 1U);
        if (state == 0U) {{
            hashes[absolute] = key;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            atomic_xchg(state_ptr, 2U);
            return 1;
        }}
        if (state == 1U) {{
            do {{
                state = *state_ptr;
            }} while (state == 1U);
        }}
        if (state == 2U && hashes[absolute] == key) return 0;
        slot = (slot + 1U) & RULE_HASH_TABLE_MASK;
    }}
    return 0; // unreachable when table capacity is >= 2x sample size
}}

inline int lookup_cracked_fingerprint(__global const ulong* hashes,
                                      __global const uint* states, ulong key) {{
    uint slot = (uint)(mix64(key) & (ulong)CRACKED_HASH_TABLE_MASK);
    for (uint probe = 0; probe <= CRACKED_HASH_TABLE_MASK; probe++) {{
        uint state = states[slot];
        if (state == 0U) return 0;
        if (state == 2U && hashes[slot] == key) return 1;
        slot = (slot + 1U) & CRACKED_HASH_TABLE_MASK;
    }}
    return 0;
}}

// ----------------------------------------------------------------------------
// Operation helpers (take input buffer, output buffer, lengths, and arguments)
// ----------------------------------------------------------------------------
static int duplicate_front(const unsigned char* in, int in_len,
                            unsigned char* out, int* out_len, int* changed, int n) {{
    // BUG FIX: 'y' operator prepends first N chars to the FRONT of the word.
    // Old code appended them to the back (which is what 'Y'/duplicate_back does).
    // Correct: out = in[0..n) + in[0..in_len)
    if (n > in_len) n = in_len;
    int new_len = in_len + n;
    if (new_len > MAX_OUTPUT_LEN) return -2;
    for (int i = 0; i < n; i++) out[i] = in[i];
    for (int i = 0; i < in_len; i++) out[n + i] = in[i];
    *out_len = new_len;
    *changed = 1;
    return 0;
}}

static int duplicate_back(const unsigned char* in, int in_len,
                           unsigned char* out, int* out_len, int* changed, int n) {{
    if (n > in_len) n = in_len;
    int new_len = in_len + n;
    if (new_len > MAX_OUTPUT_LEN) return -2;
    for (int i = 0; i < in_len; i++) out[i] = in[i];
    for (int i = 0; i < n; i++) out[in_len + i] = in[in_len - n + i];
    *out_len = new_len;
    *changed = 1;
    return 0;
}}

static int duplicate_word(const unsigned char* in, int in_len,
                           unsigned char* out, int* out_len, int* changed, int times) {{
    int new_len = in_len * (times + 1);
    if (new_len > MAX_OUTPUT_LEN) return -2;
    for (int rep = 0; rep <= times; rep++) {{
        for (int i = 0; i < in_len; i++) {{
            out[rep * in_len + i] = in[i];
        }}
    }}
    *out_len = new_len;
    *changed = 1;
    return 0;
}}

static void rotate_left(const unsigned char* in, int in_len,
                        unsigned char* out, int* out_len, int* changed, int n) {{
    if (in_len <= 0) {{ *out_len = 0; *changed = 0; return; }}
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
    if (in_len <= 0) {{ *out_len = 0; *changed = 0; return; }}
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
                if (in_len * 2 > MAX_OUTPUT_LEN) return -2;
                {{
                    *out_len = in_len * 2;
                    for (int i = 0; i < in_len; i++) {{
                        out[i] = in[i];
                        out[in_len + i] = in[i];
                    }}
                    changed = 1;
                }}
                break;
            case 'f':
                if (in_len * 2 > MAX_OUTPUT_LEN) return -2;
                {{
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
                if (in_len * 2 > MAX_OUTPUT_LEN) return -2;
                {{
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
                if (in_len + 1 > MAX_OUTPUT_LEN) return -2;
                {{
                    out[0] = arg;
                    for (int i = 0; i < in_len; i++) out[i+1] = in[i];
                    *out_len = in_len + 1;
                    changed = 1;
                }}
                break;
            case '$':
                if (in_len + 1 > MAX_OUTPUT_LEN) return -2;
                {{
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
                if (n >= 0) return duplicate_front(in, in_len, out, out_len, &changed, n);
                break;
            case 'Y':
                if (n >= 0) return duplicate_back(in, in_len, out, out_len, &changed, n);
                break;
            case 'z':
                if (n > 0) {{
                    if (in_len + n > MAX_OUTPUT_LEN) return -2;
                    out[0] = in[0];
                    for (int i = 0; i < n; i++) out[i+1] = in[0];
                    for (int i = 1; i < in_len; i++) out[n + i] = in[i];
                    *out_len = in_len + n;
                    changed = 1;
                }}
                break;
            case 'Z':
                if (n > 0) {{
                    if (in_len + n > MAX_OUTPUT_LEN) return -2;
                    for (int i = 0; i < in_len; i++) out[i] = in[i];
                    for (int i = 0; i < n; i++) out[in_len + i] = in[in_len-1];
                    *out_len = in_len + n;
                    changed = 1;
                }}
                break;
            case 'p':
                if (n >= 0) return duplicate_word(in, in_len, out, out_len, &changed, n);
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
                if (n1 >= 0) {{
                    if (in_len + 1 > MAX_OUTPUT_LEN) return -2;
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
        if (result < 0) {{
            // Reject or overflow: the entire rule is invalid for this word.
            // Never continue the rule chain with an empty intermediate.
            *out_len = 0;
            *changed = result;
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
// Independent per-rule scoring kernel. Every rule owns a disjoint hash table,
// so duplicates produced by rule A can never suppress a unique result of rule B.
// Effectiveness is counted only on the first unique output for that rule/sample.
// ----------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void ranker_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned char* rules_in,
    __global const ulong* cracked_hash_table,
    __global const uint* cracked_hash_states,
    __global ulong* rule_hash_tables,
    __global uint* rule_hash_states,
    __global uint* rule_uniqueness_counts,
    __global uint* rule_effectiveness_counts,
    const unsigned int num_words,
    const unsigned int num_rules_in_batch,
    const unsigned int max_word_len,
    const unsigned int rule_hash_table_mask)
{{
    unsigned int global_id = get_global_id(0);
    unsigned int total = num_words * num_rules_in_batch;
    if (global_id >= total) return;

    // Work is laid out rule-major so each rule's hash table has good spatial locality.
    unsigned int rule_idx = global_id / num_words;
    unsigned int word_idx = global_id % num_words;
    unsigned int table_base = rule_idx * (rule_hash_table_mask + 1U);

    unsigned char word[MAX_WORD_LEN];
    unsigned int word_len = 0;
    for (unsigned int i = 0; i < max_word_len; i++) {{
        unsigned char c = base_words_in[word_idx * max_word_len + i];
        if (c == 0) break;
        word[i] = c;
        word_len++;
    }}

    unsigned int rule_start = rule_idx * MAX_RULE_LEN;
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
        ulong word_hash = fnv1a_hash_64(result_temp, out_len);
        int is_new = insert_rule_fingerprint(rule_hash_tables, rule_hash_states, table_base, word_hash);
        if (is_new) {{
            atomic_inc(&rule_uniqueness_counts[rule_idx]);
            if (lookup_cracked_fingerprint(cracked_hash_table, cracked_hash_states, word_hash)) {{
                atomic_inc(&rule_effectiveness_counts[rule_idx]);
            }}
        }}
    }}
}}
"""

# ====================================================================
# --- INDEPENDENT 64-BIT GPU SCORER ---------------------------------
# ====================================================================
def _ceil_log2(value: int) -> int:
    value = max(1, int(value))
    return max(1, int(math.ceil(math.log2(value))))


def _read_stratified_word_sample(wordlist_path, max_len, sample_words, seed):
    """Read a small, repeatable stratified sample without loading the file.

    Each trial draws line windows from byte ranges spread across the file.
    This gives MAB fresh evidence on every trial while keeping I/O proportional
    to the requested sample rather than to the entire wordlist.
    """
    sample_words = max(1, int(sample_words))
    file_size = os.path.getsize(wordlist_path)
    if file_size == 0:
        return np.zeros((0, max_len), dtype=np.uint8), 0

    segments = min(8, max(1, sample_words // 512))
    per_segment = int(math.ceil(sample_words / segments))
    rng = np.random.default_rng(np.uint64(seed))
    out = np.zeros((sample_words, max_len), dtype=np.uint8)
    count = 0

    with open(wordlist_path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for seg in range(segments):
            if count >= sample_words:
                break
            lo = (file_size * seg) // segments
            hi = (file_size * (seg + 1)) // segments
            if hi <= lo:
                continue
            pos = int(rng.integers(lo, hi))
            if pos > 0:
                nl = mm.find(b'\n', pos)
                if nl == -1:
                    pos = file_size
                else:
                    pos = nl + 1
            local = 0
            while pos < file_size and local < per_segment and count < sample_words:
                end = mm.find(b'\n', pos)
                if end == -1:
                    end = file_size
                line = mm[pos:end].strip()
                pos = end + 1
                if not line or len(line) > max_len:
                    continue
                out[count, :len(line)] = np.frombuffer(line, dtype=np.uint8)
                count += 1
                local += 1

    if count == 0:
        return np.zeros((0, max_len), dtype=np.uint8), 0
    return out[:count], count


class IndependentRuleGpuScorer:
    """GPU scorer with independent per-rule 64-bit fingerprint sets.

    The old scorer used one shared bitmap for every rule in a dispatch. That
    made uniqueness dependent on GPU race order: the first rule to claim a
    bit received the credit. This class allocates a disjoint open-addressed
    fingerprint table for every rule row in the dispatch, so rule scores are
    statistically independent of one another.
    """

    def __init__(self, device_id, words_capacity, max_rules, cracked_hashes):
        self.words_capacity = max(1, int(words_capacity))
        self.requested_max_rules = max(1, int(max_rules))
        self.cracked_hashes = np.unique(np.asarray(cracked_hashes, dtype=np.uint64))

        platform, device = select_platform_and_device(device_id) if device_id is not None else select_platform_and_device()
        self.platform = platform
        self.device = device
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)
        _total_vram, available_vram = get_gpu_memory_info(device)
        self.available_vram = int(available_vram)

        self.rule_table_size = table_size_for_count(self.words_capacity)
        self.rule_table_bits = _ceil_log2(self.rule_table_size)
        self.cracked_table_size = table_size_for_count(len(self.cracked_hashes))
        self.cracked_table_bits = _ceil_log2(self.cracked_table_size)

        per_rule_table_bytes = self.rule_table_size * (np.dtype(np.uint64).itemsize + np.dtype(np.uint32).itemsize)
        memory_budget = max(16 * 1024 * 1024, int(self.available_vram * 0.12))
        memory_limited_rules = max(1, memory_budget // max(1, per_rule_table_bytes))
        self.max_rules = min(self.requested_max_rules, int(memory_limited_rules), MAX_RULES_IN_BATCH)
        self.max_rules = max(1, self.max_rules)

        # Avoid dispatches that exceed driver watchdog/work-item limits.
        dispatch_limited = max(1, MAX_DISPATCH_ITEMS // self.words_capacity)
        self.max_rules = min(self.max_rules, dispatch_limited)

        src = get_kernel_source(self.rule_table_bits, self.cracked_table_bits)
        self.program = cl.Program(self.context, src).build()
        self.kernel_ranker = self.program.ranker_kernel

        cracked_table, cracked_states = build_open_addressing_table_uint64(
            self.cracked_hashes, self.cracked_table_size)
        mf = cl.mem_flags
        self.cracked_hash_table_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cracked_table)
        self.cracked_hash_states_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cracked_states)

        self.base_words_g = cl.Buffer(self.context, mf.READ_ONLY,
                                      self.words_capacity * MAX_WORD_LEN * np.dtype(np.uint8).itemsize)
        self.rules_g = cl.Buffer(self.context, mf.READ_ONLY,
                                 self.max_rules * MAX_RULE_LEN * np.dtype(np.uint8).itemsize)
        self.rule_table_hashes_g = cl.Buffer(
            self.context, mf.READ_WRITE,
            self.max_rules * self.rule_table_size * np.dtype(np.uint64).itemsize)
        self.rule_table_states_g = cl.Buffer(
            self.context, mf.READ_WRITE,
            self.max_rules * self.rule_table_size * np.dtype(np.uint32).itemsize)
        self.uniqueness_g = cl.Buffer(self.context, mf.READ_WRITE,
                                      self.max_rules * np.dtype(np.uint32).itemsize)
        self.effectiveness_g = cl.Buffer(self.context, mf.READ_WRITE,
                                         self.max_rules * np.dtype(np.uint32).itemsize)

    def score(self, words_np, rules_encoded):
        """Return (unique_per_rule, cracked_per_rule) for one sample/batch."""
        words_np = np.ascontiguousarray(words_np, dtype=np.uint8)
        num_words = int(words_np.shape[0])
        if num_words <= 0:
            return np.zeros(len(rules_encoded), dtype=np.uint64), np.zeros(len(rules_encoded), dtype=np.uint64)
        if words_np.shape[1] != MAX_WORD_LEN:
            raise ValueError(f"word sample has width {words_np.shape[1]}, expected {MAX_WORD_LEN}")

        n_rules_total = len(rules_encoded)
        unique = np.zeros(n_rules_total, dtype=np.uint64)
        cracked = np.zeros(n_rules_total, dtype=np.uint64)

        cl.enqueue_copy(self.queue, self.base_words_g, words_np).wait()
        for start in range(0, n_rules_total, self.max_rules):
            end = min(start + self.max_rules, n_rules_total)
            chunk = np.ascontiguousarray(rules_encoded[start:end], dtype=np.uint8)
            n_rules = end - start
            cl.enqueue_copy(self.queue, self.rules_g, chunk).wait()
            cl.enqueue_fill_buffer(self.queue, self.rule_table_states_g, np.uint32(0),
                                   0, self.max_rules * self.rule_table_size * np.dtype(np.uint32).itemsize).wait()
            cl.enqueue_fill_buffer(self.queue, self.uniqueness_g, np.uint32(0),
                                   0, self.max_rules * np.dtype(np.uint32).itemsize).wait()
            cl.enqueue_fill_buffer(self.queue, self.effectiveness_g, np.uint32(0),
                                   0, self.max_rules * np.dtype(np.uint32).itemsize).wait()

            items = num_words * n_rules
            global_size = (int(math.ceil(items / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
            self.kernel_ranker(
                self.queue, global_size, (LOCAL_WORK_SIZE,),
                self.base_words_g, self.rules_g,
                self.cracked_hash_table_g, self.cracked_hash_states_g,
                self.rule_table_hashes_g, self.rule_table_states_g,
                self.uniqueness_g, self.effectiveness_g,
                np.uint32(num_words), np.uint32(n_rules),
                np.uint32(MAX_WORD_LEN), np.uint32(self.rule_table_size - 1),
            ).wait()
            u = np.zeros(self.max_rules, dtype=np.uint32)
            e = np.zeros(self.max_rules, dtype=np.uint32)
            cl.enqueue_copy(self.queue, u, self.uniqueness_g).wait()
            cl.enqueue_copy(self.queue, e, self.effectiveness_g).wait()
            unique[start:end] = u[:n_rules]
            cracked[start:end] = e[:n_rules]

        return unique, cracked


def _encode_rules_matrix(rules_list):
    encoded = np.zeros((len(rules_list), MAX_RULE_LEN), dtype=np.uint8)
    for i, rule in enumerate(rules_list):
        rb = rule['rule_data'].encode('latin-1')
        if len(rb) > MAX_RULE_LEN:
            raise ValueError(f"rule exceeds MAX_RULE_LEN={MAX_RULE_LEN}: {rule['rule_data']!r}")
        encoded[i, :len(rb)] = np.frombuffer(rb, dtype=np.uint8)
    return encoded


# ====================================================================
# --- EXHAUSTIVE RANKING (Legacy v3.2) ---
# ====================================================================
def rank_rules_exhaustive(wordlist_path, rules_path, cracked_list_path, ranking_output_path, top_k,
                          words_per_gpu_batch=None, global_hash_map_bits=None, cracked_hash_map_bits=None,
                          preset=None, device_id=None):
    """Full-pass ranking with independent per-rule 64-bit fingerprint sets.

    Legacy mode remains available as a reference path.  Uniqueness is measured
    independently inside each streamed word batch; batches are deliberately
    kept bounded so VRAM use does not scale with the entire wordlist.
    """
    start_time = time()
    total_words = estimate_word_count(wordlist_path)
    rules_list = load_rules(rules_path)
    setup_interrupt_handler(rules_list, ranking_output_path, top_k)
    cracked_hashes_np = load_cracked_hashes(cracked_list_path, MAX_WORD_LEN)

    if not rules_list:
        save_ranking_data([], ranking_output_path, legacy=True)
        return

    words_per_gpu_batch = int(words_per_gpu_batch or DEFAULT_WORDS_PER_GPU_BATCH)
    if words_per_gpu_batch < 1:
        raise ValueError("--batch-size must be positive")

    print(f"\n{blue('Dataset Summary:')}")
    print(f"   {bold('Words (estimated):')} {cyan(f'{total_words:,}')}")
    print(f"   {bold('Rules:')} {cyan(f'{len(rules_list):,}')}")
    print(f"   {bold('Cracked fingerprints:')} {cyan(f'{len(cracked_hashes_np):,}')}")

    try:
        scorer = IndependentRuleGpuScorer(
            device_id=device_id,
            words_capacity=words_per_gpu_batch,
            max_rules=MAX_RULES_IN_BATCH,
            cracked_hashes=cracked_hashes_np,
        )
        print(f"{green('GPU:')} {cyan(scorer.device.name.strip())}")
        print(f"{blue('Platform:')} {cyan(scorer.platform.name.strip())}")
        print(f"{blue('Independent scorer:')} {cyan(f'{scorer.max_rules} rules/dispatch group')} | "
              f"{cyan(f'{scorer.rule_table_size:,} slots/rule')} | {cyan('FNV-1a-64')}")
    except Exception as e:
        print(f"{red('OpenCL initialization failed:')} {e}")
        return

    encoded_rules = _encode_rules_matrix(rules_list)
    uniqueness = np.zeros(len(rules_list), dtype=np.uint64)
    effectiveness = np.zeros(len(rules_list), dtype=np.uint64)
    processed_rule_words = 0

    word_pbar = tqdm(total=total_words, desc="Processing words", unit="words",
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    try:
        for words_np, _hashes_np, num_words in optimized_wordlist_iterator(
                wordlist_path, MAX_WORD_LEN, words_per_gpu_batch):
            if interrupted:
                break
            # ``optimized_wordlist_iterator`` returns a 2-D word matrix.  Keep
            # a defensive flat-buffer fallback for callers using an older cached
            # iterator implementation, so legacy mode cannot regress on shape.
            if words_np.ndim == 2:
                word_matrix = words_np[:num_words]
            else:
                expected = int(num_words) * MAX_WORD_LEN
                if words_np.size < expected:
                    raise ValueError(
                        f"word batch buffer has {words_np.size} bytes, "
                        f"but {num_words} words require {expected} bytes"
                    )
                word_matrix = words_np[:expected].reshape(num_words, MAX_WORD_LEN)
            u, e = scorer.score(word_matrix, encoded_rules)
            uniqueness += u
            effectiveness += e
            processed_rule_words += int(num_words) * len(rules_list)
            word_pbar.update(num_words)
    finally:
        word_pbar.close()

    for idx, rule in enumerate(rules_list):
        rule['uniqueness_score'] = int(uniqueness[idx])
        rule['effectiveness_score'] = int(effectiveness[idx])
        rule['combined_score'] = rule['effectiveness_score'] * 10 + rule['uniqueness_score']
        rule['eliminated'] = False
        rule['mab_trials'] = 0
        rule['selections'] = 0
        rule['mab_success_prob'] = 0.0

    end_time = time()
    print(f"\n{green('EXHAUSTIVE RANKING COMPLETE')}")
    print(f"{blue('Rule-word evaluations:')} {cyan(f'{processed_rule_words:,}')}")
    print(f"{blue('Execution Time:')} {cyan(f'{end_time - start_time:.2f} s')}")
    csv_path = save_ranking_data(rules_list, ranking_output_path, legacy=True)
    if top_k > 0 and csv_path:
        save_top_k_rules(rules_list, os.path.splitext(ranking_output_path)[0] + "_optimized.rule", top_k)


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
            sort_key = ((self.trials[needs_arr].astype(np.int64) << 32)
                        - self.selection_count[needs_arr].astype(np.int64))  # BUG FIX: added parens around << 32 to prevent - from binding before <
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
        # Do not emit a separate terminal line here.  The MAB progress bar
        # reports the cumulative eliminated count in-place, avoiding console flood.

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
        if top_k == len(valid_indices):
            top_local = valid_indices
        else:
            partition_k = top_k - 1
            order_local = np.argpartition(-probs[valid_indices], partition_k)[:top_k]
            top_local = valid_indices[order_local]
        top_indices = active[top_local]
        order = np.argsort(-probs[top_local], kind='stable')
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
                   mab_screening_trials=None, mab_zero_success_elimination=None,
                   mab_sample_words=None):
    """MAB ranking using small stratified samples instead of full wordlist passes.

    Each MAB trial evaluates the selected rules on one fresh sample drawn from
    byte ranges distributed across the wordlist.  This changes the economics
    from ``rules × full-wordlist`` to ``trials × sample`` while preserving the
    bandit's Beta/Bernoulli update semantics.
    """
    start_time = time()
    total_words = estimate_word_count(wordlist_path)
    rules_list = load_rules(rules_path)
    setup_interrupt_handler(rules_list, ranking_output_path, top_k)
    cracked_hashes_np = load_cracked_hashes(cracked_list_path, MAX_WORD_LEN)

    if not rules_list:
        save_ranking_data([], ranking_output_path, legacy=False)
        return

    sample_words = int(mab_sample_words or 8192)
    sample_words = max(256, sample_words)
    # Keep the default sample comfortably below ordinary GPU batches, but allow
    # users to explicitly request a larger/smaller sample for their hardware.
    max_rules_dispatch = MAX_RULES_IN_BATCH
    encoded_rules = _encode_rules_matrix(rules_list)

    exploration_factor = mab_exploration_factor if mab_exploration_factor is not None else 2.0
    final_trials = max(1, int(mab_final_trials if mab_final_trials is not None else 50))
    screening_trials = max(1, int(mab_screening_trials if mab_screening_trials is not None else 5))
    zero_elim = mab_zero_success_elimination if mab_zero_success_elimination is not None else True
    rule_bandit = MultiPassMAB(
        rules_list, exploration_factor, final_trials, screening_trials, zero_elim,
        batch_size_words=sample_words,
    )

    print(f"\n{blue('Dataset Summary:')}")
    print(f"   {bold('Words (estimated):')} {cyan(f'{total_words:,}')}")
    print(f"   {bold('Rules:')} {cyan(f'{len(rules_list):,}')}")
    print(f"   {bold('Cracked fingerprints:')} {cyan(f'{len(cracked_hashes_np):,}')}")
    print(f"   {bold('MAB sample size:')} {cyan(f'{sample_words:,} words/trial')} (fresh stratified sample)")

    try:
        scorer = IndependentRuleGpuScorer(
            device_id=device_id,
            words_capacity=sample_words,
            max_rules=max_rules_dispatch,
            cracked_hashes=cracked_hashes_np,
        )
        print(f"{green('GPU:')} {cyan(scorer.device.name.strip())}")
        print(f"{blue('Platform:')} {cyan(scorer.platform.name.strip())}")
        print(f"{blue('Independent scorer:')} {cyan(f'{scorer.max_rules} rules/dispatch group')} | "
              f"{cyan(f'{scorer.rule_table_size:,} slots/rule')} | {cyan('FNV-1a-64')}")
    except Exception as e:
        print(f"{red('OpenCL initialization failed:')} {e}")
        return

    words_processed_total = 0
    total_unique_found = 0
    total_cracked_found = 0
    iteration = 0
    # Keep MAB progress on a single in-place terminal line.  Frequent
    # per-trial prints flood some terminals/IDE consoles, while tqdm gives us
    # elapsed time, ETA and throughput without spamming new lines.
    progress_enabled = bool(sys.stderr.isatty() or sys.stdout.isatty())

    def _estimated_remaining_iterations():
        active_arr = rule_bandit._get_active_array()
        if len(active_arr) == 0:
            return 0
        remaining_rule_trials = int(np.maximum(
            0, rule_bandit.final_trials - rule_bandit.trials[active_arr]
        ).sum())
        return int(math.ceil(remaining_rule_trials / max(MAX_RULES_IN_BATCH, 1)))

    initial_total = max(1, _estimated_remaining_iterations())
    pbar = tqdm(
        total=initial_total,
        desc="MAB sampling",
        unit="sample",
        position=0,
        leave=True,
        dynamic_ncols=True,
        mininterval=0.75,
        smoothing=0.10,
        file=sys.stderr,
        disable=not progress_enabled,
        bar_format=(
            "{desc}: {n_fmt}/{total_fmt} {bar:30} "
            "| {elapsed} | ETA {remaining} | {rate_fmt} | {postfix}"
        ),
    )

    while not interrupted and rule_bandit.active_rules:
        selected_indices = rule_bandit.select_rules(batch_size=MAX_RULES_IN_BATCH, iteration=iteration)
        if not selected_indices:
            break

        # One fresh sample is intentionally shared by all rules selected in
        # this MAB iteration, making their trial observations comparable.
        words_np, num_words = _read_stratified_word_sample(
            wordlist_path, MAX_WORD_LEN, sample_words,
            seed=(0x9E3779B9 + iteration * 0x85EBCA6B) & 0xFFFFFFFFFFFFFFFF,
        )
        if num_words == 0:
            break

        encoded_selected = encoded_rules[np.asarray(selected_indices, dtype=np.int32)]
        u_arr, e_arr = scorer.score(words_np, encoded_selected)
        rule_bandit.update(selected_indices, e_arr.astype(np.uint32), num_words)

        for i, idx in enumerate(selected_indices):
            rule_bandit.all_rules[idx]['uniqueness_score'] = rule_bandit.all_rules[idx].get('uniqueness_score', 0) + int(u_arr[i])
            rule_bandit.all_rules[idx]['effectiveness_score'] = rule_bandit.all_rules[idx].get('effectiveness_score', 0) + int(e_arr[i])
            rule_bandit.all_rules[idx]['total_successes'] = rule_bandit.all_rules[idx].get('total_successes', 0) + int(e_arr[i])
            rule_bandit.all_rules[idx]['total_trials'] = rule_bandit.all_rules[idx].get('total_trials', 0) + int(num_words)
            rule_bandit.all_rules[idx]['times_tested'] = rule_bandit.all_rules[idx].get('times_tested', 0) + 1

        total_unique_found += int(np.sum(u_arr))
        total_cracked_found += int(np.sum(e_arr))
        words_processed_total += int(num_words) * len(selected_indices)
        pbar.update(1)

        stats = rule_bandit.get_statistics()
        # Re-estimate remaining work after eliminations. This keeps the ETA
        # meaningful even when early elimination removes large rule batches.
        estimated_remaining = _estimated_remaining_iterations()
        pbar.total = max(pbar.n, pbar.n + estimated_remaining)
        pbar.set_postfix_str(
            f"active={stats['active_rules']:,} | "
            f"screen={stats['rules_needing_screening']:,} | "
            f"final={stats['rules_needing_final']:,} | "
            f"elim={stats['eliminated_rules']:,} | sample={num_words:,}",
            refresh=False,
        )
        iteration += 1

        if stats['rules_needing_screening'] == 0 and stats['rules_needing_final'] == 0:
            break

    pbar.close()

    final_stats = rule_bandit.get_statistics()
    for rule in rules_list:
        idx = rule['rule_id']
        rule['eliminated'] = idx in rule_bandit.eliminated_rules
        rule['eliminate_reason'] = 'low_success_rate' if rule['eliminated'] else ''
        rule['mab_trials'] = int(rule_bandit.trials[idx])
        rule['selections'] = int(rule_bandit.selection_count[idx])
        denom = (rule_bandit.successes[idx] + rule_bandit.failures[idx] - 2.0)
        rule['mab_success_prob'] = float((rule_bandit.successes[idx] - 1.0) / denom) if denom > 0 else 0.0
        rule['combined_score'] = (
            rule.get('effectiveness_score', 0) * 10
            + rule.get('uniqueness_score', 0)
            + rule['mab_success_prob'] * 1000
        )

    end_time = time()
    print(f"\n{green('=' * 80)}")
    print(f"{bold('SAMPLE-BASED MAB RANKING COMPLETE')}")
    print(f"{green('=' * 80)}")
    print(f"{blue('Rule-word evaluations:')} {cyan(f'{words_processed_total:,}')}")
    print(f"{blue('Unique outputs counted:')} {cyan(f'{total_unique_found:,}')}")
    print(f"{blue('Cracked outputs counted:')} {cyan(f'{total_cracked_found:,}')}")
    print(f"{blue('Execution Time:')} {cyan(f'{end_time - start_time:.2f} s')}")
    print(f"{blue('Surviving rules:')} {cyan(str(final_stats['active_rules']))}")
    print(f"{blue('Eliminated rules:')} {cyan(str(final_stats['eliminated_rules']))} ({final_stats['eliminated_percentage']:.1f}%)")

    csv_path = save_ranking_data(rules_list, ranking_output_path, legacy=False)
    if top_k > 0 and csv_path:
        save_top_k_rules(rules_list, os.path.splitext(ranking_output_path)[0] + "_optimized.rule", top_k)


# ====================================================================
# --- MAIN ENTRY POINT ---
# ====================================================================
def build_arg_parser():
    """Builds the argparse.ArgumentParser used by main(). Factored out
    so run_ranker.py / tests can introspect it without invoking main()."""
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
    parser.add_argument('--mab-screening-trials', type=int, default=5, help='Sample trials before elimination')
    parser.add_argument('--mab-sample-words', type=int, default=8192, help='Words sampled from stratified wordlist slices per MAB trial')
    parser.add_argument('--mab-no-zero-eliminate', action='store_false', dest='mab_zero_success_elimination',
                        help='Disable zero‑success elimination')

    # Legacy mode
    parser.add_argument('--legacy', action='store_true', help='Run exhaustive (v3.2) mode instead of MAB')

    # Device selection
    parser.add_argument('--device', type=int, help='OpenCL device ID')
    parser.add_argument('--list-devices', action='store_true', help='List available devices and exit')
    return parser


def main(argv=None):
    """Callable entry point (was previously only reachable via the
    `if __name__ == '__main__':` block). Accepts an optional argv list
    for programmatic/test invocation; defaults to sys.argv[1:]."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        list_platforms_and_devices()
        sys.exit(0)

    print(f"{green('=' * 80)}")
    print(f"{bold('HASHCAT RULE RANKER v6.0')}")
    if args.legacy:
        print(f"{bold('LEGACY MODE (v3.2) – Exhaustive Ranking')}")
    else:
        print(f"{bold('MULTI-PASS MAB MODE – Early Elimination')}")
    print(f"{green('=' * 80)}")
    print(f"{blue('GPU Rules:')} Validated Hashcat-compatible subset with {MAX_RULE_LEN}‑char support")
    print(f"{blue('Fingerprinting:')} FNV-1a-64 with independent per-rule open-addressing tables")
    print(f"{blue('Validation:')} Rules are filtered using rulest’s HashcatRuleValidator (banned ops excluded)")
    print(f"{blue('MAB sampling:')} fresh stratified wordlist samples per trial")
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
            mab_zero_success_elimination=args.mab_zero_success_elimination,
            mab_sample_words=args.mab_sample_words
        )


if __name__ == '__main__':
    main()
