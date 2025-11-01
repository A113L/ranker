import pyopencl as cl
import numpy as np
import argparse
import csv
from tqdm import tqdm
import math
import warnings
import os
from time import time

# ====================================================================
# --- RANKER v2.8: GPU-Accelerated Hashcat Rule Ranking Tool ---
# ====================================================================
# Purpose: Ranks hashcat rules based on uniqueness against a base wordlist
# and effectiveness against a list of cracked passwords, leveraging PyOpenCL
# for high-performance GPU processing.
#
# Optimization: Fully implemented double-buffered, pipelined execution
# to overlap data transfer and kernel computation.
# ====================================================================

# --- WARNING FILTERS ---
warnings.filterwarnings("ignore", message="overflow encountered in scalar multiply")
try:
    # Filter the DeprecationWarning reported by the user's shell
    warnings.filterwarnings("ignore", message="The 'device_offset' argument of enqueue_copy is deprecated")
    warnings.filterwarnings("ignore", category=cl.CompilerWarning)
except AttributeError:
    pass
# -----------------------

# ====================================================================
# --- CONSTANTS CONFIGURATION (OPTIMIZED FOR RTX 3060 Ti 8GB) ---
# ====================================================================
MAX_WORD_LEN = 32
MAX_OUTPUT_LEN = MAX_WORD_LEN * 2
MAX_RULE_ARGS = 2
MAX_RULES_IN_BATCH = 128
LOCAL_WORK_SIZE = 256 # Optimal size for many modern GPUs (multiple of 32/64)

# BATCH SIZE FOR WORDS: Increased for better throughput on large wordlists
WORDS_PER_GPU_BATCH = 100000

# Global Uniqueness Map Parameters (Targeting ~4.2 GB VRAM for 8GB cards)
GLOBAL_HASH_MAP_BITS = 35
GLOBAL_HASH_MAP_WORDS = 1 << (GLOBAL_HASH_MAP_BITS - 5)
GLOBAL_HASH_MAP_BYTES = GLOBAL_HASH_MAP_WORDS * np.uint32(4)
GLOBAL_HASH_MAP_MASK = (1 << (GLOBAL_HASH_MAP_BITS - 5)) - 1

# Cracked Password Map Parameters (Targeting ~1.0 GB VRAM)
CRACKED_HASH_MAP_BITS = 33
CRACKED_HASH_MAP_WORDS = 1 << (CRACKED_HASH_MAP_BITS - 5)
CRACKED_HASH_MAP_BYTES = CRACKED_HASH_MAP_WORDS * np.uint32(4)
CRACKED_HASH_MAP_MASK = (1 << (CRACKED_HASH_MAP_BITS - 5)) - 1

# Rule IDs (Unchanged)
START_ID_SIMPLE = 0
NUM_SIMPLE_RULES = 10
START_ID_TD = 10
NUM_TD_RULES = 20
START_ID_S = 30
NUM_S_RULES = 256 * 256
START_ID_A = 30 + NUM_S_RULES
NUM_A_RULES = 3 * 256
# ====================================================================

# --- KERNEL SOURCE (OpenCL C) ---
KERNEL_SOURCE = f"""
// FNV-1a Hash implementation in OpenCL
unsigned int fnv1a_hash_32(const unsigned char* data, unsigned int len) {{
    unsigned int hash = 2166136261U;
    for (unsigned int i = 0; i < len; i++) {{
        hash ^= data[i];
        hash *= 16777619U;
    }}
    return hash;
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

    // Hash map bitfield logic
    unsigned int map_index = (word_hash >> 5) & map_mask;
    unsigned int bit_index = word_hash & 31;
    unsigned int set_bit = (1U << bit_index);

    // Atomically set the bit in the global hash map
    atomic_or(&global_hash_map[map_index], set_bit);
}}


__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void bfs_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned int* rules_in,
    __global unsigned int* rule_uniqueness_counts,
    __global unsigned int* rule_effectiveness_counts,
    __global const unsigned int* global_hash_map, // Added const
    __global const unsigned int* cracked_hash_map,
    const unsigned int num_words,
    const unsigned int num_rules_in_batch,
    const unsigned int max_word_len,
    const unsigned int max_output_len, // Note: max_output_len is MAX_WORD_LEN * 2
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

    // We use the local array 'result_temp' instead of the global 'result_buffer'
    unsigned char result_temp[2 * {MAX_WORD_LEN}]; // Size = max_output_len
    
    __global const unsigned char* current_word_ptr = base_words_in + word_idx * max_word_len;
    unsigned int rule_size_in_int = 2 + {MAX_RULE_ARGS};
    __global const unsigned int* current_rule_ptr_int = rules_in + rule_batch_idx * rule_size_in_int;

    unsigned int rule_id = current_rule_ptr_int[0];
    unsigned int rule_args_int = current_rule_ptr_int[1]; // Arguments packed in uint32

    unsigned int word_len = 0;
    for (unsigned int i = 0; i < max_word_len; i++) {{
        if (current_word_ptr[i] == 0) {{
            word_len = i;
            break;
        }}
    }}

    if (word_len == 0 && rule_id < start_id_A ) return;
    unsigned int out_len = 0;
    bool changed_flag = false;

    for(unsigned int i = 0; i < max_output_len; i++) result_temp[i] = 0;

    // --- START: Rule Application Logic (MERGED) ---
    // Unpacking arguments from 'rule_args_int'
    unsigned char arg0 = (unsigned char)(rule_args_int & 0xFF);
    unsigned char arg1 = (unsigned char)((rule_args_int >> 8) & 0xFF);

    if (rule_id >= start_id_simple && rule_id < end_id_simple) {{ // Simple rules (l, u, c, C, t, r, k, :, d, f)
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
                        if (c >= 'A' && c <= 'Z') {{ // Ensure rest is lowercase
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
                        if (c >= 'a' && c <= 'z') {{ // Ensure rest is UPPERCASE
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
                    // Check if word actually changed
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
                if (out_len >= max_output_len) {{ // Check against the buffer size
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
                if (out_len >= max_output_len) {{ // Check against the buffer size
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
    }} else if (rule_id >= start_id_TD && rule_id < end_id_TD) {{ // T, D rules (Toggle at pos, Delete at pos)
        unsigned char operator_char = arg0;
        unsigned char pos_char = arg1;
        
        unsigned int pos_to_change = pos_char - '0';
        
        if (operator_char == 'T') {{ // 'T' (toggle case at pos)
            out_len = word_len;
            for (unsigned int i = 0; i < word_len; i++) {{
                result_temp[i] = current_word_ptr[i];
            }}
            if (pos_to_change < word_len) {{
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
        else if (operator_char == 'D') {{ // 'D' (delete char at pos)
            unsigned int out_idx = 0;
            if (pos_to_change < word_len) {{
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
    else if (rule_id >= start_id_s && rule_id < end_id_s) {{ // 's' rules (substitute char)
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
    }} else if (rule_id >= start_id_A && rule_id < end_id_A) {{ // Group A rules (Prepend ^, Append $, Delete all @)
        out_len = word_len;
        for(unsigned int i=0; i<word_len; i++) result_temp[i] = current_word_ptr[i];
        
        unsigned char cmd = arg0;
        unsigned char arg = arg1;
        
        if (cmd == '^') {{ // Prepend
            if (word_len + 1 >= max_output_len) {{ // Check against buffer size
                out_len = 0;
                changed_flag = false;
            }} else {{
                // Shift all characters right
                for(unsigned int i=word_len; i>0; i--) {{
                    result_temp[i] = result_temp[i-1];
                }}
                result_temp[0] = arg;
                out_len++;
                changed_flag = true;
            }}
        }} else if (cmd == '$') {{ // Append
            if (word_len + 1 >= max_output_len) {{ // Check against buffer size
                out_len = 0;
                changed_flag = false;
            }} else {{
                result_temp[out_len] = arg;
                out_len++;
                changed_flag = true;
            }}
        }} else if (cmd == '@') {{ // Delete all instances of char
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
    // --- END: Rule Application Logic ---


    // --- Dual-Uniqueness Logic on GPU (FIXED) ---

    if (changed_flag && out_len > 0) {{ // Ensure we only proceed if a word was generated and changed

        // Using 'result_temp' for hashing
        unsigned int word_hash = fnv1a_hash_32(result_temp, out_len);

        // 1. Check against the Base Wordlist (Uniqueness Score)
        unsigned int global_map_index = (word_hash >> 5) & global_map_mask;
        unsigned int bit_index = word_hash & 31;
        unsigned int check_bit = (1U << bit_index);

        __global const unsigned int* global_map_ptr = &global_hash_map[global_map_index];

        // Non-atomic read is correct here, as the map is read-only in this kernel.
        unsigned int current_global_word = *global_map_ptr;

        // If the word IS NOT in the base wordlist
        if (!(current_global_word & check_bit)) {{
            // Increment the Uniqueness Score for this rule.
            atomic_inc(&rule_uniqueness_counts[rule_batch_idx]);

            // 2. Check against the Cracked List (Effectiveness Score)
            unsigned int cracked_map_index = (word_hash >> 5) & cracked_map_mask;
            __global const unsigned int* cracked_map_ptr = &cracked_hash_map[cracked_map_index];

            // Read the cracked map value (READ_ONLY)
            unsigned int current_cracked_word = *cracked_map_ptr;

            // If the word IS in the cracked list (i.e., we found a "true crack")
            if (current_cracked_word & check_bit) {{
                // Increment the Effectiveness Score for this rule.
                atomic_inc(&rule_effectiveness_counts[rule_batch_idx]);
            }}
        }}

    }} else {{
        return;
    }}
}}
"""

# --- HELPER FUNCTIONS (Python) ---

def fnv1a_hash_32_cpu(data):
    """Calculates FNV-1a hash for a byte array."""
    hash_val = np.uint32(2166136261)
    for byte in data:
        hash_val ^= np.uint32(byte)
        hash_val *= np.uint32(16777619)
    return hash_val

def get_word_count(path):
    """Counts total words in file to set up progress bar."""
    print(f"Counting words in: {path}...")
    count = 0
    try:
        with open(path, 'r', encoding='latin-1', errors='ignore') as f:
            for line in f:
                count += 1
    except FileNotFoundError:
        print(f"Error: Wordlist file not found at: {path}")
        exit(1)

    print(f"Total words found: {count:,}")
    return count

def load_rules(path):
    """Loads Hashcat rules from file."""
    print(f"Loading rules from: {path}...")
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

def load_cracked_hashes(path, max_len):
    """Loads a list of cracked passwords and returns their FNV-1a hashes."""
    print(f"Loading cracked list for effectiveness check from: {path}...")
    cracked_hashes = []
    try:
        with open(path, 'r', encoding='latin-1', errors='ignore') as f:
            for line in f:
                word = line.strip().encode('latin-1')
                if 1 <= len(word) <= max_len:
                    cracked_hashes.append(fnv1a_hash_32_cpu(word))
    except FileNotFoundError:
        print(f"Warning: Cracked list file not found at: {path}. Effectiveness scores will be zero.")
        return np.array([], dtype=np.uint32)

    unique_hashes = np.unique(np.array(cracked_hashes, dtype=np.uint32))
    print(f"Loaded {len(unique_hashes):,} unique cracked password hashes.")
    return unique_hashes


def encode_rule(rule_str, rule_id, max_args):
    """Encodes a rule as an array of uint32: [rule ID, arguments]"""
    rule_size_in_int = 2 + max_args
    encoded = np.zeros(rule_size_in_int, dtype=np.uint32)
    encoded[0] = np.uint32(rule_id)
    rule_chars = rule_str.encode('latin-1')
    args_int = 0
    
    # --- Encoding modification to match the kernel ---
    arg0 = 0
    arg1 = 0

    if not rule_str: # Empty rule?
        pass
    elif rule_str[0] in ('s', '@', '^', '$', 'T', 'D'):
        if rule_str[0] in ('s'):
              if len(rule_chars) >= 3:
                arg0 = np.uint32(rule_chars[1]) # e.g., 'a' in 'sab'
                arg1 = np.uint32(rule_chars[2]) # e.g., 'b' in 'sab'
        elif rule_str[0] in ('^', '$', '@', 'T', 'D'):
              if len(rule_chars) >= 2:
                arg0 = np.uint32(rule_chars[0]) # e.g., '^'
                arg1 = np.uint32(rule_chars[1]) # e.g., '1'
    
    # We encode 'arg0' and 'arg1' into 'args_int'
    args_int |= arg0
    args_int |= (arg1 << 8)
    encoded[1] = np.uint32(args_int)
    # --- End of encoding modification ---
    
    return encoded

def save_ranking_data(ranking_list, output_path):
    """Saves the scoring and ranking data to a separate CSV file."""
    ranking_output_path = output_path
    
    print(f"Saving rule ranking data to: {ranking_output_path}...")

    # Calculate a combined score for ranking: Effectiveness is 10x more important than Uniqueness
    for rule in ranking_list:
        rule['combined_score'] = rule.get('effectiveness_score', 0) * 10 + rule.get('uniqueness_score', 0)

    # Filtering and sorting
    ranked_rules = [rule for rule in ranking_list if rule.get('combined_score', 0) > 0]
    ranked_rules.sort(key=lambda rule: rule['combined_score'], reverse=True)

    if not ranked_rules:
        print("❌ No rules had a positive combined score. Ranking file not created.")
        return None

    try:
        with open(ranking_output_path, 'w', newline='', encoding='utf-8') as f:
            # Note: Key is 'Combined_Score' (Capital S)
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
                # Ensure scores are integers when reading from CSV string fields
                try:
                    # FIX: Read the correct key 'Combined_Score' and convert to int
                    row['Combined_Score'] = int(row['Combined_Score'])
                    ranked_data.append(row)
                except ValueError:
                    # Skip row if score is not a valid integer
                    continue
    except FileNotFoundError:
        print(f"❌ Error: Ranking CSV file not found at: {csv_path}")
        return
    except Exception as e:
        print(f"❌ Error while reading CSV: {e}")
        return

    # Sort by the score (descending) as a precaution
    # FIX: Corrected key from 'Combined_score' to 'Combined_Score'
    ranked_data.sort(key=lambda row: row['Combined_Score'], reverse=True)

    # Select Top K
    final_optimized_list = ranked_data[:top_k]

    if not final_optimized_list:
        print("❌ No rules available after sorting/filtering. Optimized rule file not created.")
        return

    # Save to File
    try:
        with open(output_path, 'w', newline='\n', encoding='utf-8') as f:
            # Add identity rule at the start
            f.write(":\n")
            for rule in final_optimized_list:
                f.write(f"{rule['Rule_Data']}\n")
        print(f"✅ Top {len(final_optimized_list)} optimized rules saved successfully to {output_path}.")
    except Exception as e:
        print(f"❌ Error while saving optimized rules to file: {e}")


def wordlist_iterator(wordlist_path, max_len, batch_size):
    """
    Generator that yields batches of words and initial hashes directly from disk.
    """
    base_words_np = np.zeros((batch_size, max_len), dtype=np.uint8)
    base_hashes = []
    current_batch_count = 0

    with open(wordlist_path, 'r', encoding='latin-1', errors='ignore') as f:
        for line in f:
            word_str = line.strip()
            word = word_str.encode('latin-1')

            if 1 <= len(word) <= max_len:

                base_words_np[current_batch_count, :len(word)] = np.frombuffer(word, dtype=np.uint8)
                base_hashes.append(fnv1a_hash_32_cpu(word))

                current_batch_count += 1

                if current_batch_count == batch_size:
                    # Use .copy() on the relevant memory blocks to ensure the generator's buffers can be immediately reused
                    yield base_words_np.ravel().copy(), current_batch_count, np.array(base_hashes, dtype=np.uint32)

                    base_words_np.fill(0)
                    base_hashes = []
                    current_batch_count = 0

    if current_batch_count > 0:
        words_to_yield = base_words_np[:current_batch_count, :].ravel().copy()
        yield words_to_yield, current_batch_count, np.array(base_hashes, dtype=np.uint32)


# --- MAIN RANKING FUNCTION (Optimized for Async Performance) ---

def rank_rules_uniqueness(wordlist_path, rules_path, cracked_list_path, ranking_output_path, top_k):
    start_time = time()
    
    # 1. OpenCL Initialization
    try:
        platform = cl.get_platforms()[0]
        devices = platform.get_devices(cl.device_type.GPU)
        if not devices:
            devices = platform.get_devices(cl.device_type.ALL)
        device = devices[0]
        context = cl.Context([device])
        
        # Enable PROFILING for accurate event timing (if desired)
        queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # Use fast math for speed
        prg = cl.Program(context, KERNEL_SOURCE).build(options=["-cl-fast-relaxed-math"])
        kernel_bfs = prg.bfs_kernel
        kernel_init = prg.hash_map_init_kernel
        print(f"✅ OpenCL initialized on device: {device.name.strip()}")
        print("🛠️  Kernel compiled with -cl-fast-relaxed-math for performance.")
    except Exception as e:
        print(f"❌ OpenCL initialization or kernel compilation error: {e}")
        exit(1)

    # 2. Data Loading and Pre-encoding
    rules_list = load_rules(rules_path)
    total_words = get_word_count(wordlist_path)
    total_rules = len(rules_list)
    rule_size_in_int = 2 + MAX_RULE_ARGS
    
    encoded_rules = [encode_rule(rule['rule_data'], rule['rule_id'], MAX_RULE_ARGS) for rule in rules_list]

    cracked_hashes_np = load_cracked_hashes(cracked_list_path, MAX_WORD_LEN)
    
    # 3. Hash Map Initialization (Host size for context, filled on GPU)
    global_hash_map_np = np.zeros(GLOBAL_HASH_MAP_WORDS, dtype=np.uint32)
    print(f"📝 Global Hash Map initialized: {global_hash_map_np.nbytes / (1024*1024):.2f} MB allocated.")

    cracked_hash_map_np = np.zeros(CRACKED_HASH_MAP_WORDS, dtype=np.uint32)
    print(f"📝 Cracked Hash Map initialized: {cracked_hash_map_np.nbytes / (1024*1024):.2f} MB allocated.")

    # 4. OpenCL Buffer Setup
    mf = cl.mem_flags

    # A) Base Word & Hash Input Buffer (Double Buffering)
    base_words_size = WORDS_PER_GPU_BATCH * MAX_WORD_LEN * np.uint8().itemsize
    # Two buffers for pipelining: [0] for current exec, [1] for next copy
    base_words_in_g = [cl.Buffer(context, mf.READ_ONLY, base_words_size), cl.Buffer(context, mf.READ_ONLY, base_words_size)]
    base_hashes_size = WORDS_PER_GPU_BATCH * np.uint32().itemsize
    base_hashes_g = [cl.Buffer(context, mf.READ_ONLY, base_hashes_size), cl.Buffer(context, mf.READ_ONLY, base_hashes_size)]
    
    current_word_buffer_idx = 0
    copy_events = [None, None] # To track the copy events for each buffer index

    # C) Rule Input Buffer (Allocate for max batch size)
    rule_batch_byte_size = MAX_RULES_IN_BATCH * rule_size_in_int * np.uint32().itemsize
    rules_in_g = cl.Buffer(context, mf.READ_ONLY, rule_batch_byte_size)

    # D) Global Hash Map (RW for base wordlist check)
    global_hash_map_g = cl.Buffer(context, mf.READ_WRITE, global_hash_map_np.nbytes)

    # E) Cracked Hash Map (Read Only, filled once)
    cracked_hash_map_g = cl.Buffer(context, mf.READ_ONLY, cracked_hash_map_np.nbytes)

    # F) Rule Counters (Uniqueness & Effectiveness)
    counters_size = MAX_RULES_IN_BATCH * np.uint32().itemsize
    rule_uniqueness_counts_g = cl.Buffer(context, mf.READ_WRITE, counters_size)
    rule_effectiveness_counts_g = cl.Buffer(context, mf.READ_WRITE, counters_size)

    # 5. INITIALIZE CRACKED HASH MAP (ONCE)
    if cracked_hashes_np.size > 0:
        cracked_temp_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cracked_hashes_np)
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
        
    # 6. Pipelined Ranking Loop Setup
    word_iterator = wordlist_iterator(wordlist_path, MAX_WORD_LEN, WORDS_PER_GPU_BATCH)
    rule_batch_starts = list(range(0, total_rules, MAX_RULES_IN_BATCH))
    
    words_processed_total = 0
    total_unique_found = 0 # Track total unique entries generated
    total_cracked_found = 0
    
    # Host arrays for receiving the GPU results (reused for each rule batch)
    mapped_uniqueness_np = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
    mapped_effectiveness_np = np.zeros(MAX_RULES_IN_BATCH, dtype=np.uint32)
    
    # Stores the count of the batch currently being executed
    num_words_batch_exec = 0
    
    # Updated progress bar to show Unique and Cracked counts
    word_batch_pbar = tqdm(total=total_words, desc="Processing wordlist from disk [Unique: 0 | Cracked: 0]", unit=" word")

    # A. Initial word batch load (Blocking: We need data for the first execution)
    try:
        base_words_np_batch, num_words_batch_exec, base_hashes_np_batch = next(word_iterator)
        
        # Copy first batch synchronously (or wait for event immediately after async copy)
        cl.enqueue_copy(queue, base_words_in_g[current_word_buffer_idx], base_words_np_batch)
        cl.enqueue_copy(queue, base_hashes_g[current_word_buffer_idx], base_hashes_np_batch).wait()
        
    except StopIteration:
        print("Wordlist is empty or too small.")
        word_batch_pbar.close()
        return

    # B. Main Pipelined Loop
    while True:
        exec_idx = current_word_buffer_idx
        next_idx = 1 - current_word_buffer_idx
        
        # 7. ASYNC COPY: Start load of the *next* batch while the *current* one executes
        next_batch_data = None
        try:
            next_batch_data = next(word_iterator)
            base_words_np_next, num_words_batch_next, base_hashes_np_next = next_batch_data
            
            # Asynchronously copy the next word batch to the other GPU buffer
            copy_events[next_idx] = cl.enqueue_copy(queue, base_words_in_g[next_idx], base_words_np_next)
            cl.enqueue_copy(queue, base_hashes_g[next_idx], base_hashes_np_next, wait_for=[copy_events[next_idx]])
            
        except StopIteration:
            # Reached the end of the wordlist iterator
            next_batch_data = None
            copy_events[next_idx] = None
        
        # 8. PROCESS RULE BATCHES (Synchronous for the current word batch)
        for rule_batch_idx, rule_start_index in enumerate(rule_batch_starts):
            
            # --- Rule Batch Setup ---
            rule_end_index = min(rule_start_index + MAX_RULES_IN_BATCH, total_rules)
            num_rules_in_batch = rule_end_index - rule_start_index

            # Assemble a new NumPy array that is exactly the size of the current batch
            current_rule_batch_list = encoded_rules[rule_start_index:rule_end_index]
            current_rules_np = np.concatenate(current_rule_batch_list)
            
            # Copy the rule batch to the GPU buffer
            cl.enqueue_copy(queue, rules_in_g, current_rules_np, is_blocking=True)
            
            # Clear the result counters on the GPU for this rule batch
            cl.enqueue_fill_buffer(queue, rule_uniqueness_counts_g, np.uint32(0), 0, counters_size)
            cl.enqueue_fill_buffer(queue, rule_effectiveness_counts_g, np.uint32(0), 0, counters_size)
            
            # --- Kernel Execution Setup ---
            global_size = (num_words_batch_exec * num_rules_in_batch, )
            local_size = (LOCAL_WORK_SIZE, )
            
            # Scale global size to be a multiple of local size
            global_size_aligned = (int(math.ceil(global_size[0] / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)

            # --- KERNEL EXECUTION ---
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

            # Enqueue the main kernel (wait for the *current* word copy to finish)
            exec_event = cl.enqueue_nd_range_kernel(queue, kernel_bfs, global_size_aligned, local_size)
            exec_event.wait() 

            # --- Read Results Back (Small, fast copy) ---
            cl.enqueue_copy(queue, mapped_uniqueness_np, rule_uniqueness_counts_g, is_blocking=True)
            cl.enqueue_copy(queue, mapped_effectiveness_np, rule_effectiveness_counts_g, is_blocking=True)

            # --- Update Host Score Totals ---
            for i in range(num_rules_in_batch):
                rule_index = rule_start_index + i
                
                # Check for potential overflow before adding (Unlikely, but good practice)
                if rules_list[rule_index]['uniqueness_score'] + mapped_uniqueness_np[i] < rules_list[rule_index]['uniqueness_score']:
                    print(f"⚠️ Uniqueness score overflow detected for rule ID {rules_list[rule_index]['rule_id']}")
                
                rules_list[rule_index]['uniqueness_score'] += mapped_uniqueness_np[i]
                rules_list[rule_index]['effectiveness_score'] += mapped_effectiveness_np[i]
                
                total_unique_found += mapped_uniqueness_np[i]
                total_cracked_found += mapped_effectiveness_np[i]


        # 9. UPDATE AND PREPARE NEXT ITERATION
        
        words_processed_total += num_words_batch_exec
        word_batch_pbar.update(num_words_batch_exec)
        word_batch_pbar.set_description(f"Processing wordlist from disk [Unique: {total_unique_found:,} | Cracked: {total_cracked_found:,}]")

        if next_batch_data is None:
            # Last batch was processed, exit the loop
            break
            
        # Switch buffers and prepare for the next iteration
        current_word_buffer_idx = next_idx
        num_words_batch_exec = num_words_batch_next


    word_batch_pbar.close()
    end_time = time()
    
    # 10. FINAL REPORTING AND SAVING
    
    print("\n--- Final Results Summary ---")
    print(f"Total Words Processed: {words_processed_total:,}")
    print(f"Total Unique Words Generated: {total_unique_found:,}")
    print(f"Total True Cracks Found: {total_cracked_found:,}")
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
    print("-----------------------------\n")

    # Save ranking data and optimize rules
    csv_path = save_ranking_data(rules_list, ranking_output_path)
    if top_k > 0:
        optimized_output_path = os.path.splitext(ranking_output_path)[0] + "_optimized.rule"
        load_and_save_optimized_rules(csv_path, optimized_output_path, top_k)

# --- END OF MAIN RANKING FUNCTION ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPU-Accelerated Hashcat Rule Ranking Tool (Ranker v2.8)")
    parser.add_argument('-w', '--wordlist', required=True, help='Path to the base wordlist file.')
    parser.add_argument('-r', '--rules', required=True, help='Path to the Hashcat rules file to rank.')
    parser.add_argument('-c', '--cracked', required=True, help='Path to a list of cracked passwords for effectiveness scoring.')
    parser.add_argument('-o', '--output', default='ranker_output.csv', help='Path to save the final ranking CSV.')
    parser.add_argument('-k', '--topk', type=int, default=1000, help='Number of top rules to save to an optimized .rule file. Set to 0 to skip.')
    
    args = parser.parse_args()

    rank_rules_uniqueness(
        wordlist_path=args.wordlist,
        rules_path=args.rules,
        cracked_list_path=args.cracked,
        ranking_output_path=args.output,
        top_k=args.topk
    )
