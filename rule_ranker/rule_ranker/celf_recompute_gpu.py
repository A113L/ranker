#!/usr/bin/env python3
"""
celf_recompute_gpu.py -- memory-light GPU greedy coverage selection
=====================================================================
The single recompute-then-select greedy strategy used by
ranker_postprocess.py.  No per-candidate coverage matrix is materialized.

WHY THIS EXISTS
----------------
The post-processing stage used to support large, materialized coverage stores.
That design imposed memory/disk growth proportional to candidates times the
cracked universe.  The recompute path avoids that entirely: it scores the
remaining candidates directly on the GPU against a shrinking active target set
and uses monotone upper bounds to skip candidates that cannot win the current
round.

This module implements that recompute + lazy-upper-bound strategy in
rule_ranker, reusing the same GPU rule-application transform as the
post-processing kernel so results match. Hash membership uses a GPU
open-addressing table rather than a per-hit binary search through the
sorted cracked array, substantially reducing random global-memory reads.
The only state kept between
rounds is:
  - `active` : a single (W,) uint32 bitmap of cracked hashes NOT YET
    covered (W = ceil(cracked_universe/32) words -- e.g. ~1.2 MB for a
    10M-entry universe, however many candidate rules there are).
  - `upper_bound` : (n_rules,) int32 -- each rule's static full-target
    popcount, computed ONCE in a single streaming pass, accumulating one
    atomic counter per rule instead of writing a coverage row.
  - `candidate_heap` : a heap of `(-priority, rule_index, raw_gain)`, where
    priority is raw_gain/cost(rule) (cost is 1.0 for every rule unless
    cost_alpha > 0 is passed, in which case this is cost-weighted CELF)
    entries for candidates with at least one hit. After a candidate is
    rescored, its exact marginal gain becomes a tighter heap bound. This
    keeps true lazy-greedy/CELF ordering without repeatedly scanning a
    fixed list whose initial bounds become stale as coverage shrinks.

Nothing shaped (n_rules x cracked_universe) is ever allocated, in RAM
or on disk. Peak extra memory beyond the rule pool is
O(n_rules) + O(cracked_universe bits) + O(wordlist), since the
wordlist itself is kept resident (GPU memory if it fits, host RAM
otherwise) rather than re-streamed from disk on every rescore.

Usage
-----
    python3 -m rule_ranker.celf_recompute_gpu \\
        --rules-file top_optimized.rule \\
        --wordlist rockyou.txt \\
        --cracked cracked.txt \\
        --budget 5000 \\
        --output celf_selected.rule

Or import celf_select_recompute_gpu() directly and feed it the same
`rules` list load_candidate_rules()/load_cracked_universe() from
ranker_postprocess.py already produce, to drop it into an existing
pipeline as the recompute-gpu CELF implementation.
"""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import pyopencl as cl
from .progress import ProgressBar

from .hashing import build_open_addressing_table_uint64, table_size_for_count

from .ranker_postprocess import (
    MAX_WORD_LEN, MAX_OUTPUT_LEN, MAX_RULE_LEN, LOCAL_WORK_SIZE,
    DEFAULT_WORDS_PER_GPU_BATCH, MAX_DISPATCH_ITEMS,
    log, red, green, yellow, blue, cyan, bold, dim,
    optimized_wordlist_iterator, load_cracked_universe, load_candidate_rules,
    select_device, save_output, parse_budgets, save_output_multi,
)

# The full apply_hashcat_rule()/apply_single_command() C source is kept
# in this module rather than imported from another implementation. The two
# kernel entry points are specialized for scoring against the active target
# set and for clearing the winning rule's newly-covered targets. Keeping
# this source local makes each kernel independently readable/buildable.
_COMMON_KERNEL_BODY = r"""
int is_lower(unsigned char c) { return (c >= 'a' && c <= 'z'); }
int is_upper(unsigned char c) { return (c >= 'A' && c <= 'Z'); }
unsigned char to_lower(unsigned char c) { return is_upper(c) ? c + 32 : c; }
unsigned char to_upper(unsigned char c) { return is_lower(c) ? c - 32 : c; }
unsigned char toggle_case(unsigned char c) {
    if (is_lower(c)) return c - 32;
    if (is_upper(c)) return c + 32;
    return c;
}
unsigned int char_to_pos(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    if (c >= 'a' && c <= 'z') return c - 'a' + 10;
    return 0xFFFFFFFF;
}
ulong fnv1a_hash_64(const unsigned char* data, unsigned int len) {
    ulong hash = 14695981039346656037UL;
    for (unsigned int i = 0; i < len; i++) { hash ^= (ulong)data[i]; hash *= 1099511628211UL; }
    return hash;
}
// Return 0 on success, -2 on output-length overflow.
static int duplicate_front(const unsigned char* in, int in_len,
                           unsigned char* out, int* out_len, int* changed, int n) {
    if (n > in_len) n = in_len;
    int new_len = in_len + n;
    if (new_len > MAX_OUTPUT_LEN) return -2;
    for (int i = 0; i < n; i++) out[i] = in[i];
    for (int i = 0; i < in_len; i++) out[n + i] = in[i];
    *out_len = new_len; *changed = 1;
    return 0;
}
static int duplicate_back(const unsigned char* in, int in_len,
                          unsigned char* out, int* out_len, int* changed, int n) {
    if (n > in_len) n = in_len;
    int new_len = in_len + n;
    if (new_len > MAX_OUTPUT_LEN) return -2;
    for (int i = 0; i < in_len; i++) out[i] = in[i];
    for (int i = 0; i < n; i++) out[in_len + i] = in[in_len - n + i];
    *out_len = new_len; *changed = 1;
    return 0;
}
static int duplicate_word(const unsigned char* in, int in_len,
                          unsigned char* out, int* out_len, int* changed, int times) {
    int new_len = in_len * (times + 1);
    if (new_len > MAX_OUTPUT_LEN) return -2;
    for (int rep = 0; rep <= times; rep++)
        for (int i = 0; i < in_len; i++) out[rep * in_len + i] = in[i];
    *out_len = new_len; *changed = 1;
    return 0;
}
static void rotate_left(const unsigned char* in, int in_len,
                        unsigned char* out, int* out_len, int* changed, int n) {
    if (in_len <= 0) { *out_len = 0; *changed = 0; return; }
    if (n <= 0) n = 1;
    n %= in_len;
    *out_len = in_len;
    if (n == 0) { for (int i = 0; i < in_len; i++) out[i] = in[i]; *changed = 0; return; }
    for (int i = 0; i < in_len; i++) out[i] = in[(i + n) % in_len];
    *changed = 1;
}
static void rotate_right(const unsigned char* in, int in_len,
                         unsigned char* out, int* out_len, int* changed, int n) {
    if (in_len <= 0) { *out_len = 0; *changed = 0; return; }
    if (n <= 0) n = 1;
    n %= in_len;
    *out_len = in_len;
    if (n == 0) { for (int i = 0; i < in_len; i++) out[i] = in[i]; *changed = 0; return; }
    for (int i = 0; i < in_len; i++) out[i] = in[(i - n + in_len) % in_len];
    *changed = 1;
}
static int apply_single_command(const unsigned char* in, int in_len,
                                unsigned char* out, int* out_len,
                                const unsigned char* cmd, int cmd_len) {
    int changed = 0;
    *out_len = 0;
    if (cmd_len == 1) {
        switch (cmd[0]) {
            case 'l': *out_len = in_len; for (int i=0;i<in_len;i++) out[i]=to_lower(in[i]); changed=1; break;
            case 'u': *out_len = in_len; for (int i=0;i<in_len;i++) out[i]=to_upper(in[i]); changed=1; break;
            case 'c': *out_len = in_len; if (in_len>0) out[0]=to_upper(in[0]); for (int i=1;i<in_len;i++) out[i]=to_lower(in[i]); changed=1; break;
            case 'C': *out_len = in_len; if (in_len>0) out[0]=to_lower(in[0]); for (int i=1;i<in_len;i++) out[i]=to_upper(in[i]); changed=1; break;
            case 't': *out_len = in_len; for (int i=0;i<in_len;i++) out[i]=toggle_case(in[i]); changed=1; break;
            case 'r': *out_len = in_len; for (int i=0;i<in_len;i++) out[i]=in[in_len-1-i]; changed=1; break;
            case 'd': if (in_len*2>MAX_OUTPUT_LEN) return -2; *out_len=in_len*2; for(int i=0;i<in_len;i++){out[i]=in[i];out[in_len+i]=in[i];} changed=1; break;
            case 'f': if (in_len*2>MAX_OUTPUT_LEN) return -2; *out_len=in_len*2; for(int i=0;i<in_len;i++){out[i]=in[i];out[in_len+i]=in[in_len-1-i];} changed=1; break;
            case 'k': *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; if(in_len>=2){out[0]=in[1];out[1]=in[0];changed=1;} break;
            case 'K': *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; if(in_len>=2){out[in_len-2]=in[in_len-1];out[in_len-1]=in[in_len-2];changed=1;} break;
            case ':': *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; changed=0; break;
            case 'q': if (in_len*2>MAX_OUTPUT_LEN) return -2; int idx=0; for(int i=0;i<in_len;i++){out[idx++]=in[i];out[idx++]=in[i];} *out_len=in_len*2; changed=1; break;
            case 'E': { *out_len=in_len; int cap=1; for(int i=0;i<in_len;i++){ if(cap&&is_lower(in[i])) out[i]=to_upper(in[i]); else out[i]=to_lower(in[i]); cap=(in[i]==' '||in[i]=='-'||in[i]=='_'); } changed=1; } break;
            case '{': rotate_left(in,in_len,out,out_len,&changed,1); break;
            case '}': rotate_right(in,in_len,out,out_len,&changed,1); break;
            case '[': if (in_len>1) { *out_len=in_len-1; for(int i=1;i<in_len;i++) out[i-1]=in[i]; changed=1; } break;
            case ']': if (in_len>1) { *out_len=in_len-1; for(int i=0;i<in_len-1;i++) out[i]=in[i]; changed=1; } break;
            default: *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; changed=0; break;
        }
        return changed;
    }
    if (cmd_len == 2) {
        unsigned char cmd_char = cmd[0];
        unsigned char arg = cmd[1];
        int n = (int)char_to_pos(arg);
        if (n == 0xFFFFFFFF) n = -1;
        switch (cmd_char) {
            case 'T': if (n>=0&&n<in_len){ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; out[n]=toggle_case(in[n]); changed=1; } break;
            case 'D': if (n>=0&&n<in_len){ *out_len=in_len-1; for(int i=0;i<n;i++) out[i]=in[i]; for(int i=n+1;i<in_len;i++) out[i-1]=in[i]; changed=1; } break;
            case 'L': if (n>=0&&n<in_len){ *out_len=in_len-n; for(int i=n;i<in_len;i++) out[i-n]=in[i]; changed=1; } break;
            case 'R': if (n>=0&&n<in_len){ *out_len=n+1; for(int i=0;i<=n;i++) out[i]=in[i]; changed=1; } break;
            case '+': if (n>=0&&n<in_len){ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; out[n]=in[n]+1; changed=1; } break;
            case '-': if (n>=0&&n<in_len){ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; out[n]=in[n]-1; changed=1; } break;
            case '.': if (n>=0&&n<in_len){ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; out[n]=in[n]+1; changed=1; } break;
            case ',': if (n>=0&&n<in_len){ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; out[n]=in[n]-1; changed=1; } break;
            case '\'': if (n>=0&&n<in_len){ *out_len=n; for(int i=0;i<n;i++) out[i]=in[i]; changed=1; } break;
            case '^': if (in_len+1>MAX_OUTPUT_LEN) return -2; out[0]=arg; for(int i=0;i<in_len;i++) out[i+1]=in[i]; *out_len=in_len+1; changed=1; break;
            case '$': if (in_len+1>MAX_OUTPUT_LEN) return -2; for(int i=0;i<in_len;i++) out[i]=in[i]; out[in_len]=arg; *out_len=in_len+1; changed=1; break;
            case '@': *out_len=0; for(int i=0;i<in_len;i++){ if(in[i]!=arg) out[(*out_len)++]=in[i]; else changed=1; } break;
            case '!': for(int i=0;i<in_len;i++) if(in[i]==arg) return -1; *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; return 0;
            case '/': for(int i=0;i<in_len;i++) if(in[i]==arg){ *out_len=in_len; for(int j=0;j<in_len;j++) out[j]=in[j]; return 0; } return -1;
            case '(': if (in_len>0&&in[0]==arg){ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; return 0; } return -1;
            case ')': if (in_len>0&&in[in_len-1]==arg){ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; return 0; } return -1;
            case 'y': if (n>=0) return duplicate_front(in,in_len,out,out_len,&changed,n); break;
            case 'Y': if (n>=0) return duplicate_back(in,in_len,out,out_len,&changed,n); break;
            case 'z': if (n>0) { if (in_len+n>MAX_OUTPUT_LEN) return -2; out[0]=in[0]; for(int i=0;i<n;i++) out[i+1]=in[0]; for(int i=1;i<in_len;i++) out[n+i]=in[i]; *out_len=in_len+n; changed=1; } break;
            case 'Z': if (n>0) { if (in_len+n>MAX_OUTPUT_LEN) return -2; for(int i=0;i<in_len;i++) out[i]=in[i]; for(int i=0;i<n;i++) out[in_len+i]=in[in_len-1]; *out_len=in_len+n; changed=1; } break;
            case 'p': if (n>=0) return duplicate_word(in,in_len,out,out_len,&changed,n); break;
            case '{': if (n>=0) rotate_left(in,in_len,out,out_len,&changed,n); break;
            case '}': if (n>=0) rotate_right(in,in_len,out,out_len,&changed,n); break;
            case '[': if (n>=0&&n<in_len){ *out_len=in_len-n; for(int i=n;i<in_len;i++) out[i-n]=in[i]; changed=1; } break;
            case ']': if (n>=0&&n<in_len){ *out_len=in_len-n; for(int i=0;i<*out_len;i++) out[i]=in[i]; changed=1; } break;
            default: *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; changed=0; break;
        }
        return changed;
    }
    if (cmd_len == 3) {
        unsigned char cmd_char = cmd[0];
        unsigned char a1 = cmd[1];
        unsigned char a2 = cmd[2];
        int n1 = (int)char_to_pos(a1);
        int n2 = (int)char_to_pos(a2);
        if (n1 == (int)0xFFFFFFFF) n1 = -1;
        if (n2 == (int)0xFFFFFFFF) n2 = -1;
        switch (cmd_char) {
            case 'x': if (n1>=0&&n2>0&&n1<in_len){ int end=n1+n2; if(end>in_len) end=in_len; *out_len=end-n1; for(int i=0;i<*out_len;i++) out[i]=in[n1+i]; changed=1; } break;
            case 'O': if (n1>=0&&n2>0&&n1<in_len){ int end=n1+n2; if(end>in_len) end=in_len; int rm=end-n1; *out_len=in_len-rm; for(int i=0;i<n1;i++) out[i]=in[i]; for(int i=end;i<in_len;i++) out[i-rm]=in[i]; changed=1; } break;
            case 'i': if (n1>=0) { if (in_len+1>MAX_OUTPUT_LEN) return -2; int p=n1; if(p>in_len) p=in_len; for(int i=0;i<p;i++) out[i]=in[i]; out[p]=a2; for(int i=p;i<in_len;i++) out[i+1]=in[i]; *out_len=in_len+1; changed=1; } break;
            case 's': *out_len=in_len; for(int i=0;i<in_len;i++){ out[i]=(in[i]==a1)?a2:in[i]; if(in[i]==a1) changed=1; } break;
            case 'o': if (n1>=0&&n1<in_len){ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; out[n1]=a2; changed=1; } break;
            case '*': if (n1>=0&&n2>=0&&n1<in_len&&n2<in_len&&n1!=n2){ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; unsigned char tmp=out[n1]; out[n1]=out[n2]; out[n2]=tmp; changed=1; } break;
            case '3': if (n1>=0){ int count=0; *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; for(int i=0;i<in_len;i++){ if(in[i]==a2) count++; if(count==n1+1 && i+1<in_len){ out[i+1]=toggle_case(in[i+1]); changed=1; break; } } } break;
            case '%': if (n1>=0){ int cnt=0; for(int i=0;i<in_len;i++) if(in[i]==a2) cnt++; if(cnt<n1) return -1; } *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; return 0;
            case '=': if (n1>=0&&n1<in_len&&in[n1]!=a2) return -1; *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; return 0;
            default: *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; changed=0; break;
        }
        return changed;
    }
    *out_len = in_len; for (int i=0;i<in_len;i++) out[i]=in[i]; return 0;
}
void apply_hashcat_rule(const unsigned char* word, int word_len,
                        const unsigned char* rule, int rule_len,
                        unsigned char* output, int* out_len, int* changed) {
    unsigned char buf0[MAX_OUTPUT_LEN];
    unsigned char buf1[MAX_OUTPUT_LEN];
    unsigned char* in_buf = (unsigned char*)word;
    int in_len = word_len;
    int final_changed = 0;
    int pos = 0;
    while (pos < rule_len) {
        unsigned char cmd_char = rule[pos];
        int cmd_len = 1;
        if (cmd_char=='s'||cmd_char=='x'||cmd_char=='O'||cmd_char=='i'||cmd_char=='o'||
            cmd_char=='*'||cmd_char=='3'||cmd_char=='%'||cmd_char=='=') {
            cmd_len = 3;
        } else if (pos+1 < rule_len && (cmd_char=='T'||cmd_char=='D'||cmd_char=='L'||cmd_char=='R'||
                    cmd_char=='+'||cmd_char=='-'||cmd_char=='.'||cmd_char==','||cmd_char=='\''||
                    cmd_char=='^'||cmd_char=='$'||cmd_char=='@'||cmd_char=='!'||cmd_char=='/'||
                    cmd_char=='('||cmd_char==')'||cmd_char=='y'||cmd_char=='Y'||cmd_char=='z'||
                    cmd_char=='Z'||cmd_char=='p'||cmd_char=='{'||cmd_char=='}'||cmd_char=='['||
                    cmd_char==']'||cmd_char=='_'||cmd_char=='e')) {
            cmd_len = 2;
        }
        if (pos + cmd_len > rule_len) break;
        int out_len_local = 0;
        int result = apply_single_command(in_buf, in_len, buf0, &out_len_local, rule+pos, cmd_len);
        if (result < 0) { *out_len = 0; *changed = result; return; }
        if (result == 1) final_changed = 1;
        in_len = out_len_local;
        for (int i=0;i<in_len;i++) buf1[i]=buf0[i];
        in_buf = buf1;
        pos += cmd_len;
    }
    *out_len = in_len;
    for (int i=0;i<in_len;i++) output[i]=in_buf[i];
    *changed = final_changed;
}
// Open-addressed lookup over 64-bit FNV-1a fingerprints. `occupied` is a
// bitset indexed by table slot, so fingerprint 0 remains a valid key.
ulong mix64(ulong x) {{
    x += 0x9E3779B97F4A7C15UL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
    return x ^ (x >> 31);
}}

int lookup_cracked_slot(__global const ulong* hash_table,
                        __global const unsigned int* occupied,
                        unsigned int table_mask, ulong key) {{
    unsigned int slot = (unsigned int)(mix64(key) & (ulong)table_mask);
    unsigned int word = slot >> 5;
    unsigned int bit = slot & 31U;
    for (unsigned int probe = 0; probe <= table_mask; probe++) {{
        if (occupied[word] & (1U << bit)) {{
            if (hash_table[slot] == key) return (int)slot;
        }} else {{
            return -1;
        }}
        slot = (slot + 1U) & table_mask;
        word = slot >> 5;
        bit = slot & 31U;
    }}
    return -1;
}}
"""


def _rule_cost(rule, cost_alpha):
    """Cost of a candidate rule used to normalize marginal gain into a
    gain-per-cost priority (gain / cost) for the CELF heap.

    cost_alpha <= 0 (default) means every rule costs 1.0 -- an ordinary
    equal-cost greedy, identical to this module's previous behavior.
    cost_alpha > 0 makes longer/more token-heavy rules progressively more
    expensive, so the greedy prefers a cheaper rule over a pricier one
    that covers the same (or only marginally more) of the active target
    set -- mirrors rulest's `_rule_cost`/cost-weighted CELF.
    """
    if cost_alpha <= 0:
        return 1.0
    return max(1, len(rule.split())) ** float(cost_alpha)


class _ConstantCosts:
    """costs[index] == value for any index, without allocating a
    len(rules) list -- used for the (default) cost_alpha<=0 case where
    every rule's cost is the same 1.0."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __getitem__(self, _index):
        return self.value


def get_recompute_kernel_source(num_cracked, hash_table_size):
    """Two kernels sharing the same rule-transform/hash plumbing as the
    scorer's common kernel body; neither kernel writes per-rule coverage rows:

    - score_against_active_kernel: for each (word, rule) pair in the
      current dispatch, if the transformed word hashes to a cracked
      entry that is STILL SET in `active` (not yet covered by an
      already-selected rule), atomic_add 1 into gains[rule_idx]. Output
      is (num_rules_in_batch,) int32 -- O(rules), not O(rules x
      universe).
    - apply_and_clear_kernel: single-rule version that additionally
      atomic_and's the matched bit out of `active`, used once per
      greedy round for the round's chosen winner so the next round's
      scores already reflect reduced coverage.
    """
    return f"""
#define MAX_WORD_LEN {MAX_WORD_LEN}
#define MAX_OUTPUT_LEN {MAX_OUTPUT_LEN}
#define MAX_RULE_LEN {MAX_RULE_LEN}
#define NUM_CRACKED {num_cracked}
#define HASH_TABLE_SIZE {hash_table_size}
#define HASH_TABLE_MASK {hash_table_size - 1}

{_COMMON_KERNEL_BODY}

// active: ceil(NUM_CRACKED/32) words, bit=1 means "not yet covered".
// gains: one int32 accumulator per rule in this dispatch batch;
// the host zeroes it before each rule-batch.
__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void score_against_active_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned char* rules_in,
    __global const ulong* hash_table,
    __global const unsigned int* hash_table_occupied,
    __global const unsigned int* active,
    __global int* gains,
    const unsigned int num_words,
    const unsigned int num_rules_in_batch,
    const unsigned int max_word_len,
    const unsigned int table_mask)
{{
    unsigned int global_id = get_global_id(0);
    unsigned int total = num_words * num_rules_in_batch;
    if (global_id >= total) return;

    unsigned int rule_idx = global_id / num_words;
    unsigned int word_idx = global_id % num_words;

    unsigned char word[MAX_WORD_LEN];
    unsigned int word_len = 0;
    for (unsigned int i = 0; i < max_word_len; i++) {{
        unsigned char c = base_words_in[word_idx * max_word_len + i];
        if (c == 0) break;
        word[i] = c; word_len++;
    }}

    unsigned int rule_start = rule_idx * MAX_RULE_LEN;
    unsigned char rule_str[MAX_RULE_LEN];
    unsigned int rule_len = 0;
    for (unsigned int i = 0; i < MAX_RULE_LEN; i++) {{
        unsigned char c = rules_in[rule_start + i];
        if (c == 0) break;
        rule_str[i] = c; rule_len++;
    }}

    unsigned char result_temp[MAX_OUTPUT_LEN];
    int out_len = 0, changed = 0;
    apply_hashcat_rule(word, word_len, rule_str, rule_len, result_temp, &out_len, &changed);
    if (changed <= 0 || out_len <= 0) return;

    ulong h = fnv1a_hash_64(result_temp, (unsigned int)out_len);
    int slot = lookup_cracked_slot(hash_table, hash_table_occupied, table_mask, h);
    if (slot < 0) return;

    unsigned int word_pos = (unsigned int)slot >> 5;
    unsigned int bit_pos = (unsigned int)slot & 31U;
    if ((active[word_pos] & (1U << bit_pos)) == 0) return;  // already covered

    atomic_add(&gains[rule_idx], 1);
}}

// Single-rule pass for the round's chosen winner: clears matched bits
// from `active` in place and counts how many were actually cleared
// (the round's true, current gain -- used to log/verify against the
// lazy-heap estimate that picked this rule).
__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void apply_and_clear_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned char* rule_in,
    __global const ulong* hash_table,
    __global const unsigned int* hash_table_occupied,
    __global unsigned int* active,
    __global int* cleared_count,
    const unsigned int num_words,
    const unsigned int rule_len,
    const unsigned int max_word_len,
    const unsigned int table_mask)
{{
    unsigned int word_idx = get_global_id(0);
    if (word_idx >= num_words) return;

    unsigned char word[MAX_WORD_LEN];
    unsigned int word_len = 0;
    for (unsigned int i = 0; i < max_word_len; i++) {{
        unsigned char c = base_words_in[word_idx * max_word_len + i];
        if (c == 0) break;
        word[i] = c; word_len++;
    }}

    unsigned char rule_str[MAX_RULE_LEN];
    for (unsigned int i = 0; i < rule_len; i++) rule_str[i] = rule_in[i];

    unsigned char result_temp[MAX_OUTPUT_LEN];
    int out_len = 0, changed = 0;
    apply_hashcat_rule(word, word_len, rule_str, (int)rule_len, result_temp, &out_len, &changed);
    if (changed <= 0 || out_len <= 0) return;

    ulong h = fnv1a_hash_64(result_temp, (unsigned int)out_len);
    int slot = lookup_cracked_slot(hash_table, hash_table_occupied, table_mask, h);
    if (slot < 0) return;

    unsigned int word_pos = (unsigned int)slot >> 5;
    unsigned int bit_pos = (unsigned int)slot & 31U;
    unsigned int mask = (1U << bit_pos);
    unsigned int old = atomic_and(&active[word_pos], ~mask);
    if (old & mask) atomic_add(cleared_count, 1);
}}

// Single-rule variant of score_against_active_kernel that, in addition
// to counting, RECORDS each hit's hash-table slot into `hit_slots`
// (does NOT clear `active` -- the candidate is still just being
// compared against the round's current best). If this candidate goes
// on to win its round, the caller can reuse this already-known slot
// list (clear_recorded_slots_kernel below) instead of re-deriving
// positions with a second full-wordlist pass (apply_and_clear_kernel).
// hit_slots must be sized for at least num_words entries (a word
// contributes at most one hit).
__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void score_single_and_record_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned char* rule_in,
    __global const ulong* hash_table,
    __global const unsigned int* hash_table_occupied,
    __global const unsigned int* active,
    __global unsigned int* hit_slots,
    __global int* hit_count,
    const unsigned int num_words,
    const unsigned int rule_len,
    const unsigned int max_word_len,
    const unsigned int table_mask)
{{
    unsigned int word_idx = get_global_id(0);
    if (word_idx >= num_words) return;

    unsigned char word[MAX_WORD_LEN];
    unsigned int word_len = 0;
    for (unsigned int i = 0; i < max_word_len; i++) {{
        unsigned char c = base_words_in[word_idx * max_word_len + i];
        if (c == 0) break;
        word[i] = c; word_len++;
    }}

    unsigned char rule_str[MAX_RULE_LEN];
    for (unsigned int i = 0; i < rule_len; i++) rule_str[i] = rule_in[i];

    unsigned char result_temp[MAX_OUTPUT_LEN];
    int out_len = 0, changed = 0;
    apply_hashcat_rule(word, word_len, rule_str, (int)rule_len, result_temp, &out_len, &changed);
    if (changed <= 0 || out_len <= 0) return;

    ulong h = fnv1a_hash_64(result_temp, (unsigned int)out_len);
    int slot = lookup_cracked_slot(hash_table, hash_table_occupied, table_mask, h);
    if (slot < 0) return;

    unsigned int word_pos = (unsigned int)slot >> 5;
    unsigned int bit_pos = (unsigned int)slot & 31U;
    if ((active[word_pos] & (1U << bit_pos)) == 0) return;  // already covered

    int pos = atomic_add(hit_count, 1);
    hit_slots[pos] = (unsigned int)slot;
}}

// Multi-rule exact/deduplicated recording pass: like
// score_single_and_record_kernel, but for a small BATCH of rules at
// once (one kernel dispatch instead of one per candidate). Each rule
// gets its own hit-count slot and its own fixed-capacity segment of
// `hit_slots` (segment i = hit_slots[i*words_per_rule_cap .. +cap)).
// words_per_rule_cap must be >= num_words so no rule's segment can
// overflow (a rule hits at most one target per word, so at most
// num_words hits per rule per chunk). This exists purely to amortize
// GPU round-trip latency across several CELF candidates that each
// need an exact (post-dedup) gain this round -- raw counts from
// score_against_active_kernel are NOT safe to use here, same reason
// as score_single_and_record_kernel: two different words landing on
// the same still-active target must count once, not twice, and only
// the host-side np.unique over each rule's recorded slot segment
// (done by the Python caller) establishes that.
__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void score_exact_batch_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned char* rules_in,
    __global const ulong* hash_table,
    __global const unsigned int* hash_table_occupied,
    __global const unsigned int* active,
    __global unsigned int* hit_slots,
    __global int* hit_counts,
    const unsigned int num_words,
    const unsigned int num_rules_in_batch,
    const unsigned int words_per_rule_cap,
    const unsigned int max_word_len,
    const unsigned int table_mask)
{{
    unsigned int global_id = get_global_id(0);
    unsigned int total = num_words * num_rules_in_batch;
    if (global_id >= total) return;

    unsigned int rule_idx = global_id / num_words;
    unsigned int word_idx = global_id % num_words;

    unsigned char word[MAX_WORD_LEN];
    unsigned int word_len = 0;
    for (unsigned int i = 0; i < max_word_len; i++) {{
        unsigned char c = base_words_in[word_idx * max_word_len + i];
        if (c == 0) break;
        word[i] = c; word_len++;
    }}

    unsigned int rule_start = rule_idx * MAX_RULE_LEN;
    unsigned char rule_str[MAX_RULE_LEN];
    unsigned int rule_len = 0;
    for (unsigned int i = 0; i < MAX_RULE_LEN; i++) {{
        unsigned char c = rules_in[rule_start + i];
        if (c == 0) break;
        rule_str[i] = c; rule_len++;
    }}

    unsigned char result_temp[MAX_OUTPUT_LEN];
    int out_len = 0, changed = 0;
    apply_hashcat_rule(word, word_len, rule_str, rule_len, result_temp, &out_len, &changed);
    if (changed <= 0 || out_len <= 0) return;

    ulong h = fnv1a_hash_64(result_temp, (unsigned int)out_len);
    int slot = lookup_cracked_slot(hash_table, hash_table_occupied, table_mask, h);
    if (slot < 0) return;

    unsigned int word_pos = (unsigned int)slot >> 5;
    unsigned int bit_pos = (unsigned int)slot & 31U;
    if ((active[word_pos] & (1U << bit_pos)) == 0) return;  // already covered

    int pos = atomic_add(&hit_counts[rule_idx], 1);
    if ((unsigned int)pos < words_per_rule_cap) {{
        hit_slots[rule_idx * words_per_rule_cap + (unsigned int)pos] = (unsigned int)slot;
    }}
}}

// Clears a small, EXPLICIT list of previously-recorded hash-table
// slots from `active` -- the fast path for applying a round's winner
// when score_single_and_record_kernel already captured its exact hit
// positions earlier in the same round. Launched over just `n_hits`
// work-items (the candidate's gain) instead of the whole wordlist.
// Two recorded hits can still name the same slot (two different words
// both landing on the same cracked target), so this keeps the same
// atomic_and + old-bit check per entry that apply_and_clear_kernel
// uses, to get an exact cleared_count.
__kernel void clear_recorded_slots_kernel(
    __global unsigned int* active,
    __global const unsigned int* hit_slots,
    __global int* cleared_count,
    const unsigned int n_hits)
{{
    unsigned int i = get_global_id(0);
    if (i >= n_hits) return;
    unsigned int slot = hit_slots[i];
    unsigned int word_pos = slot >> 5;
    unsigned int bit_pos = slot & 31U;
    unsigned int mask = (1U << bit_pos);
    unsigned int old = atomic_and(&active[word_pos], ~mask);
    if (old & mask) atomic_add(cleared_count, 1);
}}
"""


class _ResidentWordlistMixin:
    """Parse-once, stay-resident wordlist handling, shared by every GPU
    backend in this package that needs to run MANY dispatches over the
    same wordlist across a run (the celf_recompute_gpu `_GpuScorer`).

    WHY THIS EXISTS
    ----------------
    Naively calling optimized_wordlist_iterator() fresh inside every
    per-rule/per-round GPU call means the same Python-level, line-by-
    line mmap parse re-runs every single time -- for a multi-million-
    line wordlist and many calls per run (CELF lazy-heap revalidation
    alone can call score_batch() far more times than rules actually
    get selected), that re-parse cost dwarfs the actual GPU kernel time
    by orders of magnitude. This was the root cause of an earlier
    ~35s/rule/revalidation slowdown in celf_recompute_gpu.py before
    this mixin was factored out.

    The wordlist is instead parsed and encoded exactly ONCE, in
    _preload_wordlist() below, into the same (words_buffer, count)
    chunk shape optimized_wordlist_iterator() used to yield on the fly.
    Those chunks are then pushed to the GPU as persistent buffers
    (self._word_chunks_gpu) so every later dispatch just iterates
    already-resident GPU buffers -- no disk I/O, no Python parsing, and
    no host->device copy of word data after startup. If the encoded
    wordlist doesn't fit in device memory (checked against the
    device's reported global memory size, with headroom for the
    caller's other buffers), this falls back to keeping the parsed
    chunks resident in HOST memory instead (self._word_chunks_host)
    and re-uploading each chunk into a single reusable device buffer
    on every pass. That fallback still eliminates the disk read +
    line-parsing cost (the dominant cost observed), at the price of
    repeated H2D copies, which are orders of magnitude cheaper than
    re-parsing the file.

    A subclass must, before calling _preload_wordlist(): set
    self.context / self.queue / self.device (a real pyopencl Context/
    CommandQueue/Device), self.words_per_gpu_batch, self.base_words_g
    (a READ_ONLY buffer sized for words_per_gpu_batch -- used as the
    reusable upload target for the disk-streaming fallback path, i.e.
    when _preload_wordlist() was never called at all), and initialize
    self._word_chunks_gpu = self._word_chunks_host =
    self._host_fallback_words_g = None.
    """

    # Fraction of device global memory we're willing to spend holding
    # the whole encoded wordlist resident, leaving room for the
    # cracked-hash table, active target bitset, rule batch buffers,
    # and driver overhead.
    # Conservative on purpose -- falling back to the host-resident path
    # is still a large win over per-call disk streaming, so there's no
    # need to cut this close.
    _GPU_RESIDENT_WORDLIST_FRACTION = 0.78

    def _resident_chunk_words(self, total_words=None):
        """Choose a larger resident chunk without a preliminary wordlist pass.

        Four streaming batches per resident chunk reduce kernel-launch/sync
        overhead substantially while remaining conservative about host/GPU
        memory.  The chunk is additionally bounded by OpenCL's maximum single
        allocation and the global dispatch-item ceiling.
        """
        try:
            max_alloc_bytes = self.device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
        except Exception:
            max_alloc_bytes = 0
        if max_alloc_bytes:
            words_per_alloc_limit = max_alloc_bytes // MAX_WORD_LEN
        else:
            words_per_alloc_limit = max(1, int(self.words_per_gpu_batch) * 4)
        chunk_words = min(max(1, int(self.words_per_gpu_batch) * 4),
                          words_per_alloc_limit, MAX_DISPATCH_ITEMS)
        if total_words is not None:
            chunk_words = min(chunk_words, max(1, int(total_words)))
        return max(1, int(chunk_words))

    def _preload_wordlist(self, wordlist_path):
        """Parse + encode the wordlist ONCE and keep it resident (GPU
        if it fits, host RAM otherwise) so score_batch() and
        apply_winner_and_clear() never touch disk or re-parse text
        again for the rest of the run. See class docstring."""
        t0 = time.time()
        chunk_words = self._resident_chunk_words()

        # Single pass: choose a conservative resident chunk size from device
        # limits, then retain the parsed chunks while counting them.  The old
        # implementation ran the mmap parser once merely to count words and
        # immediately ran it a second time to build the resident chunks.
        host_chunks = []
        total_words = 0
        for words_np, num_words in optimized_wordlist_iterator(
                wordlist_path, MAX_WORD_LEN, chunk_words):
            # optimized_wordlist_iterator() reuses/mutates its buffer
            # across yields except on the final partial chunk, so copy
            # defensively before holding onto it long-term.
            host_chunks.append((words_np.copy(), num_words))
            total_words += num_words

        total_bytes = sum(len(w) for w, _ in host_chunks)
        try:
            device_mem = self.device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
        except Exception:
            device_mem = 0
        budget_bytes = int(device_mem * self._GPU_RESIDENT_WORDLIST_FRACTION)

        fits_on_gpu = device_mem > 0 and total_bytes <= budget_bytes
        if fits_on_gpu:
            try:
                gpu_chunks = []
                mf = cl.mem_flags
                for words_np, num_words in host_chunks:
                    buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                     hostbuf=words_np)
                    gpu_chunks.append((buf, num_words))
                self._word_chunks_gpu = gpu_chunks
                log(f"{green('Wordlist preloaded to GPU:')} {cyan(f'{total_words:,}')} words, "
                    f"{cyan(f'{total_bytes / (1024**2):.0f} MB')} across "
                    f"{cyan(f'{len(gpu_chunks):,}')} chunk(s) of up to "
                    f"{cyan(f'{chunk_words:,}')} words each "
                    f"{dim(f'({time.time() - t0:.1f}s, one-time parse+upload)')}")
                return
            except cl.MemoryError:
                log(f"{yellow('GPU allocation for resident wordlist failed')} -- "
                    f"falling back to host-resident chunks (still avoids re-reading disk).")
                self._word_chunks_gpu = None

        self._word_chunks_host = host_chunks
        # Reusable upload buffer sized for these (larger) resident
        # chunks -- __init__'s base_words_g was sized for the old
        # streaming batch size and is too small once chunk_words has
        # been enlarged, so allocate a dedicated one here instead of
        # reusing/resizing base_words_g (which the disk-streaming
        # fallback path in _iter_word_chunks still relies on at its
        # original size).
        max_chunk_words = max((n for _, n in host_chunks), default=chunk_words)
        self._host_fallback_words_g = cl.Buffer(
            self.context, cl.mem_flags.READ_ONLY,
            max(1, max_chunk_words) * MAX_WORD_LEN * np.uint8().itemsize)
        log(f"{blue('Wordlist preloaded to host RAM:')} {cyan(f'{total_words:,}')} words, "
            f"{cyan(f'{total_bytes / (1024**2):.0f} MB')} across "
            f"{cyan(f'{len(host_chunks):,}')} chunk(s) of up to "
            f"{cyan(f'{chunk_words:,}')} words each "
            f"{dim(f'({time.time() - t0:.1f}s, one-time parse; ' + ('too large for GPU residency, ' if device_mem else '') + 'reused via H2D copy each pass)')}")

    def _iter_word_chunks(self, wordlist_path):
        """Yields (words_buffer_on_gpu, num_words) for every chunk of
        the wordlist, using whichever resident copy is available
        (GPU-resident buffers directly, or host-resident arrays copied
        into a reusable upload buffer sized for those chunks). Falls
        back to disk streaming only if preloading was never requested
        (wordlist_path was not passed to __init__ -- not used by
        celf_select_recompute_gpu(), kept only for standalone/testing
        use of this class)."""
        if self._word_chunks_gpu is not None:
            for buf, num_words in self._word_chunks_gpu:
                yield buf, num_words
        elif self._word_chunks_host is not None:
            for words_np, num_words in self._word_chunks_host:
                cl.enqueue_copy(self.queue, self._host_fallback_words_g, words_np)
                yield self._host_fallback_words_g, num_words
        else:
            for words_np, num_words in optimized_wordlist_iterator(
                    wordlist_path, MAX_WORD_LEN, self.words_per_gpu_batch):
                cl.enqueue_copy(self.queue, self.base_words_g, words_np)
                yield self.base_words_g, num_words


def _build_open_addressing_table(cracked_hashes_sorted, hash_table_size, hash_table_mask):
    """Build the uint64 cracked table plus compact active bitmap."""
    hash_table, states = build_open_addressing_table_uint64(
        np.asarray(cracked_hashes_sorted, dtype=np.uint64), int(hash_table_size))
    occupied_bool = states == 2
    occ_bytes = np.packbits(occupied_bool, bitorder='little')
    pad = (-len(occ_bytes)) % np.dtype(np.uint32).itemsize
    if pad:
        occ_bytes = np.concatenate([occ_bytes, np.zeros(pad, dtype=np.uint8)])
    occupied_words = occ_bytes.view(np.uint32).copy()
    return hash_table, occupied_words


class _GpuScorer(_ResidentWordlistMixin):
    """Owns the OpenCL context/buffers for one celf_select_recompute_gpu()
    run. `active` (the not-yet-covered bitmap) and `encoded` rules live
    on the GPU for the whole run; only word batches stream through (see
    _ResidentWordlistMixin), so the only thing that scales with n_rules
    is a small on-device buffer sized for one rule dispatch batch
    (rule_batch_size rows); the buffer holds int32 SCALARS per rule,
    not a full per-rule coverage row, so it is ~W/32 times smaller for the same
    rule_batch_size."""

    def __init__(self, encoded_rules, num_cracked, cracked_hashes_sorted,
                 rule_batch_size, words_per_gpu_batch, device_id=None,
                 wordlist_path=None, rule_lens=None):
        self.n_rules = encoded_rules.shape[0]
        self.num_cracked = num_cracked
        # Exact per-rule byte length (encoded_rules is a zero-padded
        # uint8 matrix, so this can't be recovered from a trailing-zero
        # scan alone if a rule's bytes happen to be shorter than
        # another's prefix). Used by the single-rule record/clear fast
        # path (_score_single_recording / apply_winner_and_clear)
        # instead of re-deriving it from the padded row each call.
        if rule_lens is None:
            rule_lens = (encoded_rules != 0).argmin(axis=1)
            rule_lens[encoded_rules[:, -1] != 0] = encoded_rules.shape[1]
        self.rule_lens = np.asarray(rule_lens, dtype=np.uint32)
        # The GPU uses an open-addressed hash table instead of binary-searching
        # the sorted cracked array for every word/rule pair. Keep <=50% load so
        # lookups are short and predictable. The active bitmap is indexed by
        # hash-table slot; only occupied slots are initialized to 1.
        table_size = 1
        target_size = max(2, int(math.ceil(num_cracked * 2.0)))
        while table_size < target_size:
            table_size <<= 1
        self.hash_table_size = table_size
        self.hash_table_mask = table_size - 1
        self.W = max(1, (table_size + 31) // 32)
        self.rule_batch_size = rule_batch_size
        self.words_per_gpu_batch = words_per_gpu_batch
        self.encoded = encoded_rules

        platform, device = select_device(device_id)
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)
        src = get_recompute_kernel_source(num_cracked, self.hash_table_size)
        prg = cl.Program(self.context, src).build()
        self.score_kernel = prg.score_against_active_kernel
        self.clear_kernel = prg.apply_and_clear_kernel
        self.record_kernel = prg.score_single_and_record_kernel
        self.exact_batch_kernel = prg.score_exact_batch_kernel
        self.clear_recorded_kernel = prg.clear_recorded_slots_kernel

        mf = cl.mem_flags
        # Build a compact GPU open-addressing table.  `occupied` is a bitset
        # rather than a sentinel value so all 2^64 FNV-1a fingerprints remain valid.
        # Vectorized (NumPy, wave-based) rather than a per-key Python loop --
        # see _build_open_addressing_table()'s docstring; this is the
        # dominant startup cost at multi-million-entry cracked lists
        # (e.g. ~14M) if done with a plain Python loop.
        t_hash0 = time.time()
        hash_table, occupied = _build_open_addressing_table(
            cracked_hashes_sorted, self.hash_table_size, self.hash_table_mask)
        log(f"{dim(f'Hash table built for {num_cracked:,} cracked entries in {time.time() - t_hash0:.2f}s (vectorized)')}")
        self.hash_table_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                      hostbuf=hash_table)
        self.hash_table_occupied_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                               hostbuf=occupied)
        # active bitmap is indexed by hash-table slot.  Empty table slots stay
        # zero, occupied slots start active. This preserves exact remaining
        # counts without needing a sorted-index lookup on the GPU.
        active_init = occupied.copy()
        self.active_g = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                                   hostbuf=active_init)

        words_buffer_size = words_per_gpu_batch * MAX_WORD_LEN * np.uint8().itemsize
        self.base_words_g = cl.Buffer(self.context, mf.READ_ONLY, words_buffer_size)

        rules_buffer_size = rule_batch_size * MAX_RULE_LEN * np.uint8().itemsize
        self.rules_g = cl.Buffer(self.context, mf.READ_ONLY, rules_buffer_size)

        self.gains_g = cl.Buffer(self.context, mf.READ_WRITE,
                                  rule_batch_size * np.int32().itemsize)

        self.single_rule_g = cl.Buffer(self.context, mf.READ_ONLY, MAX_RULE_LEN)
        self.cleared_count_g = cl.Buffer(self.context, mf.READ_WRITE, np.int32().itemsize)

        self.device = device
        self._word_chunks_gpu = None   # list[(cl.Buffer, num_words)] if resident on GPU
        self._word_chunks_host = None  # list[(np.ndarray, num_words)] if resident on host only
        self._host_fallback_words_g = None  # reusable upload buffer, sized for host-resident chunks
        if wordlist_path is not None:
            self._preload_wordlist(wordlist_path)

        # Buffers for the single-rule record/clear fast path (see
        # score_single_and_record_kernel / clear_recorded_slots_kernel
        # above): sized to the largest word chunk this scorer will
        # ever dispatch, since a chunk of N words can record at most N
        # hits. Allocated after preload so the real (possibly
        # enlarged-for-residency) chunk size is known; falls back to
        # words_per_gpu_batch for the disk-streaming path.
        if self._word_chunks_gpu is not None:
            max_chunk_words = max((n for _, n in self._word_chunks_gpu), default=words_per_gpu_batch)
        elif self._word_chunks_host is not None:
            max_chunk_words = max((n for _, n in self._word_chunks_host), default=words_per_gpu_batch)
        else:
            max_chunk_words = words_per_gpu_batch
        self.hit_slots_g = cl.Buffer(self.context, mf.READ_WRITE,
                                      max(1, max_chunk_words) * np.uint32().itemsize)
        self.hit_count_g = cl.Buffer(self.context, mf.READ_WRITE, np.int32().itemsize)
        # Reused host-side scratch for the single-candidate count readback
        # (one per CELF round, 50k+ times for a typical run) -- avoids a
        # fresh np.zeros(1) allocation on every call.
        self._single_count_host = np.empty(1, dtype=np.int32)
        # Micro-opt: _score_single_recording's hit_slots_g sub-region
        # depends only on num_words, which is constant across calls in
        # the (default) GPU-resident wordlist case -- cache it by
        # num_words instead of calling get_sub_region() fresh on every
        # one of the 50k+ seed_exact calls per run. Pure host-side
        # object-creation overhead, not round-trip latency, so the
        # expected win here is small (low single-digit %), not another
        # big structural fix like the batching ones.
        self._single_slots_region_cache = {}
        # EXACT_BATCH_CAP candidates can be exact/deduplicated-scored in
        # ONE score_exact_batch_kernel dispatch instead of one dispatch
        # each -- each gets its own words_per_rule_cap-sized segment of
        # exact_hit_slots_g, sized identically to hit_slots_g's single-rule
        # capacity (a rule can't hit more targets than there are words in
        # the chunk). This is purely a GPU round-trip-latency optimization:
        # it amortizes dispatch overhead across several CELF candidates
        # that each still need a true deduplicated gain this round;
        # correctness/dedup happens exactly as in _score_single_recording,
        # just for several rules' segments at once.
        self.exact_batch_cap = max(1, int(os.environ.get('RANKER_EXACT_BATCH', '64')))
        self._exact_words_per_rule_cap = max(1, max_chunk_words)
        self.exact_hit_slots_g = cl.Buffer(
            self.context, mf.READ_WRITE,
            self._exact_words_per_rule_cap * self.exact_batch_cap * np.uint32().itemsize)
        self.exact_hit_counts_g = cl.Buffer(
            self.context, mf.READ_WRITE, self.exact_batch_cap * np.int32().itemsize)
        # Reused host-side scratch for the counts readback in
        # score_exact_batch -- avoids a fresh allocation on every call.
        self._exact_counts_host = np.empty(self.exact_batch_cap, dtype=np.int32)
        # (rule_idx, slots ndarray) for the most recent SINGLE-rule
        # score_batch() call, if any -- consumed by apply_winner_and_clear()
        # when that same rule turns out to be the round's winner, and
        # invalidated (set to None) by any multi-rule score_batch() call or
        # once consumed/at the start of a round, since it's only valid
        # against the `active` set it was computed against.
        self._last_single_hits = None

    def remaining_active_count(self):
        host = np.empty(self.W, dtype=np.uint32)
        cl.enqueue_copy(self.queue, host, self.active_g).wait()
        return int(np.unpackbits(host.view(np.uint8)).sum())

    def score_batch(self, rule_indices, wordlist_path):
        """Scores len(rule_indices) rules (given as row indices into
        self.encoded) against the CURRENT active set. Returns
        (len(rule_indices),) int64 array of gains. May internally
        chunk if len(rule_indices) > self.rule_batch_size.

        All GPU commands below are enqueued WITHOUT an intermediate
        .wait() -- self.queue is an in-order queue, so the device
        already executes them in submission order without the host
        needing to block between steps. The only synchronization point
        is the final host_gains readback. Waiting after every single
        upload/fill/kernel-launch (as earlier versions of this method
        did) adds one full host<->device round trip per step for no
        correctness benefit, and that round-trip latency -- not GPU
        compute -- is what dominates a single-rule lazy-heap
        revalidation, since there's very little actual kernel work to
        hide it behind."""
        # NOTE: transfers/zeroing below are sized to the ACTUAL chunk
        # length `n`, not the allocated rule_batch_size. Most
        # score_batch() calls during lazy-heap revalidation only need
        # to rescore a small handful of candidates (that's the whole
        # point of the upper-bound pruning above this call) even
        # though rules_g/gains_g are sized for the worst case
        # (rule_batch_size, for the upper-bound pass in pass 1). An
        # earlier version always copied/zeroed/read back the full
        # rule_batch_size-sized region regardless of n, which meant a
        # 1-2 rule revalidation still paid for a 1024-row H2D copy, a
        # 1024-entry buffer fill, and a 1024-entry D2H readback -- pure
        # overhead on the hot path this strategy is supposed to keep
        # cheap. Bounding every transfer to n removes that overhead
        # entirely without changing kernel launch sizing, which was
        # already correctly bounded by n/sub_num.
        #
        # FAST PATH: a single-candidate rescore is the overwhelmingly
        # common case during greedy-round revalidation (see this
        # module's docstring -- most rounds need 0 or 1 dispatches,
        # and once a round does need one it's usually just the current
        # best candidate getting re-checked). For that case, use
        # score_single_and_record_kernel instead of score_kernel: same
        # cost (one pass over the resident wordlist), but it also
        # records every hit's exact hash-table slot. If this candidate
        # goes on to win its round, apply_winner_and_clear() below can
        # reuse that recorded list directly -- clearing just `gain`
        # slots -- instead of re-scanning the whole wordlist a second
        # time purely to rediscover positions already known here.
        if len(rule_indices) == 1:
            ridx = int(rule_indices[0])
            gain, slots = self._score_single_recording(ridx, wordlist_path)
            self._last_single_hits = (ridx, slots)
            return np.array([gain], dtype=np.int64)
        self._last_single_hits = None  # multi-rule batch: no single recorded candidate anymore
        out = np.zeros(len(rule_indices), dtype=np.int64)
        for cs in range(0, len(rule_indices), self.rule_batch_size):
            ce = min(cs + self.rule_batch_size, len(rule_indices))
            idx_chunk = rule_indices[cs:ce]
            n = len(idx_chunk)
            rules_batch_np = np.ascontiguousarray(self.encoded[idx_chunk])
            rules_region_g = self.rules_g.get_sub_region(0, n * MAX_RULE_LEN)
            gains_region_g = self.gains_g.get_sub_region(0, n * np.int32().itemsize)
            cl.enqueue_copy(self.queue, rules_region_g, rules_batch_np)
            cl.enqueue_fill_buffer(self.queue, gains_region_g, np.int32(0), 0,
                                    n * np.int32().itemsize)

            for words_g, num_words in self._iter_word_chunks(wordlist_path):
                rules_per_sub = max(1, min(n, MAX_DISPATCH_ITEMS // max(num_words, 1)))
                for sub_start in range(0, n, rules_per_sub):
                    sub_end = min(sub_start + rules_per_sub, n)
                    sub_num = sub_end - sub_start
                    sub_rules_g = self.rules_g.get_sub_region(
                        sub_start * MAX_RULE_LEN, sub_num * MAX_RULE_LEN)
                    sub_gains_g = self.gains_g.get_sub_region(
                        sub_start * np.int32().itemsize, sub_num * np.int32().itemsize)
                    global_size = (int(math.ceil(num_words * sub_num / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
                    self.score_kernel(self.queue, global_size, (LOCAL_WORK_SIZE,),
                                       words_g, sub_rules_g, self.hash_table_g,
                                       self.hash_table_occupied_g, self.active_g,
                                       sub_gains_g, np.uint32(num_words),
                                       np.uint32(sub_num), np.uint32(MAX_WORD_LEN),
                                       np.uint32(self.hash_table_mask))

            host_gains = np.zeros(n, dtype=np.int32)
            cl.enqueue_copy(self.queue, host_gains, gains_region_g).wait()
            out[cs:ce] = host_gains
        return out

    def _score_single_recording(self, rule_idx, wordlist_path):
        """Scores ONE rule (by row index into self.encoded) against the
        current active set using score_single_and_record_kernel, and
        returns (gain, slots) where `slots` is the (gain,) uint32 array
        of every hash-table slot it hit -- the information
        score_against_active_kernel's plain counting mode discards.
        Costs the same one wordlist pass score_kernel would have for a
        single rule; the recording is "free" (same kernel launch,
        extra output buffer) rather than an extra pass."""
        rule_len = int(self.rule_lens[rule_idx])
        cl.enqueue_copy(self.queue, self.single_rule_g,
                         np.ascontiguousarray(self.encoded[rule_idx]))
        slot_parts = []
        for words_g, num_words in self._iter_word_chunks(wordlist_path):
            hit_slots_region_g = self._single_slots_region_cache.get(num_words)
            if hit_slots_region_g is None:
                hit_slots_region_g = self.hit_slots_g.get_sub_region(
                    0, max(1, num_words) * np.uint32().itemsize)
                self._single_slots_region_cache[num_words] = hit_slots_region_g
            cl.enqueue_fill_buffer(self.queue, self.hit_count_g, np.int32(0), 0,
                                    np.int32().itemsize)
            global_size = (int(math.ceil(num_words / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
            self.record_kernel(self.queue, global_size, (LOCAL_WORK_SIZE,),
                                words_g, self.single_rule_g, self.hash_table_g,
                                self.hash_table_occupied_g, self.active_g,
                                hit_slots_region_g, self.hit_count_g,
                                np.uint32(num_words), np.uint32(rule_len),
                                np.uint32(MAX_WORD_LEN), np.uint32(self.hash_table_mask))
            # The slots copy is unavoidably dependent on the count (we
            # can't size/issue it before the kernel tells us how many
            # hits there were), so this is a genuine 2-round-trip
            # dependency chain, not an artifact of how the code is
            # written. An earlier version of this comment claimed
            # reading back a worst-case-sized (max_chunk_words) slots
            # buffer unconditionally would collapse this to one round
            # trip -- it does remove a .wait(), but at RANKER_EXACT_BATCH
            # / large-chunk scale it traded a cheap latency wait for an
            # expensive host-side np.zeros() allocation+memset AND a
            # PCIe transfer sized to the chunk instead of to the actual
            # (usually much smaller) hit count, which measured SLOWER
            # end-to-end. Reverted: read the count, then copy exactly
            # `cnt` elements into a reused (not freshly allocated) host
            # buffer, so the only recurring per-call cost is the two
            # waits plus a transfer proportional to real data.
            cl.enqueue_copy(self.queue, self._single_count_host, self.hit_count_g).wait()
            cnt = int(self._single_count_host[0])
            if cnt:
                host_slots = np.empty(cnt, dtype=np.uint32)
                cl.enqueue_copy(self.queue, host_slots,
                                 hit_slots_region_g.get_sub_region(
                                     0, cnt * np.uint32().itemsize)).wait()
                slot_parts.append(host_slots)
        # DEDUPLICATE across the whole wordlist before this becomes the
        # round's "exact" gain. Two different base words can both produce
        # the same cracked output under one rule (e.g. two words that
        # collapse to the same string after a truncate/lowercase rule, or
        # simply two independent words that happen to equal the same
        # cracked password); each is a separate recorded hit but they
        # cover the SAME target. Counting them separately -- as the raw
        # per-chunk `cnt` used to -- inflates the candidate's gain above
        # its true marginal coverage, exactly the gap rulest's
        # `_celf_select_from_upper_bounds` docstring calls out: only a
        # deduplicated score may ever decide a round's winner. The old
        # code here (and the batch score_against_active_kernel path
        # callers must not treat as exact) counted raw hits, which is why
        # a rule that happened to re-cover already-picked ground many
        # times over could out-rank a rule with genuinely higher unique
        # coverage.
        slots = (np.unique(np.concatenate(slot_parts)) if slot_parts
                 else np.zeros(0, dtype=np.uint32))
        total_gain = int(slots.shape[0])
        return total_gain, slots

    def score_exact_batch(self, rule_indices, wordlist_path):
        """Exact (deduplicated) gains for up to self.exact_batch_cap rules
        at once, in one GPU dispatch per wordlist chunk rather than one
        dispatch per rule. Returns a dict rule_idx -> (gain, slots), where
        `slots` is that rule's deduplicated hit-slot array (same shape/use
        as _score_single_recording's return, so it can feed the same
        apply_winner_and_clear() fast path if this rule goes on to win).

        Chunked exactly like _score_single_recording: hit_counts is reset
        every chunk and per-rule segments are deduplicated and
        concatenated across chunks before the final np.unique.
        """
        n = len(rule_indices)
        if n == 0:
            return {}
        if n > self.exact_batch_cap:
            raise ValueError(
                f"score_exact_batch got {n} rules, exceeds exact_batch_cap={self.exact_batch_cap}")
        cap = self._exact_words_per_rule_cap
        rules_batch_np = np.ascontiguousarray(self.encoded[np.asarray(rule_indices, dtype=np.int64)])
        rules_region_g = self.rules_g.get_sub_region(0, n * MAX_RULE_LEN)
        cl.enqueue_copy(self.queue, rules_region_g, rules_batch_np)

        slot_parts = {int(ridx): [] for ridx in rule_indices}
        counts_region_g = self.exact_hit_counts_g.get_sub_region(0, n * np.int32().itemsize)
        if self._exact_counts_host is None or self._exact_counts_host.shape[0] < n:
            self._exact_counts_host = np.empty(self.exact_batch_cap, dtype=np.int32)
        host_counts = self._exact_counts_host[:n]
        for words_g, num_words in self._iter_word_chunks(wordlist_path):
            cl.enqueue_fill_buffer(self.queue, counts_region_g, np.int32(0), 0,
                                    n * np.int32().itemsize)
            global_size = (int(math.ceil(num_words * n / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
            self.exact_batch_kernel(self.queue, global_size, (LOCAL_WORK_SIZE,),
                                     words_g, rules_region_g, self.hash_table_g,
                                     self.hash_table_occupied_g, self.active_g,
                                     self.exact_hit_slots_g, counts_region_g,
                                     np.uint32(num_words), np.uint32(n),
                                     np.uint32(cap), np.uint32(MAX_WORD_LEN),
                                     np.uint32(self.hash_table_mask))
            # Must know each rule's count before its slots copy can be
            # sized, so this first readback is a hard dependency -- one
            # wait, unavoidable.
            cl.enqueue_copy(self.queue, host_counts, counts_region_g).wait()
            # But the per-rule slots copies that follow do NOT depend on
            # each other, only on this same count readback -- so issue
            # all of them back-to-back on the in-order queue WITHOUT an
            # intermediate .wait() each (that per-candidate blocking is
            # what made this call latency-dominated: up to exact_batch_cap
            # serialized round trips), then block ONCE for all of them
            # together. Each copy is still sized to that rule's true cnt,
            # never to the cap -- unlike reading back the whole n*cap
            # region unconditionally, this keeps the transferred bytes
            # proportional to real data regardless of how large
            # exact_batch_cap is tuned.
            pending = []
            for local_i, ridx in enumerate(rule_indices):
                cnt = min(int(host_counts[local_i]), cap)  # capped segment, see kernel comment
                if cnt:
                    seg_g = self.exact_hit_slots_g.get_sub_region(
                        (local_i * cap) * np.uint32().itemsize, cnt * np.uint32().itemsize)
                    buf = np.empty(cnt, dtype=np.uint32)
                    evt = cl.enqueue_copy(self.queue, buf, seg_g)
                    pending.append((ridx, buf, evt))
            if pending:
                cl.wait_for_events([evt for _, _, evt in pending])
                for ridx, buf, _ in pending:
                    slot_parts[int(ridx)].append(buf)

        out = {}
        for ridx in rule_indices:
            ridx = int(ridx)
            parts = slot_parts[ridx]
            slots = np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.uint32)
            out[ridx] = (int(slots.shape[0]), slots)
        return out

    def apply_winner_and_clear(self, rule_idx, rule_str, wordlist_path):
        """Runs the winning rule once against the wordlist, clearing
        every cracked hash it (still) covers out of `active`. Returns
        the true number of bits actually cleared this round.

        FAST PATH: if `rule_idx` is the rule _score_single_recording()
        most recently scored (i.e. score_batch() was last called with
        just this one candidate -- the common case, see score_batch's
        docstring), its exact hit-slot list is already known, captured
        against this same, still-unchanged `active` set (nothing else
        can have modified `active` between that scoring call and this
        one -- rounds only mutate it here, once, for the chosen
        winner). Clearing those recorded slots directly with
        clear_recorded_slots_kernel costs one tiny launch over `gain`
        items instead of a full extra wordlist pass.

        Falls back to the original full-wordlist apply_and_clear_kernel
        pass whenever that cache doesn't apply: round 1 (winner is
        read directly off the upper-bound pass, never individually
        rescored), or a round whose winner came from a multi-rule
        rescore batch (no per-rule recording happens there)."""
        if self._last_single_hits is not None and self._last_single_hits[0] == rule_idx:
            _, slots = self._last_single_hits
            self._last_single_hits = None
            n_hits = len(slots)
            if n_hits == 0:
                return 0
            slots_g = cl.Buffer(self.context,
                                 cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                 hostbuf=np.ascontiguousarray(slots, dtype=np.uint32))
            cl.enqueue_fill_buffer(self.queue, self.cleared_count_g, np.int32(0), 0,
                                    np.int32().itemsize)
            global_size = (int(math.ceil(n_hits / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
            self.clear_recorded_kernel(self.queue, global_size, (LOCAL_WORK_SIZE,),
                                        self.active_g, slots_g, self.cleared_count_g,
                                        np.uint32(n_hits))
            host_count = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(self.queue, host_count, self.cleared_count_g).wait()
            return int(host_count[0])

        self._last_single_hits = None
        rb = rule_str.encode('latin-1', errors='ignore')[:MAX_RULE_LEN]
        rule_np = np.zeros(MAX_RULE_LEN, dtype=np.uint8)
        rule_np[:len(rb)] = np.frombuffer(rb, dtype=np.uint8)
        cl.enqueue_copy(self.queue, self.single_rule_g, rule_np)
        cl.enqueue_fill_buffer(self.queue, self.cleared_count_g, np.int32(0), 0,
                                np.int32().itemsize)

        for words_g, num_words in self._iter_word_chunks(wordlist_path):
            global_size = (int(math.ceil(num_words / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
            self.clear_kernel(self.queue, global_size, (LOCAL_WORK_SIZE,),
                               words_g, self.single_rule_g, self.hash_table_g,
                               self.hash_table_occupied_g, self.active_g,
                               self.cleared_count_g, np.uint32(num_words),
                               np.uint32(len(rb)), np.uint32(MAX_WORD_LEN),
                               np.uint32(self.hash_table_mask))

        host_count = np.zeros(1, dtype=np.int32)
        cl.enqueue_copy(self.queue, host_count, self.cleared_count_g).wait()
        return int(host_count[0])


def celf_select_recompute_gpu(rules, wordlist_path, cracked_hashes_sorted,
                               rule_batch_size, words_per_gpu_batch,
                               device_id=None, budget=None, cost_alpha=0.0):
    """Lazy-greedy max-coverage selection using a batched CELF heap.

    The previous implementation kept a fixed array sorted by the original
    full-target upper bound. As the active target set shrank, more candidates
    had to be revalidated because the round's best marginal gain decreased,
    causing more full-wordlist GPU rescans per selected rule.

    This version keeps the current upper bound of each candidate in a max-heap
    (implemented as min-heap of negative gains). Once a candidate is rescored,
    its exact gain becomes its new bound and is pushed back into the heap.
    A round stops as soon as the heap's next bound cannot beat the current
    exact winner. This preserves the CELF lazy-greedy invariant and the
    original deterministic tie-break: smaller rule index wins equal gain.

    cost_alpha enables cost-weighted CELF (ported from rulest's
    `_celf_select_from_upper_bounds`): the heap is ordered by
    gain / cost(rule) instead of raw gain, where cost(rule) grows with
    token count as max(1, len(rule.split())) ** cost_alpha. With the
    default cost_alpha=0.0 every rule costs 1.0 and this is byte-for-byte
    the old equal-cost greedy -- selection order and output are unchanged.
    With cost_alpha > 0, a cheaper rule that covers the same (or only
    marginally more) of the active target set is preferred over a pricier
    one, which is the one capability the old equal-cost greedy had no way
    to express at all.

    RANKER_GREEDY_RESCORE_BATCH controls speculative contenders rescored
    together after the first candidate (default 128, raised from an
    earlier default of 32 after profiling on an RTX 3060 Ti showed no
    VRAM or correctness regression at this size -- see
    RANKER_EXACT_BATCH below for the matching exact-resolve default).
    Smaller values reduce speculative work; larger values reduce
    kernel-launch overhead.

    RANKER_EXACT_BATCH (read in _GpuScorer.__init__) controls how many
    lazily-promoted candidates get resolved to an exact, deduplicated
    gain in one GPU dispatch (default 64, raised from 8). This only
    costs extra VRAM in exact_hit_slots_g, sized
    max_chunk_words * RANKER_EXACT_BATCH * 4 bytes -- for a ~130k-word
    resident chunk that's roughly 32 MB at 64 versus 4 MB at 8, trivial
    next to the cracked-hash table (which dominates VRAM use and scales
    with the cracked list size, not with either of these two knobs).
    Both defaults can still be overridden via their environment
    variables for memory-constrained GPUs or very different
    candidate-pool/wordlist sizes.
    """
    import heapq

    overlong = [r for r in rules if len(r.encode('latin-1', errors='ignore')) > MAX_RULE_LEN]
    if overlong:
        raise ValueError(
            f"{len(overlong)} candidate rules exceed MAX_RULE_LEN={MAX_RULE_LEN}; "
            "refusing to truncate rule bytes."
        )

    n_rules = len(rules)
    log(f"{blue('Recompute-GPU greedy select:')} {cyan(f'{n_rules:,}')} {bold('candidates,')} "
        f"{cyan(f'{len(cracked_hashes_sorted):,}')} {bold('cracked universe')}")

    encoded = np.zeros((n_rules, MAX_RULE_LEN), dtype=np.uint8)
    rule_lens = np.zeros(n_rules, dtype=np.uint32)
    for i, r in enumerate(rules):
        rb = r.encode('latin-1', errors='ignore')[:MAX_RULE_LEN]
        encoded[i, :len(rb)] = np.frombuffer(rb, dtype=np.uint8)
        rule_lens[i] = len(rb)

    t_scorer0 = time.time()
    scorer = _GpuScorer(encoded, len(cracked_hashes_sorted), cracked_hashes_sorted,
                         rule_batch_size, words_per_gpu_batch, device_id,
                         wordlist_path=wordlist_path, rule_lens=rule_lens)
    t_scorer_init = time.time() - t_scorer0

    # --- Profiling (RANKER_GREEDY_PROFILE=1) -----------------------------
    # Per-stage {count, total_seconds} so a tuning pass can see WHERE the
    # time actually goes -- GPU-dispatch latency (many small round-trips),
    # kernel compute time against a big resident wordlist, or Python/heap
    # overhead -- instead of guessing from the overall rules/s figure.
    # Overhead when disabled is one bool check per call site.
    profiling = os.environ.get('RANKER_GREEDY_PROFILE', '0') == '1'
    prof = {} if profiling else None
    resolve_batch_sizes = [] if profiling else None

    def _prof(name, dt, n=1):
        if prof is None:
            return
        c, t = prof.get(name, (0, 0.0))
        prof[name] = (c + n, t + dt)

    class _Timed:
        __slots__ = ('name', 'n', 't0')
        def __init__(self, name, n=1):
            self.name = name
            self.n = n
        def __enter__(self):
            if profiling:
                self.t0 = time.time()
            return self
        def __exit__(self, *exc):
            if profiling:
                _prof(self.name, time.time() - self.t0, self.n)

    if profiling:
        log(f"{dim(f'[CELF profile] scorer init (hash table + wordlist preload): {t_scorer_init:.2f}s')}")

    # Pass 1: exact score against the full target set. These values are valid
    # upper bounds for every later round because coverage is monotone.
    upper_bound = np.zeros(n_rules, dtype=np.int64)
    total_batches = math.ceil(n_rules / rule_batch_size)
    pbar = ProgressBar(total=total_batches, desc=cyan("Upper-bound pass (rule batches)"),
                unit="batch")
    for start in range(0, n_rules, rule_batch_size):
        end = min(start + rule_batch_size, n_rules)
        idx_chunk = np.arange(start, end)
        with _Timed("upper_bound_pass", n=end - start):
            upper_bound[start:end] = scorer.score_batch(idx_chunk, wordlist_path)
        pbar.update(1)
    pbar.close()

    # costs[index] is read every round by priority()/push(), including for
    # rules that start out with bound=0 and never enter the heap, so it has
    # to stay a full-length, O(1)-indexable sequence either way -- but it
    # skips all per-rule Python work in the (default) cost_alpha<=0 case,
    # where cost is always 1.0 and priority == raw gain.
    if cost_alpha <= 0:
        costs = _ConstantCosts(1.0)
    else:
        costs = [_rule_cost(r, cost_alpha) for r in rules]

    # Max-heap by gain/cost priority; smaller rule index wins ties naturally
    # because heap entries are (-priority, rule_index, raw_gain). A
    # candidate's exact gain replaces its older, looser bound after it is
    # rescored. The raw gain is carried alongside the priority so winner
    # bookkeeping (cleared counts, apply_winner_and_clear) always deals in
    # true coverage counts, never in cost-normalized scores.
    candidate_heap = [
        (-int(upper_bound[i]) / costs[i], int(i), int(upper_bound[i]))
        for i in range(n_rules)
        if upper_bound[i] > 0
    ]
    heapq.heapify(candidate_heap)
    n_active = len(candidate_heap)
    log(f"{green('Candidates with >=1 hit:')} {cyan(f'{n_active:,}')}/{cyan(f'{n_rules:,}')}")

    limit = budget if budget else n_active
    greedy_rescore_batch = max(
        1,
        int(os.environ.get('RANKER_GREEDY_RESCORE_BATCH', '128')),
    )
    greedy_rescore_batch = min(
        greedy_rescore_batch,
        max(1, int(rule_batch_size)),
    )

    selected = []
    round_num = 0
    total_rescored_rules = 0
    round_t0 = time.time()

    pbar = ProgressBar(total=limit, desc=cyan("CELF greedy select [recompute-gpu+heap]"),
                unit="rule")

    def _can_beat(bound, idx, cur_priority, cur_idx):
        p = bound / costs[idx]
        return p > cur_priority or (p == cur_priority and idx < cur_idx)

    has_exact_batch = hasattr(scorer, 'score_exact_batch')
    exact_resolve_cap = max(1, int(getattr(scorer, 'exact_batch_cap', 1)))

    while len(selected) < limit and candidate_heap:
        round_num += 1
        dispatches_this_pick = 0
        rescored_this_pick = 0

        # Seed this round with one EXACT (deduplicated) evaluation -- never
        # trust the pass-1 upper_bound or a batch-tightened bound as the
        # round winner outright, even for round 1 and even for the heap's
        # current top: score_against_active_kernel (used for the upper-bound
        # pass and for multi-rule batch rescoring below) counts every
        # (word, rule) hit on an active target, so it can overcount a
        # candidate whose rule maps several different words onto the same
        # still-uncovered cracked password. Only _score_single_recording's
        # deduplicated slot set -- what score_batch() of length 1 now
        # returns -- is trustworthy as an exact gain. This mirrors rulest's
        # `_celf_select_from_upper_bounds`, which never lets a batch bound
        # pick a winner either; it only ever tightens the heap.
        _neg_priority, seed_idx, _bound = heapq.heappop(candidate_heap)
        seed_idx = int(seed_idx)
        with _Timed("seed_exact"):
            seed_gain = int(scorer.score_batch(
                np.asarray([seed_idx], dtype=np.int64), wordlist_path)[0])
        dispatches_this_pick += 1
        rescored_this_pick += 1
        total_rescored_rules += 1

        best_idx = seed_idx
        best_gain = seed_gain
        best_priority = best_gain / costs[best_idx]
        # Indices whose heap entry is currently a batch-tightened bound,
        # not yet an exact score -- they must be singly exact-scored before
        # they are allowed to beat/replace best_idx this round.
        fresh_this_round = set()

        while candidate_heap:
            top_bound = int(candidate_heap[0][2])
            top_idx = int(candidate_heap[0][1])
            # No unseen candidate can beat the exact winner's priority. For
            # equal priority, only a smaller original index could replace it.
            if not _can_beat(top_bound, top_idx, best_priority, best_idx):
                break

            if top_idx in fresh_this_round:
                # This candidate's heap entry is a batch bound, not an
                # exact score yet. Gather a small group of fresh
                # contenders that still beat the current best and resolve
                # them in ONE dedup-exact dispatch (score_exact_batch)
                # instead of one GPU round-trip per candidate -- the
                # per-candidate version of this is what made round
                # latency dominate after the correctness fix (every
                # promoted candidate used to cost a full wordlist-pass
                # round-trip on its own). Correctness is unchanged: each
                # resolved candidate still gets a true deduplicated gain
                # before it is allowed to contend for the win, same as a
                # single score_batch([idx]) call would give.
                resolve_batch = []
                while candidate_heap and len(resolve_batch) < exact_resolve_cap:
                    top_bound = int(candidate_heap[0][2])
                    top_idx = int(candidate_heap[0][1])
                    if top_idx not in fresh_this_round:
                        break
                    if not _can_beat(top_bound, top_idx, best_priority, best_idx):
                        break
                    heapq.heappop(candidate_heap)
                    fresh_this_round.discard(top_idx)
                    resolve_batch.append(top_idx)
                if not resolve_batch:
                    continue
                if profiling:
                    resolve_batch_sizes.append(len(resolve_batch))

                if has_exact_batch:
                    with _Timed("fresh_resolve_exact_batch", n=len(resolve_batch)):
                        exact = scorer.score_exact_batch(resolve_batch, wordlist_path)
                    dispatches_this_pick += 1
                else:
                    # Fallback for scorers without score_exact_batch (e.g.
                    # the pure-Python test fake, which is already exact
                    # per-call and not GPU-latency bound).
                    exact = {}
                    with _Timed("fresh_resolve_fallback", n=len(resolve_batch)):
                        for ridx in resolve_batch:
                            g = int(scorer.score_batch(
                                np.asarray([ridx], dtype=np.int64), wordlist_path)[0])
                            exact[ridx] = (g, None)
                    dispatches_this_pick += len(resolve_batch)
                rescored_this_pick += len(resolve_batch)
                total_rescored_rules += len(resolve_batch)

                for ridx in resolve_batch:
                    gain, slots = exact[ridx]
                    priority = gain / costs[ridx]
                    if priority > best_priority or (
                        priority == best_priority and ridx < best_idx
                    ):
                        if best_gain > 0:
                            heapq.heappush(
                                candidate_heap, (-best_priority, best_idx, best_gain))
                        best_idx, best_gain, best_priority = ridx, gain, priority
                        if slots is not None:
                            scorer._last_single_hits = (ridx, slots)
                    elif gain > 0:
                        heapq.heappush(candidate_heap, (-priority, ridx, gain))
                continue

            batch = []
            while candidate_heap and len(batch) < greedy_rescore_batch:
                top_bound = int(candidate_heap[0][2])
                top_idx = int(candidate_heap[0][1])
                if not _can_beat(top_bound, top_idx, best_priority, best_idx):
                    break
                if top_idx in fresh_this_round:
                    break
                heapq.heappop(candidate_heap)
                batch.append(top_idx)

            if not batch:
                continue

            # score_against_active_kernel's raw per-rule counts: admissible
            # (never smaller than true marginal gain) but possibly inflated
            # by duplicate word->target hits, so these only TIGHTEN each
            # candidate's bound -- they never get compared against
            # best_priority directly, and never replace best_idx.
            with _Timed("batch_tighten", n=len(batch)):
                gains = scorer.score_batch(
                    np.asarray(batch, dtype=np.int64), wordlist_path)
            dispatches_this_pick += 1
            rescored_this_pick += len(batch)
            total_rescored_rules += len(batch)

            for local_i, ridx in enumerate(batch):
                ridx = int(ridx)
                gain = int(gains[local_i])
                if gain > 0:
                    heapq.heappush(candidate_heap, (-gain / costs[ridx], ridx, gain))
                    fresh_this_round.add(ridx)

        if best_gain <= 0:
            break

        with _Timed("apply_winner_and_clear"):
            true_cleared = scorer.apply_winner_and_clear(
                best_idx, rules[best_idx], wordlist_path)
        selected.append((rules[best_idx], true_cleared))
        n_active -= 1

        pbar.update(1)
        round_elapsed = time.time() - round_t0
        round_t0 = time.time()
        pbar.set_postfix({
            "round": round_num,
            "gpu_dispatches": dispatches_this_pick,
            "rescored": rescored_this_pick,
            "heap": n_active,
            "s/round": f"{round_elapsed:.1f}",
        })

    pbar.close()

    remaining = scorer.remaining_active_count()
    total_covered = len(cracked_hashes_sorted) - remaining
    if profiling:
        log(f"{blue('Greedy profile:')} {cyan(f'{total_rescored_rules:,}')} "
            f"candidate rescoring(s) after the initial pass; "
            f"rescore_batch={cyan(str(greedy_rescore_batch))}, "
            f"exact_batch_cap={cyan(str(exact_resolve_cap))}.")

        total_time = sum(t for _, t in prof.values())
        log(f"{blue('--- CELF stage breakdown (RANKER_GREEDY_PROFILE) ---')}")
        for name in ("upper_bound_pass", "seed_exact", "fresh_resolve_exact_batch",
                     "fresh_resolve_fallback", "batch_tighten", "apply_winner_and_clear"):
            if name not in prof:
                continue
            count, secs = prof[name]
            pct = (secs / total_time * 100) if total_time else 0.0
            avg_ms = (secs / count * 1000) if count else 0.0
            log(f"  {cyan(name):<32} calls={count:>7,}  total={secs:>7.1f}s "
                f"({pct:4.1f}%)  avg={avg_ms:7.2f}ms/call")
        log(f"  {cyan('TOTAL (instrumented)'):<32} {dim(f'{total_time:.1f}s')}")

        if resolve_batch_sizes:
            sizes = np.asarray(resolve_batch_sizes)
            at_cap = int((sizes >= exact_resolve_cap).sum())
            resolve_msg = (
                f"[CELF profile] fresh-resolve batches: n={len(sizes):,}, "
                f"mean size={sizes.mean():.1f}, at cap={at_cap:,}/{len(sizes):,}"
            )
            log(dim(resolve_msg))
            if len(sizes) and at_cap / len(sizes) > 0.3:
                hint_msg = (
                    "[CELF profile] >30% of fresh-resolve batches hit exact_batch_cap -- "
                    f"try a higher RANKER_EXACT_BATCH (currently {exact_resolve_cap}) "
                    "to cut dispatch count further."
                )
                log(yellow(hint_msg))
            elif len(sizes) and sizes.mean() < max(2, exact_resolve_cap / 4):
                hint_msg = (
                    "[CELF profile] fresh-resolve batches are small relative to "
                    "exact_batch_cap -- GPU round-trip latency is likely not the "
                    "bottleneck here; look at batch_tighten/upper_bound_pass instead."
                )
                log(dim(hint_msg))

        if "batch_tighten" in prof and "fresh_resolve_exact_batch" in prof:
            bt_count, bt_secs = prof["batch_tighten"]
            fr_count, fr_secs = prof["fresh_resolve_exact_batch"]
            if bt_count and fr_count:
                bt_ms = bt_secs / bt_count * 1000
                fr_ms = fr_secs / fr_count * 1000
                profile_msg = (
                    f"[CELF profile] avg batch_tighten dispatch: {bt_ms:.2f}ms/candidate vs "
                    f"avg fresh_resolve_exact_batch dispatch: {fr_ms:.2f}ms/candidate"
                )
                log(dim(profile_msg))

    log(f"{green('Done.')} {bold('Selected')} {cyan(f'{len(selected):,}')} {bold('rules,')} "
        f"{bold('covering')} {cyan(f'{total_covered:,}')}/{cyan(f'{len(cracked_hashes_sorted):,}')} "
        f"{bold('cracked entries')} -- {dim('selection state: O(n_rules) + O(universe bits)')}")
    return selected

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Memory-light GPU recompute+lazy-greedy max-coverage selection.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('-r', '--ranking-csv', help="ranker output CSV (Rule_Data/Combined_Score columns)")
    src.add_argument('-f', '--rules-file', help="Plain .rule file (already ranked/optimized)")
    ap.add_argument('-w', '--wordlist', required=True)
    ap.add_argument('-k', '--cracked', required=True, help="Cracked passwords list")
    ap.add_argument('-o', '--output', required=True, help="Output .rule path")
    ap.add_argument('-c', '--candidates', type=int, default=20000,
                     help="How many top-scored rules to feed into greedy select (default 20000).")
    ap.add_argument('-b', '--budget', type=int, default=None,
                     help="Max rules in final selection (default: run to saturation).")
    ap.add_argument('-B', '--budgets', type=str, default=None,
                     help="Comma-separated budget cutoffs exported as separate files, e.g. '64,250,5000'.")
    ap.add_argument('-R', '--rule-batch-size', type=int, default=1024,
                     help="Candidate rules scored per GPU dispatch during the "
                          "upper-bound pass, AND the chunk size the greedy-"
                          "select round loop scans candidates in (see "
                          "'gpu_dispatches' in the greedy-select progress "
                          "bar -- most rounds should need 0 or 1, since a "
                          "round's scan stops as soon as the best remaining "
                          "candidate's upper bound can't beat what's already "
                          "been found; a value staying high most rounds "
                          "means genuinely many close candidates need "
                          "rescoring, not that this is set wrong). Raise "
                          "this to amortize each GPU dispatch's near-fixed "
                          "launch cost across more candidates when dispatches "
                          "ARE needed.")
    ap.add_argument('-W', '--words-batch-size', type=int, default=DEFAULT_WORDS_PER_GPU_BATCH)
    ap.add_argument('--max-word-len', type=int, default=MAX_WORD_LEN)
    ap.add_argument('-d', '--device', type=int, default=None)
    args = ap.parse_args(argv)

    class _Ns:
        pass
    ns = _Ns()
    ns.rules_file = args.rules_file
    ns.ranking_csv = args.ranking_csv
    ns.candidates = args.candidates
    rules = load_candidate_rules(ns)

    cracked_hashes_sorted, n_skipped = load_cracked_universe(args.cracked, args.max_word_len)
    if n_skipped:
        log(f"{yellow('Warning:')} skipped {cyan(f'{n_skipped:,}')} cracked entries longer than --max-word-len")

    selected = celf_select_recompute_gpu(
        rules, args.wordlist, cracked_hashes_sorted,
        rule_batch_size=args.rule_batch_size,
        words_per_gpu_batch=args.words_batch_size,
        device_id=args.device,
        budget=args.budget,
    )

    budgets = parse_budgets(args.budgets) if args.budgets else None
    if budgets:
        save_output_multi(selected, args.output, budgets)
    else:
        save_output(selected, args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
