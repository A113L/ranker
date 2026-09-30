#!/usr/bin/env python3
"""
celf_recompute_gpu.py -- memory-light GPU greedy coverage selection
=====================================================================
Alternative to celf_postprocess.py's compute_coverage_bitmaps() +
celf_select(), using a recompute-then-select greedy strategy instead
of a memoize-then-select one.

WHY THIS EXISTS
----------------
celf_postprocess.py's coverage-evaluation stage materializes a
(n_candidates x ceil(cracked_universe/32)) uint32 bitmap matrix -- one
row per candidate rule, one bit per cracked-password target. For large
inputs (tens of thousands of candidates x millions of cracked hashes)
that matrix is tens to hundreds of GB. The existing code already
mitigates this heavily (streamed to a memmap, hybrid dense/sparse
row packing, --in-ram opt-out), but the *shape* of the algorithm is
unchanged: it is CELF (Cost-Effective Lazy Forward selection), which
is a *memoize-then-select* algorithm -- it computes and stores each
candidate's full coverage vector once, then does lazy-greedy picks by
re-reading rows from that stored matrix. Memory (RAM or disk) scales
with n_candidates x cracked_universe.

A recompute-then-select greedy solves the same "pick the next best
rule" problem completely differently from CELF's memoize-then-select
approach: it never stores a coverage matrix at all. Each round it
re-scores every still-viable candidate rule directly on the GPU
against the CURRENT (shrinking) uncracked target set, keeps only the
round's single best rule, removes the words it covered from the
target set, and repeats. It prunes most rules from re-evaluation with
a classic lazy-greedy bound: a rule's ORIGINAL (full-target) hit count
is a monotonically non-increasing upper bound on what it can score in
any later round (targets only ever shrink), so once a round's current
best is found, any untested rule whose upper bound is <= that best
can't possibly beat it and is skipped without a GPU launch.

This module implements that recompute + lazy-upper-bound strategy in
rule_ranker, reusing this package's own GPU rule-application kernel
(the same apply_hashcat_rule() transform celf_postprocess.py's
coverage kernel uses, so results match). Hash membership uses a GPU
open-addressing table rather than a per-hit binary search through the
sorted cracked array, substantially reducing random global-memory reads.
The only state kept between
rounds is:
  - `active` : a single (W,) uint32 bitmap of cracked hashes NOT YET
    covered (W = ceil(cracked_universe/32) words -- e.g. ~1.2 MB for a
    10M-entry universe, however many candidate rules there are).
  - `upper_bound` : (n_rules,) int32 -- each rule's static full-target
    popcount, computed ONCE in a single streaming pass (same shape as
    compute_coverage_bitmaps()'s pass, but accumulating one atomic
    counter per rule instead of writing a whole bitmap row).
  - `order`/`last_bound`/`excluded`, three (n_rules,)-shaped arrays
    that together play the same role as celf_select()'s lazy heap
    (see celf_select_recompute_gpu()'s block comment for why this is a
    fixed sorted array with two upper bounds rather than an actual
    heapq, which an earlier version of this module used) -- either
    way, "revalidating" a candidate means one GPU rescore of that rule
    against the current wordlist (cheap: one rule x the wordlist, or a
    batch of several at once) instead of a memmap row read.

Nothing shaped (n_rules x cracked_universe) is ever allocated, in RAM
or on disk. Peak extra memory beyond the wordlist/rule pool is
O(n_rules) + O(cracked_universe bits) + O(wordlist), since the
wordlist itself is now kept resident (GPU memory if it fits, host RAM
otherwise) rather than re-streamed from disk on every rescore -- see
_GpuScorer's docstring below for why.

Trade-off vs. the matrix approach: this recomputes each surviving
candidate's coverage from scratch (a GPU pass over the wordlist) every
time the lazy heap needs to revalidate it, instead of reading a
precomputed row. That is more GPU compute, but GPU rule-application +
hashing over a wordlist is exactly what this package's OpenCL kernel
is already fast at (it's the same kernel ranker.py's exhaustive/MAB
scoring stages use), and it fully replaces slow, memory/IO-bound
random reads against a huge on-disk matrix -- which celf_select()'s
own comments identify as its actual bottleneck at scale ("CELF is
I/O-bound on random reads of the coverage matrix, not CPU-bound").
For candidate pools where the coverage matrix would be very large but
the final --budget is much smaller than n_candidates (the common
case), this mode is both lower-memory AND faster in practice, since
CELF's lazy bound means most rounds validate in one GPU rescore and
most candidates are pruned by `upper_bound` without ever touching the
GPU again.

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
pipeline as an alternative to compute_coverage_bitmaps()+celf_select().
"""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import pyopencl as cl
from tqdm import tqdm

from .ranker_postprocess import (
    MAX_WORD_LEN, MAX_OUTPUT_LEN, MAX_RULE_LEN, LOCAL_WORK_SIZE,
    DEFAULT_WORDS_PER_GPU_BATCH, MAX_DISPATCH_ITEMS,
    log, red, green, yellow, blue, cyan, bold, dim, get_rss_mb,
    optimized_wordlist_iterator, load_cracked_universe, load_candidate_rules,
    select_device, save_output, parse_budgets, save_output_multi,
)

# The full apply_hashcat_rule()/apply_single_command()/
# binary_search_cracked() C source is identical to the one in
# ranker_postprocess.get_celf_kernel_source(); it is not re-derived
# here, only re-templated with two different __kernel entry points
# (score-against-active-set, and apply-and-clear-for-the-winner)
# instead of celf_coverage_kernel's bitmap-row write. Keeping this as
# a separate string (rather than string-surgery on the other module's
# kernel) keeps each kernel source independently readable/buildable.
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
unsigned int fnv1a_hash_32(const unsigned char* data, unsigned int len) {
    unsigned int hash = 2166136261U;
    for (unsigned int i = 0; i < len; i++) { hash ^= data[i]; hash *= 16777619U; }
    return hash;
}
static void duplicate_front(const unsigned char* in, int in_len,
                            unsigned char* out, int* out_len, int* changed, int n) {
    if (n > in_len) n = in_len;
    int new_len = in_len + n;
    if (new_len > MAX_OUTPUT_LEN) return;
    for (int i = 0; i < n; i++) out[i] = in[i];
    for (int i = 0; i < in_len; i++) out[n + i] = in[i];
    *out_len = new_len; *changed = 1;
}
static void duplicate_back(const unsigned char* in, int in_len,
                           unsigned char* out, int* out_len, int* changed, int n) {
    if (n > in_len) n = in_len;
    int new_len = in_len + n;
    if (new_len > MAX_OUTPUT_LEN) return;
    for (int i = 0; i < in_len; i++) out[i] = in[i];
    for (int i = 0; i < n; i++) out[in_len + i] = in[in_len - n + i];
    *out_len = new_len; *changed = 1;
}
static void duplicate_word(const unsigned char* in, int in_len,
                           unsigned char* out, int* out_len, int* changed, int times) {
    int new_len = in_len * (times + 1);
    if (new_len > MAX_OUTPUT_LEN) return;
    for (int rep = 0; rep <= times; rep++)
        for (int i = 0; i < in_len; i++) out[rep * in_len + i] = in[i];
    *out_len = new_len; *changed = 1;
}
static void rotate_left(const unsigned char* in, int in_len,
                        unsigned char* out, int* out_len, int* changed, int n) {
    if (n <= 0) n = 1;
    n %= in_len;
    *out_len = in_len;
    if (n == 0) { for (int i = 0; i < in_len; i++) out[i] = in[i]; *changed = 0; return; }
    for (int i = 0; i < in_len; i++) out[i] = in[(i + n) % in_len];
    *changed = 1;
}
static void rotate_right(const unsigned char* in, int in_len,
                         unsigned char* out, int* out_len, int* changed, int n) {
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
            case 'd': if (in_len*2<=MAX_OUTPUT_LEN) { *out_len=in_len*2; for(int i=0;i<in_len;i++){out[i]=in[i];out[in_len+i]=in[i];} changed=1; } break;
            case 'f': if (in_len*2<=MAX_OUTPUT_LEN) { *out_len=in_len*2; for(int i=0;i<in_len;i++){out[i]=in[i];out[in_len+i]=in[in_len-1-i];} changed=1; } break;
            case 'k': *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; if(in_len>=2){out[0]=in[1];out[1]=in[0];changed=1;} break;
            case 'K': *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; if(in_len>=2){out[in_len-2]=in[in_len-1];out[in_len-1]=in[in_len-2];changed=1;} break;
            case ':': *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; changed=0; break;
            case 'q': if (in_len*2<=MAX_OUTPUT_LEN) { int idx=0; for(int i=0;i<in_len;i++){out[idx++]=in[i];out[idx++]=in[i];} *out_len=in_len*2; changed=1; } break;
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
            case '^': if (in_len+1<=MAX_OUTPUT_LEN){ out[0]=arg; for(int i=0;i<in_len;i++) out[i+1]=in[i]; *out_len=in_len+1; changed=1; } break;
            case '$': if (in_len+1<=MAX_OUTPUT_LEN){ for(int i=0;i<in_len;i++) out[i]=in[i]; out[in_len]=arg; *out_len=in_len+1; changed=1; } break;
            case '@': *out_len=0; for(int i=0;i<in_len;i++){ if(in[i]!=arg) out[(*out_len)++]=in[i]; else changed=1; } break;
            case '!': for(int i=0;i<in_len;i++) if(in[i]==arg) return -1; *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; return 0;
            case '/': for(int i=0;i<in_len;i++) if(in[i]==arg){ *out_len=in_len; for(int j=0;j<in_len;j++) out[j]=in[j]; return 0; } return -1;
            case '(': if (in_len>0&&in[0]==arg){ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; return 0; } return -1;
            case ')': if (in_len>0&&in[in_len-1]==arg){ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; return 0; } return -1;
            case 'y': if (n>=0) duplicate_front(in,in_len,out,out_len,&changed,n); break;
            case 'Y': if (n>=0) duplicate_back(in,in_len,out,out_len,&changed,n); break;
            case 'z': if (n>0 && in_len+n<=MAX_OUTPUT_LEN){ out[0]=in[0]; for(int i=0;i<n;i++) out[i+1]=in[0]; for(int i=1;i<in_len;i++) out[n+i]=in[i]; *out_len=in_len+n; changed=1; } break;
            case 'Z': if (n>0 && in_len+n<=MAX_OUTPUT_LEN){ for(int i=0;i<in_len;i++) out[i]=in[i]; for(int i=0;i<n;i++) out[in_len+i]=in[in_len-1]; *out_len=in_len+n; changed=1; } break;
            case 'p': if (n>=0) duplicate_word(in,in_len,out,out_len,&changed,n); break;
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
            case 'i': if (n1>=0&&in_len+1<=MAX_OUTPUT_LEN){ int p=n1; if(p>in_len) p=in_len; for(int i=0;i<p;i++) out[i]=in[i]; out[p]=a2; for(int i=p;i<in_len;i++) out[i+1]=in[i]; *out_len=in_len+1; changed=1; } break;
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
        if (result == -1) { *out_len = 0; *changed = -1; return; }
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
// Binary search over the sorted cracked-hash array: O(log2(N)) random
// global-memory reads per lookup. Kept here (alongside the newer
// lookup_cracked_slot() hash-table probe below) because it's a real
// extern dependency of sparse_coverage.py's kernels, which are built
// around a plain sorted cracked_hashes_sorted buffer rather than the
// open-addressing table celf_recompute_gpu's own kernels use -- both
// modules import this same _COMMON_KERNEL_BODY string, so both lookup
// styles need to be present for whichever module's kernel calls them.
// Dead code (never called) from celf_recompute_gpu's own kernels,
// which use lookup_cracked_slot() instead for its O(1)-probe win.
int binary_search_cracked(__global const unsigned int* sorted_hashes, unsigned int n, unsigned int key) {
    int lo = 0, hi = (int)n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        unsigned int v = sorted_hashes[mid];
        if (v == key) return mid;
        if (v < key) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}
// Open-addressed hash table lookup.  The old implementation used a
// binary search over the sorted cracked-hash array (O(log2(N)) random
// global-memory reads for every word/rule pair).  On GPUs that random
// memory traffic becomes the dominant cost once rule transforms are cheap.
// The table is deliberately kept at <=50% load, so a lookup normally needs
// only 1-2 probes.  `occupied` is separate because every uint32 hash value is
// valid, including 0 and UINT_MAX.
int lookup_cracked_slot(__global const unsigned int* hash_table,
                        __global const unsigned int* occupied,
                        unsigned int table_mask, unsigned int key) {
    unsigned int slot = key * 2654435761U;
    slot &= table_mask;
    unsigned int word = slot >> 5;
    unsigned int bit = slot & 31U;
    for (unsigned int probe = 0; probe <= table_mask; probe++) {
        if (occupied[word] & (1U << bit)) {
            if (hash_table[slot] == key) return (int)slot;
        } else {
            return -1;
        }
        slot = (slot + 1U) & table_mask;
        word = slot >> 5;
        bit = slot & 31U;
    }
    return -1;
}
"""


def get_recompute_kernel_source(num_cracked, hash_table_size):
    """Two kernels sharing the same rule-transform/hash/binary-search
    plumbing as celf_postprocess.get_celf_kernel_source(), but neither
    one ever writes a per-rule bitmap ROW:

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
// gains: one int32 accumulator per rule in this dispatch batch,
// caller zeroes it before each rule-batch (same convention as
// celf_coverage_kernel's bitmap zeroing in ranker_postprocess.py).
__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void score_against_active_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned char* rules_in,
    __global const unsigned int* hash_table,
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

    unsigned int h = fnv1a_hash_32(result_temp, (unsigned int)out_len);
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
    __global const unsigned int* hash_table,
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

    unsigned int h = fnv1a_hash_32(result_temp, (unsigned int)out_len);
    int slot = lookup_cracked_slot(hash_table, hash_table_occupied, table_mask, h);
    if (slot < 0) return;

    unsigned int word_pos = (unsigned int)slot >> 5;
    unsigned int bit_pos = (unsigned int)slot & 31U;
    unsigned int mask = (1U << bit_pos);
    unsigned int old = atomic_and(&active[word_pos], ~mask);
    if (old & mask) atomic_add(cleared_count, 1);
}}
"""


class _ResidentWordlistMixin:
    """Parse-once, stay-resident wordlist handling, shared by every GPU
    backend in this package that needs to run MANY dispatches over the
    same wordlist across a run (celf_recompute_gpu.py's `_GpuScorer`,
    and sparse_coverage.py's coverage-extraction backend).

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
    # the whole encoded wordlist resident, leaving room for whatever
    # else the subclass's buffers need (cracked-hash array, coverage/
    # active buffers, rule batch buffers, driver overhead, ...).
    # Conservative on purpose -- falling back to the host-resident path
    # is still a large win over per-call disk streaming, so there's no
    # need to cut this close.
    _GPU_RESIDENT_WORDLIST_FRACTION = 0.78

    def _resident_chunk_words(self, total_words):
        """How many words to pack into each resident chunk when
        preloading. The original 150,000-word default
        (--words-batch-size) was sized for STREAMING batches off disk
        within a fixed memory budget; it has nothing to do with how
        large a single GPU dispatch can be. Once the wordlist is
        resident, using that same small size just means many more
        kernel-launch/.wait() round trips than necessary -- the
        dominant cost once disk I/O and re-parsing are already
        eliminated (each launch+wait is a host<->device sync point,
        and a single-rule revalidation does exactly one dispatch per
        chunk). So for resident chunks we instead pick the LARGEST
        chunk size that still respects (a) MAX_DISPATCH_ITEMS -- the
        total (words x rules) work-items allowed in one kernel launch
        -- for a single-rule dispatch, and (b) the device's reported
        max single allocation size, so one chunk's buffer is always a
        legal OpenCL allocation. This collapses what used to be
        dozens-to-hundreds of small chunks into a handful of large
        ones, cutting launch/sync overhead by the same factor."""
        try:
            max_alloc_bytes = self.device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
        except Exception:
            max_alloc_bytes = 0
        words_per_alloc_limit = (max_alloc_bytes // MAX_WORD_LEN) if max_alloc_bytes else total_words
        # MAX_DISPATCH_ITEMS bounds num_words * num_rules_in_batch for
        # one launch; for the single-rule case (the hot path during
        # lazy-heap revalidation) that's just num_words itself.
        chunk_words = min(total_words, words_per_alloc_limit, MAX_DISPATCH_ITEMS)
        return max(1, int(chunk_words))

    def _preload_wordlist(self, wordlist_path):
        """Parse + encode the wordlist ONCE and keep it resident (GPU
        if it fits, host RAM otherwise) so score_batch() and
        apply_winner_and_clear() never touch disk or re-parse text
        again for the rest of the run. See class docstring."""
        t0 = time.time()

        # First pass: figure out how many words there are so we can
        # size resident chunks correctly (see _resident_chunk_words).
        # This still only reads/parses the file once for that count --
        # the iterator below (which builds the actual resident chunks)
        # is what matters for total preload cost, this pass is cheap
        # relative to it since it doesn't allocate any GPU buffers.
        total_words = 0
        for _words_np, num_words in optimized_wordlist_iterator(
                wordlist_path, MAX_WORD_LEN, self.words_per_gpu_batch):
            total_words += num_words

        chunk_words = self._resident_chunk_words(total_words)

        host_chunks = []
        for words_np, num_words in optimized_wordlist_iterator(
                wordlist_path, MAX_WORD_LEN, chunk_words):
            # optimized_wordlist_iterator() reuses/mutates its buffer
            # across yields except on the final partial chunk, so copy
            # defensively before holding onto it long-term.
            host_chunks.append((words_np.copy(), num_words))

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
    """Vectorized (NumPy, all-CPU) construction of the open-addressing
    hash table + occupied bitset uploaded to the GPU by _GpuScorer.

    An earlier version of this built the table with a plain Python
    `for h in ...: while occupied[...]: ...` loop -- correct, but pure
    per-element Python/NumPy-scalar overhead, which is fine for
    thousands of cracked entries but becomes the dominant one-time
    startup cost at real hashcat-scale cracked lists (multi-million
    entries; ~14M was observed taking tens of seconds in the loop
    form, all of it before the GPU does any useful work).

    This does the same linear-probing open-addressing insertion, but
    processes all keys in parallel "waves" instead of one Python
    `while` per key:
      1. Compute every key's initial probe slot in one vectorized op.
      2. Each wave: for every slot requested by more than one pending
         key this wave, arbitrarily pick ONE winner (np.unique's first
         occurrence) -- the others retry next wave. A winner is only
         actually placed if its slot isn't already occupied (from an
         earlier wave); if it is, it also retries next wave, slot+1.
      3. Repeat until no keys are pending.

    This still performs strictly sequential linear probing overall (no
    key is placed further from its ideal slot than the loop version
    would place SOME valid assignment), it just resolves an entire
    wave's worth of non-conflicting placements per NumPy call instead
    of one per Python-level loop iteration. Total wave count is
    bounded by the longest probe chain any key needs (typically small
    at the ~50% max load factor this table is sized for -- dozens of
    waves even at 14M keys, each wave O(pending) vectorized NumPy work),
    not by n_keys itself.

    Insertion ORDER differs from the sequential-loop version (ties for
    a slot within a wave are broken arbitrarily, not by original list
    order), but this doesn't affect correctness: open addressing with
    linear probing only requires that a key's insertion follow
    contiguous forward probing from its hash slot with no gaps left
    before its final resting slot, which every key here still does --
    lookup (unchanged, still a GPU-side linear probe from the same
    hash slot) finds exactly the same key regardless of where
    colliding keys ended up relative to each other.

    Returns (hash_table, occupied_words) in the exact same dtypes/
    layout __init__ previously built directly: hash_table is
    (hash_table_size,) uint32, occupied_words is
    (ceil(hash_table_size/32),) uint32 with bit (slot & 31) of word
    (slot >> 5) set for occupied slots -- i.e. bit-for-bit identical
    format/convention to the original loop's output, verified against
    it for correctness (see tests)."""
    keys = np.asarray(cracked_hashes_sorted, dtype=np.uint32)
    hash_table = np.zeros(hash_table_size, dtype=np.uint32)
    occ_bool = np.zeros(hash_table_size, dtype=bool)

    mask64 = np.uint64(hash_table_mask)
    pend_keys = keys
    pend_slot = ((keys.astype(np.uint64) * np.uint64(2654435761)) & mask64).astype(np.int64)

    while pend_keys.size:
        uniq_slots, first_pos = np.unique(pend_slot, return_index=True)
        winner_mask = np.zeros(pend_keys.size, dtype=bool)
        winner_mask[first_pos] = True

        cand_slots = pend_slot[winner_mask]
        cand_keys = pend_keys[winner_mask]
        free_mask = ~occ_bool[cand_slots]
        hash_table[cand_slots[free_mask]] = cand_keys[free_mask]
        occ_bool[cand_slots[free_mask]] = True

        placed_local = np.zeros(cand_slots.size, dtype=bool)
        placed_local[free_mask] = True
        retry_mask = ~winner_mask
        winner_pos = np.nonzero(winner_mask)[0]
        retry_mask[winner_pos[~placed_local]] = True

        pend_keys = pend_keys[retry_mask]
        pend_slot = (pend_slot[retry_mask] + 1) & hash_table_mask

    occ_bytes = np.packbits(occ_bool, bitorder='little')
    pad = (-len(occ_bytes)) % 4
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
    (rule_batch_size rows), exactly like compute_coverage_bitmaps()'s
    bitmap_g -- except this buffer holds int32 SCALARS per rule, not a
    full bitmap row, so it's ~W/32 times smaller for the same
    rule_batch_size."""

    def __init__(self, encoded_rules, num_cracked, cracked_hashes_sorted,
                 rule_batch_size, words_per_gpu_batch, device_id=None,
                 wordlist_path=None):
        self.n_rules = encoded_rules.shape[0]
        self.num_cracked = num_cracked
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

        mf = cl.mem_flags
        # Build a compact GPU open-addressing table.  `occupied` is a bitset
        # rather than a sentinel value so all 2^32 FNV hashes remain valid.
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

    def apply_winner_and_clear(self, rule_str, wordlist_path):
        """Runs the winning rule once against the wordlist, clearing
        every cracked hash it (still) covers out of `active`. Returns
        the true number of bits actually cleared this round. Same
        no-intermediate-.wait() reasoning as score_batch() above --
        only the final host_count readback blocks."""
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
                               device_id=None, budget=None):
    """Lazy-greedy max-coverage selection with the same selection
    semantics/order as celf_select() (marginal-gain-order, ties broken
    by discovery order), but with O(n_rules) + O(cracked_universe bits)
    peak memory instead of O(n_rules x cracked_universe bits) -- see
    module docstring. Returns list[(rule, gain)] best-first, same
    shape as celf_select()'s return value so callers (save_output(),
    save_output_multi()) don't need to change."""
    n_rules = len(rules)
    log(f"{blue('Recompute-GPU greedy select:')} {cyan(f'{n_rules:,}')} {bold('candidates,')} "
        f"{cyan(f'{len(cracked_hashes_sorted):,}')} {bold('cracked universe')} "
        f"-- {green('no coverage matrix allocated (RAM or disk)')}")

    encoded = np.zeros((n_rules, MAX_RULE_LEN), dtype=np.uint8)
    for i, r in enumerate(rules):
        rb = r.encode('latin-1', errors='ignore')[:MAX_RULE_LEN]
        encoded[i, :len(rb)] = np.frombuffer(rb, dtype=np.uint8)

    scorer = _GpuScorer(encoded, len(cracked_hashes_sorted), cracked_hashes_sorted,
                         rule_batch_size, words_per_gpu_batch, device_id,
                         wordlist_path=wordlist_path)

    # --- Pass 1: static upper bound for every candidate, against the
    # full (all-active) target set. Same GPU work as
    # compute_coverage_bitmaps()'s single streaming pass, but the only
    # thing retained afterwards is one int per rule.
    upper_bound = np.zeros(n_rules, dtype=np.int64)
    total_batches = math.ceil(n_rules / rule_batch_size)
    pbar = tqdm(total=total_batches, desc=cyan("Upper-bound pass (rule batches)"),
                unit="batch", colour="cyan")
    for start in range(0, n_rules, rule_batch_size):
        end = min(start + rule_batch_size, n_rules)
        idx_chunk = np.arange(start, end)
        upper_bound[start:end] = scorer.score_batch(idx_chunk, wordlist_path)
        pbar.set_postfix({"rss_mb": f"{get_rss_mb():.0f}"})
        pbar.update(1)
    pbar.close()

    # --- Greedy round loop: fixed-order array scan with two upper
    # bounds, NOT a heap. See the module docstring's "WHY NOT A HEAP"
    # note below for why this replaced an earlier heap-based version.
    #
    # `order` is EVERY candidate with upper_bound > 0, sorted descending
    # by upper_bound ONCE and never re-sorted (upper_bound is a static,
    # global ceiling -- valid for every round, by submodularity: a
    # rule's gain against any round's `active` set can never exceed its
    # gain against the full original target set). `last_bound[idx]`
    # additionally tracks each rule's most recently COMPUTED exact gain
    # (tighter than upper_bound after round 1, since it reflects a
    # smaller, more-covered `active` set than the original one) -- this
    # is the same "lazy" refinement celf_select()'s heap gets from
    # re-pushing a candidate with a fresh version stamp, here realized
    # as a plain array write instead of a heap push.
    #
    # Each round scans `order` (skipping already-picked rules) in
    # FIXED, RULE_BATCH_SIZE-sized chunks:
    #   - if even the chunk's first (best-upper-bound) rule can't beat
    #     the round's current best, BREAK the whole round's scan --
    #     sorted order means nothing later can beat it either.
    #   - within a surviving chunk, only candidates whose last_bound
    #     still exceeds the round's current best are actually sent to
    #     the GPU (one score_batch() call for the whole surviving
    #     sub-chunk); the rest are skipped without any dispatch.
    #
    # WHY NOT A HEAP: a heap only ever exposes ONE item at a time
    # (heappop), so batching "several items per GPU call" on top of it
    # means blindly grabbing a fixed number of pops (the previous
    # version of this function always grabbed up to `rule_batch_size`
    # pops the instant it saw even one stale entry) with no way to
    # check whether they were actually still needed once the round's
    # winner had already been established by the first one or two --
    # in practice this meant nearly every round rescored close to a
    # full rule_batch_size candidates even when 1-2 would have settled
    # it, which is what made greedy selection "terribly slow" despite
    # each individual GPU dispatch being reasonably large/well-utilized
    # (the problem was too many such dispatches, not too little work in
    # each one). The fixed sorted array lets `chunk[0]`'s bound decide,
    # BEFORE any GPU call, whether the rest of the round's candidates
    # are even worth dispatching at all -- once a round's best_gain is
    # high relative to what's left, most later chunks fail that check
    # and cost zero GPU dispatches, not a wasted rule_batch_size-sized
    # one.
    order = np.argsort(-upper_bound, kind='stable')
    order = order[upper_bound[order] > 0]
    n_active = len(order)
    log(f"{green('Candidates with >=1 hit:')} {cyan(f'{n_active:,}')}/{cyan(f'{n_rules:,}')}")

    last_bound = upper_bound.astype(np.int64).copy()
    excluded = np.zeros(n_rules, dtype=bool)

    limit = budget if budget else n_active

    # No heuristic warning here: the recompute path is intentionally a
    # selectable memory-light strategy. Runtime depends strongly on GPU,
    # wordlist residency, candidate distribution and budget; printing a
    # fixed pool/budget warning was noisy and did not predict actual runtime.

    selected = []
    round_num = 0
    dispatches_this_pick = 0
    round_t0 = time.time()

    pbar = tqdm(total=limit, desc=cyan("CELF greedy select [recompute-gpu]"), unit="rule", colour="cyan")
    while len(selected) < limit and n_active > 0:
        round_num += 1
        dispatches_this_pick = 0

        if round_num == 1:
            # Free: `active` is still every cracked hash, so
            # upper_bound[order[0]] (the largest full-target hit
            # count, already computed) IS this round's exact answer --
            # no GPU rescore needed to find it. Mirrors
            # RuleOptimizer-CUDA's generatePhase2(), which seeds
            # lastBestFitness from rules[0].Fitness the same way.
            best_idx = int(order[0])
            best_gain = int(upper_bound[best_idx])
        else:
            best_idx = -1
            best_gain = 0
            live = order[~excluded[order]]
            pos = 0
            while pos < len(live):
                chunk_end = min(pos + rule_batch_size, len(live))
                chunk = live[pos:chunk_end]
                if int(upper_bound[chunk[0]]) <= best_gain:
                    break  # sorted descending -- nothing later can beat best_gain either
                need_rescore = chunk[last_bound[chunk] > best_gain]
                if len(need_rescore):
                    gains = scorer.score_batch(need_rescore, wordlist_path)
                    dispatches_this_pick += 1
                    last_bound[need_rescore] = gains
                    for local_i, ridx in enumerate(need_rescore):
                        g = int(gains[local_i])
                        if g > best_gain or (g == best_gain and best_idx >= 0 and int(ridx) < best_idx):
                            best_gain = g
                            best_idx = int(ridx)
                pos = chunk_end

        if best_idx < 0 or best_gain <= 0:
            break

        true_cleared = scorer.apply_winner_and_clear(rules[best_idx], wordlist_path)
        selected.append((rules[best_idx], true_cleared))
        excluded[best_idx] = True
        n_active -= 1
        pbar.update(1)
        round_elapsed = time.time() - round_t0
        round_t0 = time.time()
        pbar.set_postfix({
            "round": round_num,
            "gpu_dispatches": dispatches_this_pick,
            "s/round": f"{round_elapsed:.1f}",
            "rss_mb": f"{get_rss_mb():.0f}",
        })
    pbar.close()

    remaining = scorer.remaining_active_count()
    total_covered = len(cracked_hashes_sorted) - remaining
    log(f"{green('Done.')} {bold('Selected')} {cyan(f'{len(selected):,}')} {bold('rules,')} "
        f"{bold('covering')} {cyan(f'{total_covered:,}')}/{cyan(f'{len(cracked_hashes_sorted):,}')} "
        f"{bold('cracked entries')} -- {dim('peak extra memory: O(n_rules) + O(universe bits), no matrix')}")
    return selected


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Memory-light GPU recompute+lazy-greedy max-coverage "
                     "selection, drop-in alternative to celf_postprocess.py's "
                     "matrix-based CELF.")
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
