#!/usr/bin/env python3
"""
celf_postprocess.py -- GPU/CPU CELF greedy coverage post-stage for Ranker v5.2
================================================================================
Run this AFTER rank_rules_exhaustive() or rank_rules_mab() (ranker_v5.2.py).

This revision streams the coverage-bitmap matrix to disk via np.memmap
instead of building it in RAM, and computes CELF's seed popcounts
incrementally during that same streaming pass. See the notes near
compute_coverage_bitmaps() and celf_select() for details.

Pass --in-ram to skip the memmap entirely and allocate the full coverage
matrix as a plain in-RAM ndarray instead (faster, since CELF's lazy
re-validation reads become plain memory reads instead of potential random
disk seeks -- but needs the full "estimated" size printed at startup as
free RAM).

Usage
-----
    python3 celf_postprocess.py \\
        --ranking-csv ranker_output.csv \\
        --wordlist rockyou.txt \\
        --cracked cracked.txt \\
        --candidates 20000 \\
        --budget 5000 \\
        --output celf_selected.rule

Or feed it a plain rules file instead of a ranking CSV:
    python3 celf_postprocess.py --rules-file top_optimized.rule ...

Streaming / memory notes (read this if you're here because of a
MemoryError or swap thrashing)
--------------------------------
The coverage matrix is (n_candidates x ceil(cracked_universe/32)) uint32
words. For 20,000 candidates x 14.3M unique cracked hashes that's ~34 GB --
too big for RAM on most machines, and it used to be allocated as one
np.zeros(...) up front. It no longer is by default:

  1. The matrix now lives in a disk-backed np.memmap
     (--bitmap-path, defaults next to --output), UNLESS --in-ram is
     passed, in which case it's a plain np.zeros(...) ndarray like
     before. The GPU pass still produces results one rule-batch at a
     time (as before), but each batch's host_bitmap is written straight
     into its slice of the matrix (memmap or ndarray) and then dropped
     -- only one batch's worth (rule_batch_size x bitmap_row_bytes, a
     few MB by default) is ever live as a separate/staging array in RAM.
  2. The CELF seed popcounts (the initial per-rule gains used to build
     the lazy-greedy heap) are computed incrementally *during that same
     write pass*, per batch, instead of in a second pass over the full
     matrix. A second full-matrix pass -- even with the SWAR popcount
     fix (no 16x blow-up from the old byte-LUT version) -- would still
     mean holding an (n_rules, W) array (or a same-shape temporary) in
     RAM at once, which is exactly what we're avoiding in memmap mode.
     Folding it into the write loop means each chunk is touched once,
     popcounted, and released.
  3. The CELF greedy loop itself already only ever touched one row
     (bitmaps[idx]) or the small OR-accumulator (covered, W words) at a
     time; that part is unchanged except that it now accepts the
     precomputed initial gains instead of calling popcount_rows() over
     the whole matrix itself. A memmap row-access pages in just that
     row's bytes from disk, same as it would from a real ndarray -- in
     --in-ram mode it's just a normal in-memory slice, no disk I/O ever.

The memmap file defaults to <output_base>.bitmap.dat and is deleted at
the end of a successful run unless --keep-bitmap is passed. Put it on
fast local storage with room for roughly the "estimated" size printed
at startup (same figure that used to be an "estimated RAM" warning --
now it's disk space instead, unless --in-ram is set, in which case it's
back to being an estimated-RAM warning and --bitmap-path/--keep-bitmap
are ignored since no file is ever created).
"""

import argparse
import csv
import math
import mmap
import multiprocessing as mp
import os
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyopencl as cl
from tqdm import tqdm

# ============================================================
# --- CONSTANTS (kept identical to ranker_v5.2 for correctness) ---
# ============================================================
# Defaults -- overridable via --max-word-len/--max-rule-len/--max-output-len.
# The old 256/512/255 values were far larger than any real wordlist/rule
# needs and cost a lot of private-memory-per-thread on the GPU (register
# spilling -> private arrays land in global/VRAM instead of registers).
# Smaller, realistic values cut that private footprint roughly 8x.
MAX_WORD_LEN = 32
MAX_OUTPUT_LEN = 64
MAX_RULE_LEN = 32
LOCAL_WORK_SIZE = 256
DEFAULT_WORDS_PER_GPU_BATCH = 150000
MAX_DISPATCH_ITEMS = 32 * 1024 * 1024


# ----------------------------------------------------------------------
# Colors & helpers (identical palette/helpers to ranker_v5.2's `C`/
# red/green/yellow/blue/cyan/bold/dim, so output looks consistent
# across both scripts in the pipeline)
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


def log(msg):
    """Prefixes every line with a dim, bracketed [CELF] tag -- callers
    pass an already-colored message body."""
    print(f"{dim('[CELF]')} {msg}", flush=True)


def get_rss_mb():
    """Current process resident set size, in MB. ru_maxrss is KB on
    Linux but bytes on macOS -- normalize for that."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return rss / (1024 * 1024)
    return rss / 1024


# ============================================================
# --- Loading helpers (trimmed copies of ranker_v5.2's) ---
# ============================================================
def estimate_word_count(path):
    file_size = os.path.getsize(path)
    sample_size = min(10 * 1024 * 1024, file_size)
    with open(path, 'rb') as f:
        sample = f.read(sample_size)
        lines = sample.count(b'\n')
        if file_size <= sample_size:
            return max(lines, 1)
        avg_line_length = sample_size / max(lines, 1)
        return max(int(file_size / avg_line_length), 1)


def fast_fnv1a_hash_32(data):
    hash_val = 2166136261
    for byte in data:
        hash_val = (hash_val ^ byte) * 16777619 & 0xFFFFFFFF
    return hash_val


def optimized_wordlist_iterator(wordlist_path, max_len, batch_size):
    """Memory-mapped iterator: yields (words_buffer, count) batches."""
    file_size = os.path.getsize(wordlist_path)
    batch_elements = batch_size * max_len
    words_buffer = np.zeros(batch_elements, dtype=np.uint8)
    with open(wordlist_path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            pos = 0
            batch_count = 0
            fsize = len(mm)
            while pos < fsize:
                end_pos = mm.find(b'\n', pos)
                if end_pos == -1:
                    end_pos = fsize
                line = mm[pos:end_pos].strip()
                line_len = len(line)
                pos = end_pos + 1
                if line_len == 0 or line_len > max_len:
                    continue
                start_idx = batch_count * max_len
                words_buffer[start_idx:start_idx + line_len] = np.frombuffer(
                    line, dtype=np.uint8, count=line_len)
                batch_count += 1
                if batch_count >= batch_size:
                    yield words_buffer.copy(), batch_count
                    batch_count = 0
                    words_buffer.fill(0)
            if batch_count > 0:
                yield words_buffer, batch_count


def load_cracked_universe(path, max_len):
    """Load cracked passwords -> sorted unique FNV-1a hash array.
    This sorted array IS the coverage universe: bit i in every rule's
    bitmap corresponds to cracked_hashes_sorted[i].

    Returns (arr, n_skipped) -- n_skipped counts non-empty lines longer
    than max_len that were dropped entirely (not truncated), so callers
    can warn if that's shrinking the coverage universe unexpectedly."""
    log(f"{blue('Loading cracked list:')} {path}")
    hashes = []
    n_skipped = 0
    with open(path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            pos = 0
            fsize = len(mm)
            while pos < fsize:
                end_pos = mm.find(b'\n', pos)
                if end_pos == -1:
                    end_pos = fsize
                line = mm[pos:end_pos].strip()
                pos = end_pos + 1
                if len(line) == 0:
                    continue
                if len(line) <= max_len:
                    hashes.append(fast_fnv1a_hash_32(line))
                else:
                    n_skipped += 1
    arr = np.unique(np.array(hashes, dtype=np.uint32))
    log(f"{green('Cracked universe size (unique hashes):')} {cyan(f'{len(arr):,}')}")
    return arr, n_skipped


def _char_to_pos(c):
    """Mirrors the GPU kernel's char_to_pos(): '0'-'9' -> 0-9,
    'A'-'Z'/'a'-'z' -> 10-35, else invalid (-1)."""
    if '0' <= c <= '9':
        return ord(c) - ord('0')
    if 'A' <= c <= 'Z':
        return ord(c) - ord('A') + 10
    if 'a' <= c <= 'z':
        return ord(c) - ord('a') + 10
    return -1


def _tokenize_rule(rule_str):
    """Mirrors the GPU kernel's cmd_len classification in
    apply_hashcat_rule(). Yields (cmd_char, cmd_len, arg1_or_None,
    arg2_or_None) tuples; stops early on a truncated trailing command,
    same as the kernel (`if (pos + cmd_len > rule_len) break;`)."""
    pos = 0
    n = len(rule_str)
    two_char = set("TDLR+-.,'^$@!/()yYzZp{}[]_e")
    while pos < n:
        c = rule_str[pos]
        if c in ('s', 'x', 'O', 'i', 'o', '*', '3', '%', '='):
            cmd_len = 3
        elif pos + 1 < n and c in two_char:
            cmd_len = 2
        else:
            cmd_len = 1
        if pos + cmd_len > n:
            break
        args = rule_str[pos + 1:pos + cmd_len]
        yield c, cmd_len, args
        pos += cmd_len


def estimate_output_len(rule_str, input_len):
    """Pure-Python mirror of the GPU kernel's length transformations
    (not byte content) for one rule applied to a word of length
    `input_len`. Data-dependent rejects (!, /, (, ), _, %, =, and s/x/O
    with a specific char) are assumed to NOT trigger -- i.e. this is a
    conservative upper bound on output length, not an exact simulation,
    since we don't know the actual wordlist content here. Used only to
    size --max-output-len, never for correctness of the GPU pass
    itself."""
    L = input_len
    for c, cmd_len, args in _tokenize_rule(rule_str):
        if cmd_len == 1:
            if c in ('d', 'f', 'q'):
                L = L * 2
            elif c in ('[', ']'):
                if L > 1:
                    L = L - 1
            # l,u,c,C,t,r,k,K,:,E,{,},default -> unchanged
        elif cmd_len == 2:
            arg = args[0]
            n = _char_to_pos(arg)
            if c == 'D':
                if 0 <= n < L:
                    L = L - 1
            elif c == 'L':
                if 0 <= n < L:
                    L = L - n
            elif c == 'R':
                if 0 <= n < L:
                    L = n + 1
            elif c == "'":
                if 0 <= n < L:
                    L = n
            elif c in ('^', '$'):
                L = L + 1
            elif c == 'y':
                if n >= 0:
                    L = L + min(n, L)
            elif c == 'Y':
                if n >= 0:
                    L = L + min(n, L)
            elif c == 'z':
                if n > 0:
                    L = L + n
            elif c == 'Z':
                if n > 0:
                    L = L + n
            elif c == 'p':
                if n >= 0:
                    L = L * (n + 1)
            elif c == '[':
                if 0 <= n < L:
                    L = L - n
            elif c == ']':
                if 0 <= n < L:
                    L = L - n
            # T,+,-,.,,,@,!,/,(,),{,},_,e,default -> unchanged (or
            # reject, assumed not to trigger -- see docstring)
        elif cmd_len == 3:
            a1, a2 = args[0], args[1]
            n1, n2 = _char_to_pos(a1), _char_to_pos(a2)
            if c == 'x':
                if n1 >= 0 and n2 > 0 and n1 < L:
                    end = min(n1 + n2, L)
                    L = end - n1
            elif c == 'O':
                if n1 >= 0 and n2 > 0 and n1 < L:
                    end = min(n1 + n2, L)
                    L = L - (end - n1)
            elif c == 'i':
                if n1 >= 0:
                    L = L + 1
            # s,o,*,3,%,=,default -> unchanged
    return max(L, 0)


def estimate_worst_case_output_len(rules, max_word_len):
    """Runs estimate_output_len() for every candidate rule against a
    word of length max_word_len, and returns (worst_len, worst_rule).
    Conservative upper bound -- see estimate_output_len()'s docstring."""
    worst_len = max_word_len
    worst_rule = None
    for r in rules:
        L = estimate_output_len(r, max_word_len)
        if L > worst_len:
            worst_len = L
            worst_rule = r
    return worst_len, worst_rule


def load_candidate_rules(args):
    """Returns list[str] of candidate rules, best-score first, capped
    at --candidates."""
    rules = []
    if args.rules_file:
        with open(args.rules_file, 'r', encoding='latin-1') as f:
            for line in f:
                r = line.strip()
                if r and r != ':' and not r.startswith('#'):
                    rules.append(r)
    elif args.ranking_csv:
        with open(args.ranking_csv, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        def score_of(row):
            try:
                return float(row.get('Combined_Score', 0))
            except (TypeError, ValueError):
                return 0.0
        rows.sort(key=score_of, reverse=True)
        rules = [row['Rule_Data'] for row in rows if row.get('Rule_Data')]
    else:
        raise ValueError("Provide --ranking-csv or --rules-file")

    if args.candidates and len(rules) > args.candidates:
        rules = rules[:args.candidates]
    log(f"{green('Candidate pool:')} {cyan(f'{len(rules):,}')} {bold('rules')}")
    return rules


# ============================================================
# --- Popcount helpers (SWAR bit-trick, in place on uint32 arrays) ---
#
# Fixed version (no 256-entry byte LUT / no 16x memory blow-up), now
# called on one rule-batch's rows at a time instead of the whole matrix.
# ============================================================
def _popcount32(x):
    """x: np.ndarray[uint32], any shape. Returns same-shape uint32
    array of per-element popcounts."""
    x = x.astype(np.uint32, copy=True)
    x -= (x >> np.uint32(1)) & np.uint32(0x55555555)
    x = (x & np.uint32(0x33333333)) + ((x >> np.uint32(2)) & np.uint32(0x33333333))
    x = (x + (x >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
    x = (x * np.uint32(0x01010101)) >> np.uint32(24)
    return x


def popcount_rows(bitmap_2d):
    """bitmap_2d: (n, W) uint32 -> (n,) int64 popcount per row."""
    return _popcount32(bitmap_2d).sum(axis=1, dtype=np.int64)


def popcount_row(bitmap_row):
    return int(_popcount32(bitmap_row).sum(dtype=np.int64))


# ============================================================
# --- Hybrid dense/sparse row storage ---------------------------------
#
# Real ranking runs (see the Marginal_Gain column from `rank`/`handler`
# output) follow a steep power-law: a small head of rules crack a huge
# fraction of the cracked universe, and the long tail of low-ranked
# candidates -- typically most of a 20k-150k candidate pool -- each
# crack only a tiny sliver of it. Storing every row as a full dense
# bitmap means the tail wastes almost all of its bytes on zero words.
#
# Each row is stored in whichever representation is smaller, decided
# purely from its own popcount (already computed for initial_gains, so
# this costs nothing extra):
#   - dense  (kind 0): the row's W=bitmap_words_per_rule uint32 words,
#     verbatim, exactly as before.
#   - sparse (kind 1): the sorted uint32 bit-indices of the set bits,
#     i.e. popcount x 4 bytes.
# Sparse wins exactly when popcount < W (byte-for-byte break-even:
# popcount*4 < W*4), so no tuning knob is needed -- worst case for any
# row is min(dense, sparse), i.e. this can never be bigger than the
# old format and is usually far smaller whenever coverage is uneven.
#
# Rows are variable-length, so instead of one fixed-stride memmap the
# on-disk layout is:
#   <bitmap_path>            -- row payloads, concatenated, no padding
#   <bitmap_path>.index.npz  -- 'offsets' (n_rules+1 int64 byte offsets
#                                into the data file), 'kinds' (n_rules
#                                uint8), 'W' (bitmap_words_per_rule)
# The index is O(n_rules), ~9 bytes/rule (150k rules ~= 1.3 MB) -- same
# size class as initial_gains, always fine to hold in RAM / duplicate
# into worker processes, unlike the matrix itself.
# ============================================================
def _bits_to_indices(row_u32):
    """(W,) uint32 dense row -> sorted uint32 array of set-bit indices."""
    bits = np.unpackbits(row_u32.view(np.uint8), bitorder='little')
    return np.nonzero(bits)[0].astype(np.uint32)


def _indices_to_dense(idx_arr, bitmap_words_per_rule):
    """Inverse of _bits_to_indices."""
    bits = np.zeros(bitmap_words_per_rule * 32, dtype=np.uint8)
    if len(idx_arr):
        bits[idx_arr] = 1
    return np.packbits(bits, bitorder='little').view(np.uint32).copy()


def _pack_row(dense_row, popcount, bitmap_words_per_rule):
    """Picks whichever of dense/sparse is smaller for THIS row, from
    its popcount alone. Returns (kind, payload_bytes)."""
    if popcount < bitmap_words_per_rule:
        return 1, _bits_to_indices(dense_row).tobytes()
    return 0, dense_row.tobytes()


def _unpack_row(kind, raw_bytes, bitmap_words_per_rule):
    if kind == 0:
        return np.frombuffer(raw_bytes, dtype=np.uint32).copy()
    idx = np.frombuffer(raw_bytes, dtype=np.uint32)
    return _indices_to_dense(idx, bitmap_words_per_rule)


class HybridRowWriter:
    """Write side: append one rule-batch's dense rows at a time,
    packing each row to its smaller representation, tracking a running
    byte offset. Never holds more than one batch of dense rows (the
    caller's host_bitmap) plus the small O(n_rules) offsets/kinds
    arrays in RAM."""

    def __init__(self, data_path, n_rules, bitmap_words_per_rule):
        self.data_path = data_path
        self.W = bitmap_words_per_rule
        self.offsets = np.zeros(n_rules + 1, dtype=np.int64)
        self.kinds = np.zeros(n_rules, dtype=np.uint8)
        self._f = open(data_path, 'wb')
        self._pos = 0

    def write_batch(self, start, host_bitmap, popcounts):
        """host_bitmap: (n, W) uint32 dense rows for rules[start:start+n].
        popcounts: (n,) int-like, already computed for initial_gains."""
        for i in range(host_bitmap.shape[0]):
            kind, payload = _pack_row(host_bitmap[i], int(popcounts[i]), self.W)
            self._f.write(payload)
            self._pos += len(payload)
            self.kinds[start + i] = kind
            self.offsets[start + i + 1] = self._pos

    def close(self):
        self._f.close()

    def bytes_written(self):
        return self._pos

    def to_store(self):
        store = HybridRowStore(self.data_path, self.offsets, self.kinds, self.W)
        store.save_index()
        return store


class HybridRowStore:
    """Read side: mmap'd view over the hybrid dense/sparse coverage
    file. store[idx] -> dense (W,) uint32 ndarray (materializing sparse
    rows on the fly), and .shape -- a drop-in substitute for the old
    fixed-stride memmap at every bitmaps[idx] call site."""

    def __init__(self, data_path, offsets, kinds, bitmap_words_per_rule):
        self.data_path = data_path
        self.offsets = offsets
        self.kinds = kinds
        self.W = bitmap_words_per_rule
        self.shape = (len(kinds), bitmap_words_per_rule)
        self._f = open(data_path, 'rb')
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def __getitem__(self, idx):
        start, end = int(self.offsets[idx]), int(self.offsets[idx + 1])
        raw = self._mm[start:end]
        return _unpack_row(int(self.kinds[idx]), raw, self.W)

    def flush(self):
        pass  # read-only, nothing to flush

    def close(self):
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            self._f.close()
        except Exception:
            pass

    @staticmethod
    def index_path(data_path):
        return data_path + '.index.npz'

    def save_index(self):
        np.savez(self.index_path(self.data_path),
                  offsets=self.offsets, kinds=self.kinds,
                  W=np.array([self.W], dtype=np.int64))

    @classmethod
    def load(cls, data_path):
        npz = np.load(cls.index_path(data_path))
        return cls(data_path, npz['offsets'], npz['kinds'], int(npz['W'][0]))

    def __del__(self):
        self.close()


# ============================================================
# --- OpenCL kernel: rule application (verbatim logic from ranker_v5.2)
#     + a coverage-bitmap kernel instead of the counter kernel.
# ============================================================
def get_celf_kernel_source(num_cracked, bitmap_words_per_rule):
    return f"""
#define MAX_WORD_LEN {MAX_WORD_LEN}
#define MAX_OUTPUT_LEN {MAX_OUTPUT_LEN}
#define MAX_RULE_LEN {MAX_RULE_LEN}
#define NUM_CRACKED {num_cracked}
#define BITMAP_WORDS_PER_RULE {bitmap_words_per_rule}

int is_lower(unsigned char c) {{ return (c >= 'a' && c <= 'z'); }}
int is_upper(unsigned char c) {{ return (c >= 'A' && c <= 'Z'); }}
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
unsigned int fnv1a_hash_32(const unsigned char* data, unsigned int len) {{
    unsigned int hash = 2166136261U;
    for (unsigned int i = 0; i < len; i++) {{ hash ^= data[i]; hash *= 16777619U; }}
    return hash;
}}

static void duplicate_front(const unsigned char* in, int in_len,
                            unsigned char* out, int* out_len, int* changed, int n) {{
    if (n > in_len) n = in_len;
    int new_len = in_len + n;
    if (new_len > MAX_OUTPUT_LEN) return;
    for (int i = 0; i < n; i++) out[i] = in[i];
    for (int i = 0; i < in_len; i++) out[n + i] = in[i];
    *out_len = new_len; *changed = 1;
}}
static void duplicate_back(const unsigned char* in, int in_len,
                           unsigned char* out, int* out_len, int* changed, int n) {{
    if (n > in_len) n = in_len;
    int new_len = in_len + n;
    if (new_len > MAX_OUTPUT_LEN) return;
    for (int i = 0; i < in_len; i++) out[i] = in[i];
    for (int i = 0; i < n; i++) out[in_len + i] = in[in_len - n + i];
    *out_len = new_len; *changed = 1;
}}
static void duplicate_word(const unsigned char* in, int in_len,
                           unsigned char* out, int* out_len, int* changed, int times) {{
    int new_len = in_len * (times + 1);
    if (new_len > MAX_OUTPUT_LEN) return;
    for (int rep = 0; rep <= times; rep++)
        for (int i = 0; i < in_len; i++) out[rep * in_len + i] = in[i];
    *out_len = new_len; *changed = 1;
}}
static void rotate_left(const unsigned char* in, int in_len,
                        unsigned char* out, int* out_len, int* changed, int n) {{
    if (n <= 0) n = 1;
    n %= in_len;
    *out_len = in_len;
    if (n == 0) {{ for (int i = 0; i < in_len; i++) out[i] = in[i]; *changed = 0; return; }}
    for (int i = 0; i < in_len; i++) out[i] = in[(i + n) % in_len];
    *changed = 1;
}}
static void rotate_right(const unsigned char* in, int in_len,
                         unsigned char* out, int* out_len, int* changed, int n) {{
    if (n <= 0) n = 1;
    n %= in_len;
    *out_len = in_len;
    if (n == 0) {{ for (int i = 0; i < in_len; i++) out[i] = in[i]; *changed = 0; return; }}
    for (int i = 0; i < in_len; i++) out[i] = in[(i - n + in_len) % in_len];
    *changed = 1;
}}

static int apply_single_command(const unsigned char* in, int in_len,
                                unsigned char* out, int* out_len,
                                const unsigned char* cmd, int cmd_len) {{
    int changed = 0;
    *out_len = 0;
    if (cmd_len == 1) {{
        switch (cmd[0]) {{
            case 'l': *out_len = in_len; for (int i=0;i<in_len;i++) out[i]=to_lower(in[i]); changed=1; break;
            case 'u': *out_len = in_len; for (int i=0;i<in_len;i++) out[i]=to_upper(in[i]); changed=1; break;
            case 'c': *out_len = in_len; if (in_len>0) out[0]=to_upper(in[0]); for (int i=1;i<in_len;i++) out[i]=to_lower(in[i]); changed=1; break;
            case 'C': *out_len = in_len; if (in_len>0) out[0]=to_lower(in[0]); for (int i=1;i<in_len;i++) out[i]=to_upper(in[i]); changed=1; break;
            case 't': *out_len = in_len; for (int i=0;i<in_len;i++) out[i]=toggle_case(in[i]); changed=1; break;
            case 'r': *out_len = in_len; for (int i=0;i<in_len;i++) out[i]=in[in_len-1-i]; changed=1; break;
            case 'd': if (in_len*2<=MAX_OUTPUT_LEN) {{ *out_len=in_len*2; for(int i=0;i<in_len;i++){{out[i]=in[i];out[in_len+i]=in[i];}} changed=1; }} break;
            case 'f': if (in_len*2<=MAX_OUTPUT_LEN) {{ *out_len=in_len*2; for(int i=0;i<in_len;i++){{out[i]=in[i];out[in_len+i]=in[in_len-1-i];}} changed=1; }} break;
            case 'k': *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; if(in_len>=2){{out[0]=in[1];out[1]=in[0];changed=1;}} break;
            case 'K': *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; if(in_len>=2){{out[in_len-2]=in[in_len-1];out[in_len-1]=in[in_len-2];changed=1;}} break;
            case ':': *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; changed=0; break;
            case 'q': if (in_len*2<=MAX_OUTPUT_LEN) {{ int idx=0; for(int i=0;i<in_len;i++){{out[idx++]=in[i];out[idx++]=in[i];}} *out_len=in_len*2; changed=1; }} break;
            case 'E': {{ *out_len=in_len; int cap=1; for(int i=0;i<in_len;i++){{ if(cap&&is_lower(in[i])) out[i]=to_upper(in[i]); else out[i]=to_lower(in[i]); cap=(in[i]==' '||in[i]=='-'||in[i]=='_'); }} changed=1; }} break;
            case '{{': rotate_left(in,in_len,out,out_len,&changed,1); break;
            case '}}': rotate_right(in,in_len,out,out_len,&changed,1); break;
            case '[': if (in_len>1) {{ *out_len=in_len-1; for(int i=1;i<in_len;i++) out[i-1]=in[i]; changed=1; }} break;
            case ']': if (in_len>1) {{ *out_len=in_len-1; for(int i=0;i<in_len-1;i++) out[i]=in[i]; changed=1; }} break;
            default: *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; changed=0; break;
        }}
        return changed;
    }}
    if (cmd_len == 2) {{
        unsigned char cmd_char = cmd[0];
        unsigned char arg = cmd[1];
        int n = (int)char_to_pos(arg);
        if (n == 0xFFFFFFFF) n = -1;
        switch (cmd_char) {{
            case 'T': if (n>=0&&n<in_len){{ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; out[n]=toggle_case(in[n]); changed=1; }} break;
            case 'D': if (n>=0&&n<in_len){{ *out_len=in_len-1; for(int i=0;i<n;i++) out[i]=in[i]; for(int i=n+1;i<in_len;i++) out[i-1]=in[i]; changed=1; }} break;
            case 'L': if (n>=0&&n<in_len){{ *out_len=in_len-n; for(int i=n;i<in_len;i++) out[i-n]=in[i]; changed=1; }} break;
            case 'R': if (n>=0&&n<in_len){{ *out_len=n+1; for(int i=0;i<=n;i++) out[i]=in[i]; changed=1; }} break;
            case '+': if (n>=0&&n<in_len){{ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; out[n]=in[n]+1; changed=1; }} break;
            case '-': if (n>=0&&n<in_len){{ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; out[n]=in[n]-1; changed=1; }} break;
            case '.': if (n>=0&&n<in_len){{ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; out[n]=in[n]+1; changed=1; }} break;
            case ',': if (n>=0&&n<in_len){{ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; out[n]=in[n]-1; changed=1; }} break;
            case '\\'': if (n>=0&&n<in_len){{ *out_len=n; for(int i=0;i<n;i++) out[i]=in[i]; changed=1; }} break;
            case '^': if (in_len+1<=MAX_OUTPUT_LEN){{ out[0]=arg; for(int i=0;i<in_len;i++) out[i+1]=in[i]; *out_len=in_len+1; changed=1; }} break;
            case '$': if (in_len+1<=MAX_OUTPUT_LEN){{ for(int i=0;i<in_len;i++) out[i]=in[i]; out[in_len]=arg; *out_len=in_len+1; changed=1; }} break;
            case '@': *out_len=0; for(int i=0;i<in_len;i++){{ if(in[i]!=arg) out[(*out_len)++]=in[i]; else changed=1; }} break;
            case '!': for(int i=0;i<in_len;i++) if(in[i]==arg) return -1; *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; return 0;
            case '/': for(int i=0;i<in_len;i++) if(in[i]==arg){{ *out_len=in_len; for(int j=0;j<in_len;j++) out[j]=in[j]; return 0; }} return -1;
            case '(': if (in_len>0&&in[0]==arg){{ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; return 0; }} return -1;
            case ')': if (in_len>0&&in[in_len-1]==arg){{ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; return 0; }} return -1;
            case 'y': if (n>=0) duplicate_front(in,in_len,out,out_len,&changed,n); break;
            case 'Y': if (n>=0) duplicate_back(in,in_len,out,out_len,&changed,n); break;
            case 'z': if (n>0 && in_len+n<=MAX_OUTPUT_LEN){{ out[0]=in[0]; for(int i=0;i<n;i++) out[i+1]=in[0]; for(int i=1;i<in_len;i++) out[n+i]=in[i]; *out_len=in_len+n; changed=1; }} break;
            case 'Z': if (n>0 && in_len+n<=MAX_OUTPUT_LEN){{ for(int i=0;i<in_len;i++) out[i]=in[i]; for(int i=0;i<n;i++) out[in_len+i]=in[in_len-1]; *out_len=in_len+n; changed=1; }} break;
            case 'p': if (n>=0) duplicate_word(in,in_len,out,out_len,&changed,n); break;
            case '{{': if (n>=0) rotate_left(in,in_len,out,out_len,&changed,n); break;
            case '}}': if (n>=0) rotate_right(in,in_len,out,out_len,&changed,n); break;
            case '[': if (n>=0&&n<in_len){{ *out_len=in_len-n; for(int i=n;i<in_len;i++) out[i-n]=in[i]; changed=1; }} break;
            case ']': if (n>=0&&n<in_len){{ *out_len=in_len-n; for(int i=0;i<*out_len;i++) out[i]=in[i]; changed=1; }} break;
            case '_': if (n>=0 && in_len!=n) return -1; *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; return 0;
            case 'e': {{ *out_len=in_len; int cap_sep=1; for(int i=0;i<in_len;i++){{ if(cap_sep&&is_lower(in[i])) out[i]=to_upper(in[i]); else out[i]=to_lower(in[i]); cap_sep=(in[i]==arg); }} changed=1; }} break;
            default: *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; changed=0; break;
        }}
        return changed;
    }}
    if (cmd_len == 3) {{
        unsigned char cmd_char = cmd[0];
        unsigned char a1 = cmd[1];
        unsigned char a2 = cmd[2];
        int n1 = (int)char_to_pos(a1);
        int n2 = (int)char_to_pos(a2);
        switch (cmd_char) {{
            case 's': *out_len=in_len; for(int i=0;i<in_len;i++){{ out[i]=(in[i]==a1)?a2:in[i]; if(in[i]==a1) changed=1; }} break;
            case 'x': if (n1>=0&&n2>0&&n1<in_len){{ int end=n1+n2; if(end>in_len) end=in_len; *out_len=end-n1; for(int i=n1;i<end;i++) out[i-n1]=in[i]; changed=1; }} break;
            case 'O': if (n1>=0&&n2>0&&n1<in_len){{ int end=n1+n2; if(end>in_len) end=in_len; *out_len=in_len-(end-n1); for(int i=0;i<n1;i++) out[i]=in[i]; for(int i=end;i<in_len;i++) out[i-n2]=in[i]; changed=1; }} break;
            case 'i': if (n1>=0 && in_len+1<=MAX_OUTPUT_LEN){{ if(n1>in_len) n1=in_len; *out_len=in_len+1; for(int i=0;i<n1;i++) out[i]=in[i]; out[n1]=a2; for(int i=n1;i<in_len;i++) out[i+1]=in[i]; changed=1; }} break;
            case 'o': if (n1>=0&&n1<in_len){{ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; out[n1]=a2; changed=1; }} break;
            case '*': if (n1>=0&&n2>=0&&n1<in_len&&n2<in_len&&n1!=n2){{ *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; unsigned char tmp=out[n1]; out[n1]=out[n2]; out[n2]=tmp; changed=1; }} break;
            case '3': if (n1>=0){{ int count=0; *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; for(int i=0;i<in_len;i++){{ if(in[i]==a2) count++; if(count==n1+1 && i+1<in_len){{ out[i+1]=toggle_case(in[i+1]); changed=1; break; }} }} }} break;
            case '%': if (n1>=0){{ int cnt=0; for(int i=0;i<in_len;i++) if(in[i]==a2) cnt++; if(cnt<n1) return -1; }} *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; return 0;
            case '=': if (n1>=0&&n1<in_len&&in[n1]!=a2) return -1; *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; return 0;
            default: *out_len=in_len; for(int i=0;i<in_len;i++) out[i]=in[i]; changed=0; break;
        }}
        return changed;
    }}
    *out_len = in_len; for (int i=0;i<in_len;i++) out[i]=in[i]; return 0;
}}

void apply_hashcat_rule(const unsigned char* word, int word_len,
                        const unsigned char* rule, int rule_len,
                        unsigned char* output, int* out_len, int* changed) {{
    unsigned char buf0[MAX_OUTPUT_LEN];
    unsigned char buf1[MAX_OUTPUT_LEN];
    unsigned char* in_buf = (unsigned char*)word;
    int in_len = word_len;
    int final_changed = 0;
    int pos = 0;
    while (pos < rule_len) {{
        unsigned char cmd_char = rule[pos];
        int cmd_len = 1;
        if (cmd_char=='s'||cmd_char=='x'||cmd_char=='O'||cmd_char=='i'||cmd_char=='o'||
            cmd_char=='*'||cmd_char=='3'||cmd_char=='%'||cmd_char=='=') {{
            cmd_len = 3;
        }} else if (pos+1 < rule_len && (cmd_char=='T'||cmd_char=='D'||cmd_char=='L'||cmd_char=='R'||
                    cmd_char=='+'||cmd_char=='-'||cmd_char=='.'||cmd_char==','||cmd_char=='\\''||
                    cmd_char=='^'||cmd_char=='$'||cmd_char=='@'||cmd_char=='!'||cmd_char=='/'||
                    cmd_char=='('||cmd_char==')'||cmd_char=='y'||cmd_char=='Y'||cmd_char=='z'||
                    cmd_char=='Z'||cmd_char=='p'||cmd_char=='{{'||cmd_char=='}}'||cmd_char=='['||
                    cmd_char==']'||cmd_char=='_'||cmd_char=='e')) {{
            cmd_len = 2;
        }}
        if (pos + cmd_len > rule_len) break;
        int out_len_local = 0;
        int result = apply_single_command(in_buf, in_len, buf0, &out_len_local, rule+pos, cmd_len);
        if (result == -1) {{ *out_len = 0; *changed = -1; return; }}
        if (result == 1) final_changed = 1;
        in_len = out_len_local;
        for (int i=0;i<in_len;i++) buf1[i]=buf0[i];
        in_buf = buf1;
        pos += cmd_len;
    }}
    *out_len = in_len;
    for (int i=0;i<in_len;i++) output[i]=in_buf[i];
    *changed = final_changed;
}}

// Binary search: returns index into cracked_hashes_sorted, or -1
int binary_search_cracked(__global const unsigned int* sorted_hashes, unsigned int n, unsigned int key) {{
    int lo = 0, hi = (int)n - 1;
    while (lo <= hi) {{
        int mid = (lo + hi) >> 1;
        unsigned int v = sorted_hashes[mid];
        if (v == key) return mid;
        if (v < key) lo = mid + 1; else hi = mid - 1;
    }}
    return -1;
}}

// ------------------------------------------------------------
// Coverage kernel: for each (word, candidate-rule) pair, apply the
// rule, hash the result, and if it matches a cracked-list entry,
// set that bit in this rule's row of the coverage bitmap.
// coverage_bitmap persists across word-batch dispatches for a given
// rule-batch (caller zeroes it once per rule-batch, not per word-batch).
// ------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void celf_coverage_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned char* rules_in,
    __global const unsigned int* cracked_hashes_sorted,
    __global unsigned int* coverage_bitmap,
    const unsigned int num_words,
    const unsigned int num_rules_in_batch,
    const unsigned int max_word_len)
{{
    unsigned int global_id = get_global_id(0);
    unsigned int total = num_words * num_rules_in_batch;
    if (global_id >= total) return;

    // rule_idx varies slowest, word_idx fastest: consecutive global_ids
    // (i.e. threads within the same warp/wavefront) share the SAME rule
    // and only differ in word_idx. apply_single_command() is one huge
    // switch() on the rule's command chars, so same rule = same branch
    // taken by the whole warp = no intra-warp divergence from rule
    // choice. The old word_idx/rule_idx = id/N, id%N split put a
    // DIFFERENT rule on every lane of a warp, which serialized the warp
    // through up to 32 different switch-paths at once.
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
    int idx = binary_search_cracked(cracked_hashes_sorted, NUM_CRACKED, h);
    if (idx < 0) return;

    unsigned int word_pos = (unsigned int)idx >> 5;
    unsigned int bit_pos = (unsigned int)idx & 31;
    __global unsigned int* row = &coverage_bitmap[rule_idx * BITMAP_WORDS_PER_RULE];
    atomic_or(&row[word_pos], (1U << bit_pos));
}}
"""


# ============================================================
# --- Device selection (minimal) ---
# ============================================================
def select_device(device_id=None):
    platforms = cl.get_platforms()
    if not platforms:
        log(red("No OpenCL platforms found!"))
        sys.exit(1)
    for p in platforms:
        try:
            devices = p.get_devices()
        except Exception:
            continue
        gpus = [d for d in devices if d.type == cl.device_type.GPU]
        if gpus:
            dev = gpus[device_id] if device_id is not None and device_id < len(gpus) else gpus[0]
            log(f"{green('Using GPU:')} {cyan(dev.name.strip())} {dim(f'(platform: {p.name.strip()})')}")
            return p, dev
    p = platforms[0]
    dev = p.get_devices()[0]
    log(f"{yellow('No GPU found, falling back to:')} {cyan(dev.name.strip())}")
    return p, dev


# ============================================================
# --- GPU coverage evaluation over candidate rule batches, streamed
#     to a disk-backed memmap (default) or a plain in-RAM ndarray
#     (--in-ram). ---
# ============================================================
def compute_coverage_bitmaps(rules, wordlist_path, cracked_hashes_sorted,
                             rule_batch_size, words_per_gpu_batch,
                             bitmap_path, device_id=None, in_ram=False,
                             hybrid=True):
    """Builds the (n_rules, bitmap_words_per_rule) uint32 coverage
    matrix either as a disk-backed np.memmap at `bitmap_path` (default)
    or, if in_ram=True, as a plain in-RAM np.zeros(...) ndarray.

    In memmap mode, only one rule-batch's worth of rows
    (rule_batch_size x bitmap_row_bytes) is ever live in a separate
    plain ndarray at a time; everything else lives on disk and is paged
    in on demand. In --in-ram mode the whole matrix is resident in RAM
    up front (same peak footprint as the old pre-streaming version),
    but there's no disk I/O at all, including during CELF's lazy
    re-validation row reads later.

    Returns (bitmaps, bitmap_words_per_rule, initial_gains) where
    bitmaps is either the memmap or the ndarray, and initial_gains is
    an (n_rules,) int64 array of per-rule popcounts, computed
    incrementally during this same streaming pass so CELF's heap-seeding
    step doesn't need a second full scan of the matrix.
    """
    n_rules = len(rules)
    num_cracked = len(cracked_hashes_sorted)
    bitmap_words_per_rule = max(1, (num_cracked + 31) // 32)

    bitmap_row_bytes = bitmap_words_per_rule * 4
    est_bytes = n_rules * bitmap_row_bytes
    est_gb = est_bytes / (1024 ** 3)

    if in_ram:
        log(f"{yellow('--in-ram set:')} {blue('Coverage bitmap matrix:')} "
            f"{cyan(f'{n_rules:,}')} {bold('candidates x')} "
            f"{cyan(f'{bitmap_row_bytes/1024:.1f} KB')}/rule = {cyan(f'{est_gb:.2f} GB')} {bold('total')} "
            f"-- {dim('allocated directly in RAM, no memmap/disk file')}.")
        bitmaps = np.zeros((n_rules, bitmap_words_per_rule), dtype=np.uint32)
    elif not hybrid:
        log(f"{blue('Coverage bitmap matrix:')} {cyan(f'{n_rules:,}')} {bold('candidates x')} "
            f"{cyan(f'{bitmap_row_bytes/1024:.1f} KB')}/rule = {cyan(f'{est_gb:.2f} GB')} {bold('total')} "
            f"-- {dim(f'streamed to disk at {bitmap_path}, not held in RAM')} "
            f"{yellow('(--no-hybrid: dense, fixed-stride)')}.")
        log(f"{blue('Peak extra RAM for this pass is ~one rule-batch:')} "
            f"{cyan(f'{rule_batch_size} x {bitmap_row_bytes/1024:.1f} KB')} "
            f"= {cyan(f'{rule_batch_size * bitmap_row_bytes / (1024**2):.1f} MB')}, "
            f"{dim('regardless of how large the full matrix above is')}.")
        # Disk-backed matrix. 'w+' creates/truncates; every rule-batch
        # write below fills in one row-slice of it.
        bitmaps = np.memmap(bitmap_path, dtype=np.uint32, mode='w+',
                             shape=(n_rules, bitmap_words_per_rule))
    else:
        log(f"{blue('Coverage bitmap matrix (worst case):')} {cyan(f'{n_rules:,}')} {bold('candidates x')} "
            f"{cyan(f'{bitmap_row_bytes/1024:.1f} KB')}/rule = {cyan(f'{est_gb:.2f} GB')} {bold('total')} "
            f"-- {dim(f'hybrid dense/sparse rows streamed to {bitmap_path}')} "
            f"{green('(actual size will be <= this, usually far less)')}.")
        log(f"{blue('Peak extra RAM for this pass is ~one rule-batch:')} "
            f"{cyan(f'{rule_batch_size} x {bitmap_row_bytes/1024:.1f} KB')} "
            f"= {cyan(f'{rule_batch_size * bitmap_row_bytes / (1024**2):.1f} MB')}, "
            f"{dim('regardless of how large the worst-case matrix above is')}.")
        writer = HybridRowWriter(bitmap_path, n_rules, bitmap_words_per_rule)
        bitmaps = None  # built at the end of the streaming loop below

    initial_gains = np.zeros(n_rules, dtype=np.int64)

    platform, device = select_device(device_id)
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    kernel_src = get_celf_kernel_source(num_cracked, bitmap_words_per_rule)
    prg = cl.Program(context, kernel_src).build()
    kernel = prg.celf_coverage_kernel

    mf = cl.mem_flags
    cracked_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cracked_hashes_sorted)

    words_buffer_size = words_per_gpu_batch * MAX_WORD_LEN * np.uint8().itemsize
    base_words_g = cl.Buffer(context, mf.READ_ONLY, words_buffer_size)

    rules_buffer_size = rule_batch_size * MAX_RULE_LEN * np.uint8().itemsize
    rules_g = cl.Buffer(context, mf.READ_ONLY, rules_buffer_size)

    bitmap_buffer_bytes = rule_batch_size * bitmap_row_bytes
    bitmap_g = cl.Buffer(context, mf.READ_WRITE, bitmap_buffer_bytes)

    # Pre-encode all candidate rules once (host side, cheap: n_rules x
    # MAX_RULE_LEN bytes, e.g. 20000 x 255 =~ 5 MB -- not what was
    # blowing up memory).
    encoded = np.zeros((n_rules, MAX_RULE_LEN), dtype=np.uint8)
    for i, r in enumerate(rules):
        rb = r.encode('latin-1', errors='ignore')[:MAX_RULE_LEN]
        encoded[i, :len(rb)] = np.frombuffer(rb, dtype=np.uint8)

    total_rule_batches = math.ceil(n_rules / rule_batch_size)
    log(f"{blue('GPU coverage pass:')} {cyan(f'{n_rules:,}')} {bold('rules x cracked universe')} "
        f"{cyan(f'{num_cracked:,}')} {dim(f'({bitmap_row_bytes/1024:.1f} KB/rule,')} "
        f"{cyan(str(total_rule_batches))} {dim(f'rule-batches of <= {rule_batch_size})')}")

    rules_batch_np = np.zeros((rule_batch_size, MAX_RULE_LEN), dtype=np.uint8)

    pbar = tqdm(total=total_rule_batches, desc=cyan("CELF coverage (rule batches)"),
                unit="batch", colour="cyan")
    for rb_idx, start in enumerate(range(0, n_rules, rule_batch_size)):
        end = min(start + rule_batch_size, n_rules)
        num_rules_here = end - start

        rules_batch_np[:num_rules_here] = encoded[start:end]
        rules_batch_np[num_rules_here:] = 0
        cl.enqueue_copy(queue, rules_g, rules_batch_np).wait()
        cl.enqueue_fill_buffer(queue, bitmap_g, np.uint32(0), 0, bitmap_buffer_bytes).wait()

        for words_np, num_words in optimized_wordlist_iterator(wordlist_path, MAX_WORD_LEN, words_per_gpu_batch):
            cl.enqueue_copy(queue, base_words_g, words_np).wait()

            rules_per_sub = max(1, min(num_rules_here, MAX_DISPATCH_ITEMS // max(num_words, 1)))
            for sub_start in range(0, num_rules_here, rules_per_sub):
                sub_end = min(sub_start + rules_per_sub, num_rules_here)
                sub_num = sub_end - sub_start
                sub_rules_g = rules_g.get_sub_region(sub_start * MAX_RULE_LEN, sub_num * MAX_RULE_LEN)
                sub_bitmap_g = bitmap_g.get_sub_region(sub_start * bitmap_row_bytes, sub_num * bitmap_row_bytes)
                global_size = (int(math.ceil(num_words * sub_num / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
                kernel(queue, global_size, (LOCAL_WORK_SIZE,),
                       base_words_g, sub_rules_g, cracked_g, sub_bitmap_g,
                       np.uint32(num_words), np.uint32(sub_num), np.uint32(MAX_WORD_LEN)).wait()

        # Pull this rule-batch's slice of the coverage bitmap back to a
        # small plain ndarray (at most rule_batch_size rows, a few MB) --
        # NOT the whole matrix. enqueue_copy for device->host infers the
        # copy size from the destination host array's shape (it does not
        # take a `size=` kwarg on this pyopencl version -- that was the
        # earlier crash); on the last (possibly partial) rule-batch,
        # num_rules_here < rule_batch_size, so we read only the valid
        # prefix via get_sub_region instead of over-reading.
        host_bitmap = np.zeros((num_rules_here, bitmap_words_per_rule), dtype=np.uint32)
        if num_rules_here == rule_batch_size:
            cl.enqueue_copy(queue, host_bitmap, bitmap_g).wait()
        else:
            sub_read_g = bitmap_g.get_sub_region(0, num_rules_here * bitmap_row_bytes)
            cl.enqueue_copy(queue, host_bitmap, sub_read_g).wait()

        # Compute this batch's popcounts first (SWAR, no 16x blow-up,
        # only over this small batch -- never the full matrix at once),
        # then write the rows into the matrix:
        #  - in_ram:        plain in-RAM slice assignment, dense, as before
        #  - hybrid (disk):  each row packed to whichever of dense/sparse
        #                     is smaller, appended to the data file
        #  - --no-hybrid:   dense, fixed-stride memmap slice, as before
        batch_popcounts = popcount_rows(host_bitmap)
        initial_gains[start:end] = batch_popcounts
        if bitmaps is not None:
            bitmaps[start:end] = host_bitmap
        else:
            writer.write_batch(start, host_bitmap, batch_popcounts)
        del host_bitmap

        pbar.set_postfix({"rss_mb": f"{get_rss_mb():.0f}"})
        pbar.update(1)
    pbar.close()

    if bitmaps is not None:
        if hasattr(bitmaps, 'flush'):
            bitmaps.flush()
        return bitmaps, bitmap_words_per_rule, initial_gains

    writer.close()
    actual_bytes = writer.bytes_written()
    actual_gb = actual_bytes / (1024 ** 3)
    saved_pct = 100.0 * (1 - actual_bytes / max(est_bytes, 1))
    log(f"{green('Hybrid coverage file written:')} {cyan(f'{actual_gb:.2f} GB')} "
        f"{dim(f'vs {est_gb:.2f} GB worst-case dense')} -- "
        f"{green(f'{saved_pct:.1f}% smaller')}")
    bitmaps = writer.to_store()
    return bitmaps, bitmap_words_per_rule, initial_gains


# ============================================================
# --- CELF lazy-greedy on CPU (bitwise, vectorised popcount) ---
# ============================================================
def celf_select(rules, bitmaps, initial_gains=None, budget=None):
    """bitmaps: (n_rules, W) uint32 -- ndarray OR memmap, doesn't matter,
    since only single-row accesses (bitmaps[idx]) happen here, which
    page in just that row regardless of backing store (a plain memory
    read for an in-RAM ndarray, a potential disk seek for a memmap).

    initial_gains: precomputed (n_rules,) popcounts for heap seeding.
    If omitted, falls back to computing them here in chunks (kept for
    callers that don't have a streaming pass to piggyback on -- still
    chunked so it never materializes the whole matrix as one temporary).
    Returns list[(rule, gain)] best-first.
    """
    import heapq

    n_rules = bitmaps.shape[0]
    W = bitmaps.shape[1]
    covered = np.zeros(W, dtype=np.uint32)

    log(f"{blue('Seeding heap for')} {cyan(f'{n_rules:,}')} {bold('candidates...')}")
    if initial_gains is None:
        chunk = 4096
        initial_gains = np.zeros(n_rules, dtype=np.int64)
        for s in range(0, n_rules, chunk):
            e = min(s + chunk, n_rules)
            initial_gains[s:e] = popcount_rows(np.asarray(bitmaps[s:e]))

    heap = []
    seed_pbar = tqdm(total=n_rules, desc=cyan("Seeding heap"), unit="rule", colour="cyan")
    rss_update_every = max(1, n_rules // 200)  # ~200 postfix refreshes, not one per rule
    for i in range(n_rules):
        g = int(initial_gains[i])
        if g > 0:
            heapq.heappush(heap, (-g, i, 0))
        if i % rss_update_every == 0 or i == n_rules - 1:
            seed_pbar.set_postfix({"rss_mb": f"{get_rss_mb():.0f}"})
        seed_pbar.update(1)
    seed_pbar.close()
    log(f"{green('Candidates with >=1 hit:')} {cyan(f'{len(heap):,}')}/{cyan(f'{n_rules:,}')}")

    limit = budget if budget else len(heap)
    selected = []
    version = 0
    revalidations = 0

    pbar = tqdm(total=limit, desc=cyan("CELF greedy select"), unit="rule", colour="cyan")
    while heap and len(selected) < limit:
        neg_gain, idx, s = heapq.heappop(heap)
        if s == version:
            gain = -neg_gain
            if gain <= 0:
                break
            selected.append((rules[idx], gain))
            covered |= bitmaps[idx]
            version += 1
            pbar.update(1)
            pbar.set_postfix({
                "covered_bits": int(popcount_row(covered)),
                "reval/pick": f"{revalidations / max(1, len(selected)):.1f}",
            })
            revalidations = 0
            continue
        # Lazy re-validation: recompute this rule's TRUE marginal gain
        # against the current covered set. bitmaps[idx] is a single-row
        # read (a few KB-MB), same cost whether bitmaps is a memmap or
        # an in-RAM ndarray -- EXCEPT that on a memmap backed by a file
        # much larger than available RAM, this read is a real random
        # disk seek, not a page-cache hit. A high reval/pick count here
        # combined with low rule/s is the signature of that: CELF is
        # I/O-bound on random reads of the coverage matrix, not CPU-bound
        # on the popcount itself. --in-ram mode avoids this entirely
        # since there's no disk file to seek against.
        revalidations += 1
        not_covered = np.bitwise_and(bitmaps[idx], np.bitwise_not(covered))
        new_gain = popcount_row(not_covered)
        if new_gain <= 0:
            continue
        heapq.heappush(heap, (-new_gain, idx, version))
    pbar.close()

    total_covered = int(popcount_row(covered))
    log(f"{green('Done.')} {bold('Selected')} {cyan(f'{len(selected):,}')} {bold('rules,')} "
        f"{bold('covering')} {cyan(f'{total_covered:,}')}/{cyan(f'{W*32:,}')} {bold('bit-slots')} "
        f"{dim('(exact cracked universe size may be slightly less than W*32)')}")
    return selected


# ============================================================
# --- CELF lazy-greedy on CPU, PARALLEL across all cores + threads
#     (multiprocessing.Pool of worker processes, each additionally
#     using a ThreadPoolExecutor of os.pread() calls) ---
#
# Why this exists: celf_select()'s cost is dominated by LAZY
# REVALIDATION -- re-popcounting one candidate row against the current
# `covered` set before trusting its heap position. Each revalidation is
# a single random-offset read of one row out of a file that's typically
# much bigger than RAM. That's disk-latency-bound, not compute-bound: a
# lone blocking reader only ever has ONE read outstanding, so you get
# roughly one seek's worth of IOPS no matter how fast the popcount
# itself is. A modern NVMe drive's real throughput only shows up once
# queue depth > 1 (often >>1); the fix is to have MANY row reads in
# flight at once, not a faster popcount.
#
# Two levels of concurrency here:
#   1. multiprocessing.Pool(processes=n_workers) -- real OS processes,
#      one per CPU core, each with its OWN file descriptor on
#      coverage.dat.
#   2. Inside each worker, a ThreadPoolExecutor(io_threads) issues
#      several os.pread() calls concurrently. os.pread() is a genuine
#      blocking syscall, which CPython always executes with the GIL
#      released, so io_threads reads really do overlap in the kernel/
#      on the device queue, not just in Python bytecode.
# Net queue depth: up to n_workers * io_threads reads in flight.
#
# Exactness: this is the textbook *batched* lazy-greedy construction,
# and it returns the identical selection order to celf_select(). The
# heap/version scheme is unchanged; the only difference is that instead
# of revalidating one stale top-of-heap entry at a time, each round
# pulls the current best `batch_size` STALE entries (stopping early if
# a fresh one surfaces), revalidates all of them against the *same*
# `covered` array in parallel (valid -- `covered` cannot change until a
# selection happens, and no selection happens mid-batch), and pushes
# them back with the current version stamp. Once a round produces a
# fresh top-of-heap entry, it is selected exactly as in the serial
# version.
# ============================================================
_wk_fd = None
_wk_row_bytes = None
_wk_offsets = None
_wk_kinds = None
_wk_W = None


def _celf_pool_init(bitmap_path, row_bytes, offsets=None, kinds=None, W=None):
    """Pool initializer: runs once per worker PROCESS. Opens its own
    read-only file descriptor on the coverage-bitmap file -- NOT a
    memmap, so that row reads go through plain os.pread() (see module
    docstring above for why that matters for real thread concurrency).

    Two on-disk formats, both use the same os.pread()-based worker
    path, just with a different byte range per row:
      - legacy dense fixed-stride (--no-hybrid): offsets/kinds/W are
        None, row i is always at [i*row_bytes, (i+1)*row_bytes).
      - hybrid dense/sparse (default): offsets/kinds/W are the small
        O(n_rules) arrays from HybridRowStore, duplicated into every
        worker process (a few MB, negligible) so each row's byte
        range and representation can be looked up locally with no
        cross-process calls.
    """
    global _wk_fd, _wk_row_bytes, _wk_offsets, _wk_kinds, _wk_W
    _wk_row_bytes = row_bytes
    _wk_offsets = offsets
    _wk_kinds = kinds
    _wk_W = W
    _wk_fd = os.open(bitmap_path, os.O_RDONLY)


def _celf_pool_read_row(idx):
    if _wk_offsets is not None:
        start = int(_wk_offsets[idx])
        end = int(_wk_offsets[idx + 1])
        raw = os.pread(_wk_fd, end - start, start)
        return _unpack_row(int(_wk_kinds[idx]), raw, _wk_W)
    raw = os.pread(_wk_fd, _wk_row_bytes, idx * _wk_row_bytes)
    return np.frombuffer(raw, dtype=np.uint32)


def _celf_pool_revalidate(task):
    """task = (idx_list, covered_bytes, io_threads).
    Returns [(idx, new_gain), ...] -- the TRUE marginal gain of each
    candidate row against the covered set, computed via `io_threads`
    concurrent os.pread() calls."""
    idx_list, covered_bytes, io_threads = task
    covered = np.frombuffer(covered_bytes, dtype=np.uint32)
    results = []
    if io_threads <= 1 or len(idx_list) <= 1:
        for idx in idx_list:
            row = _celf_pool_read_row(idx)
            not_covered = np.bitwise_and(row, np.bitwise_not(covered))
            results.append((idx, int(popcount_row(not_covered))))
        return results
    with ThreadPoolExecutor(max_workers=io_threads) as ex:
        rows = list(ex.map(_celf_pool_read_row, idx_list))
    for idx, row in zip(idx_list, rows):
        not_covered = np.bitwise_and(row, np.bitwise_not(covered))
        results.append((idx, int(popcount_row(not_covered))))
    return results


def _chunk_list(lst, n):
    """Split lst into at most n roughly-equal, non-empty chunks."""
    if n <= 1 or len(lst) <= 1:
        return [lst] if lst else []
    k, m = divmod(len(lst), n)
    chunks = []
    start = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        if size:
            chunks.append(lst[start:start + size])
            start += size
    return chunks


def celf_select_parallel(rules, bitmaps, bitmap_path, initial_gains,
                          budget=None, n_workers=None, io_threads=4,
                          batch_multiplier=8):
    """Multi-core, multi-threaded lazy-greedy CELF. See the block
    comment above this function for the full rationale.

    rules, initial_gains: same as celf_select().
    bitmaps: the memmap/ndarray from compute_coverage_bitmaps(), used
        ONLY for the (rare) selection-time `covered |= row` read in the
        main process -- one read per accepted rule, not per
        revalidation, so it's cheap regardless of backing store.
    bitmap_path: path to the on-disk coverage-bitmap file. REQUIRED --
        this function needs a real file for workers to open their own
        fd on (i.e. it doesn't apply in --in-ram mode; use celf_select()
        there, there's no disk bottleneck to parallelize away).
    n_workers: worker processes, default os.cpu_count().
    io_threads: concurrent os.pread() calls per worker process.
    batch_multiplier: batch_size = n_workers * io_threads * this, i.e.
        how much I/O queue-depth headroom each round requests versus
        the theoretical max concurrency, to keep workers fed even when
        individual reads finish at different times.
    Returns list[(rule, gain)] best-first -- identical to celf_select().
    """
    import heapq

    if not bitmap_path or not os.path.exists(bitmap_path):
        log(yellow("celf_select_parallel: no on-disk bitmap file "
                    "(--in-ram mode?) -- falling back to celf_select()."))
        return celf_select(rules, bitmaps, initial_gains=initial_gains, budget=budget)

    n_rules = bitmaps.shape[0]
    W = bitmaps.shape[1]
    row_bytes = W * 4  # only meaningful for the legacy dense fixed-stride format
    is_hybrid = isinstance(bitmaps, HybridRowStore)
    covered = np.zeros(W, dtype=np.uint32)

    n_workers = n_workers or max(1, os.cpu_count() or 1)
    batch_size = max(n_workers * io_threads * batch_multiplier, n_workers)

    log(f"{blue('[Parallel CELF] processes=')}{cyan(n_workers)} "
        f"{blue('io_threads/process=')}{cyan(io_threads)} "
        f"{blue('batch_size=')}{cyan(batch_size)} "
        f"{dim(f'(up to {n_workers * io_threads} concurrent row reads against {bitmap_path})')}")

    log(f"{blue('Seeding heap for')} {cyan(f'{n_rules:,}')} {bold('candidates...')}")
    heap = []
    for i in range(n_rules):
        g = int(initial_gains[i])
        if g > 0:
            heapq.heappush(heap, (-g, i, 0))
    log(f"{green('Candidates with >=1 hit:')} {cyan(f'{len(heap):,}')}/{cyan(f'{n_rules:,}')}")

    limit = budget if budget else len(heap)
    selected = []
    version = 0
    revalidations = 0

    # spawn (not fork): the parent process may still hold a pyopencl
    # Context/CommandQueue from compute_coverage_bitmaps() -- OpenCL/
    # CUDA driver state is generally not fork-safe. spawn re-imports
    # this module fresh in each worker (a few hundred ms, paid once for
    # the whole run) and never touches GPU state, which is also what
    # makes it portable to platforms without fork (Windows).
    ctx = mp.get_context('spawn')
    if is_hybrid:
        initargs = (bitmap_path, row_bytes, bitmaps.offsets, bitmaps.kinds, bitmaps.W)
    else:
        initargs = (bitmap_path, row_bytes)
    pool = ctx.Pool(processes=n_workers, initializer=_celf_pool_init,
                     initargs=initargs)

    pbar = tqdm(total=limit, desc=cyan("CELF greedy select [parallel]"), unit="rule", colour="cyan")
    try:
        while heap and len(selected) < limit:
            neg_gain, idx, s = heap[0]
            if s == version:
                heapq.heappop(heap)
                gain = -neg_gain
                if gain <= 0:
                    break
                selected.append((rules[idx], gain))
                covered |= np.asarray(bitmaps[idx])
                version += 1
                pbar.update(1)
                pbar.set_postfix({
                    "covered_bits": int(popcount_row(covered)),
                    "reval/pick": f"{revalidations / max(1, len(selected)):.1f}",
                })
                revalidations = 0
                continue

            # Pull a batch of the current best STALE entries (stop as
            # soon as a fresh one surfaces -- it's already correct and
            # will be selected next iteration without any I/O).
            batch_idx = []
            while heap and len(batch_idx) < batch_size:
                ng2, i2, s2 = heap[0]
                if s2 == version:
                    break
                heapq.heappop(heap)
                batch_idx.append(i2)
            if not batch_idx:
                continue

            revalidations += len(batch_idx)
            covered_bytes = covered.tobytes()
            tasks = [(chunk, covered_bytes, io_threads)
                     for chunk in _chunk_list(batch_idx, n_workers)]
            for result in pool.imap_unordered(_celf_pool_revalidate, tasks):
                for i2, new_gain in result:
                    if new_gain > 0:
                        heapq.heappush(heap, (-new_gain, i2, version))
            pbar.set_postfix({
                "covered_bits": int(popcount_row(covered)),
                "reval/pick": f"{revalidations / max(1, len(selected) + 1):.1f}",
                "batch": len(batch_idx),
            })
    finally:
        pool.close()
        pool.join()
        pbar.close()

    total_covered = int(popcount_row(covered))
    log(f"{green('Done.')} {bold('Selected')} {cyan(f'{len(selected):,}')} {bold('rules,')} "
        f"{bold('covering')} {cyan(f'{total_covered:,}')}/{cyan(f'{W*32:,}')} {bold('bit-slots')} "
        f"{dim('(exact cracked universe size may be slightly less than W*32)')}")
    return selected


# ============================================================
# --- Output ---
# ============================================================
def save_output(selected, output_path):
    base = os.path.splitext(output_path)[0]
    rule_path = output_path if output_path.endswith('.rule') else base + '.rule'
    csv_path = base + '_celf.csv'

    with open(rule_path, 'w', newline='\n', encoding='utf-8') as f:
        f.write(":\n")
        for rule, _gain in selected:
            f.write(f"{rule}\n")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Rank', 'Marginal_Gain', 'Rule_Data'])
        for i, (rule, gain) in enumerate(selected, 1):
            w.writerow([i, gain, rule])
    log(f"{green('Saved')} {cyan(f'{len(selected):,}')} {bold('rules to')} {rule_path}")
    log(f"{green('Saved selection detail to')} {csv_path}")


def parse_budgets(budgets_str):
    """'64,250,5000' -> sorted unique list of ints. Raises on bad input."""
    if not budgets_str:
        return []
    out = set()
    for part in budgets_str.split(','):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n <= 0:
            raise ValueError(f"--budgets values must be positive, got {n}")
        out.add(n)
    return sorted(out)


def save_output_multi(selected, output_path, budgets):
    """Save one .rule/.csv pair per budget cutoff, e.g. --budgets 64,250,5000
    produces <base>_top64.rule, <base>_top250.rule, <base>_top5000.rule.
    Just slicing prefixes of one CELF run (CELF's selection order is a
    nested sequence of near-optimal solutions), no re-running CELF.
    """
    base = os.path.splitext(output_path)[0]
    ext = os.path.splitext(output_path)[1] or '.rule'
    for n in budgets:
        if n > len(selected):
            log(f"{yellow('Warning:')} --budgets {cyan(str(n))} {bold('exceeds')} "
                f"{cyan(f'{len(selected):,}')} {bold('selected rules')} "
                f"{dim('(saturation reached earlier)')} -- {bold('writing all')} {cyan(f'{len(selected):,}')}")
        subset = selected[:n]
        path = f"{base}_top{n}{ext}"
        save_output(subset, path)


# ============================================================
# --- Main ---
# ============================================================
def main(argv=None):
    global MAX_WORD_LEN, MAX_RULE_LEN, MAX_OUTPUT_LEN
    ap = argparse.ArgumentParser(description="CELF greedy max-coverage post-stage for ranker_v5.2")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('-r', '--ranking-csv', help="ranker_v5.2 output CSV (legacy or MAB mode)")
    src.add_argument('-f', '--rules-file', help="Plain .rule file (already ranked/optimized)")
    ap.add_argument('-w', '--wordlist', required=True)
    ap.add_argument('-k', '--cracked', required=True, help="Cracked passwords list")
    ap.add_argument('-o', '--output', required=True, help="Output .rule path")
    ap.add_argument('-c', '--candidates', type=int, default=20000,
                     help="How many top-scored rules to feed into CELF (default 20000). "
                          "This is NOT the final selection size -- see --budget.")
    ap.add_argument('-b', '--budget', type=int, default=None,
                     help="Max rules in final selection (default: run to saturation). "
                          "Ignored if --budgets is given.")
    ap.add_argument('-B', '--budgets', type=str, default=None,
                     help="Comma-separated list of budget cutoffs to export as "
                          "separate files in one CELF run, e.g. '64,250,5000'.")
    ap.add_argument('-R', '--rule-batch-size', type=int, default=1024,
                     help="Candidate rules evaluated per GPU dispatch batch. Also "
                          "controls peak host RAM for the coverage pass in memmap "
                          "mode (this many rows are ever live in a plain ndarray "
                          "at once). Ignored as a RAM bound in --in-ram mode, where "
                          "the whole matrix is resident regardless.")
    ap.add_argument('-W', '--words-batch-size', type=int, default=DEFAULT_WORDS_PER_GPU_BATCH)
    ap.add_argument('--max-word-len', type=int, default=MAX_WORD_LEN,
                     help=f"Words/cracked entries longer than this are SKIPPED "
                          f"entirely (not truncated). Also sizes the GPU kernel's "
                          f"per-thread word buffer -- smaller means less private "
                          f"memory per thread and less register spilling on weak "
                          f"GPUs, but check your wordlist's max line length first "
                          f"(default {MAX_WORD_LEN}).")
    ap.add_argument('--max-rule-len', type=int, default=MAX_RULE_LEN,
                     help=f"Max characters per hashcat rule considered "
                          f"(default {MAX_RULE_LEN}). Longer rules are silently "
                          f"truncated when encoded for the GPU (rare in practice "
                          f"-- real hashcat rules are almost always well under "
                          f"this).")
    ap.add_argument('--max-output-len', type=int, default=MAX_OUTPUT_LEN,
                     help=f"Max length of a rule's output word the GPU kernel "
                          f"will produce (default {MAX_OUTPUT_LEN}). If a rule "
                          f"would produce something longer, that command is "
                          f"skipped for that word (same as today, just a lower "
                          f"ceiling). Must be >= --max-word-len. Ignored if "
                          f"--auto-max-output-len is set.")
    ap.add_argument('--auto-max-output-len', action='store_true',
                     help="Before the GPU pass, run a fast CPU-only static "
                          "estimate over every candidate rule (length "
                          "transformations only, e.g. d/f/q/p/z/Z/y/Y/^/$ -- "
                          "not actual word content) against --max-word-len, "
                          "and set --max-output-len to the worst case found "
                          "(+ small margin) instead of using the fixed "
                          "default/flag value. Conservative upper bound: "
                          "assumes data-dependent rule rejects (!,/,(,),_,%%,=) "
                          "never trigger, so it can be larger than strictly "
                          "necessary but never smaller.")
    ap.add_argument('--print-output-len-estimate', action='store_true',
                     help="Run the same static estimate as "
                          "--auto-max-output-len, print the recommended "
                          "--max-output-len and the rule responsible for the "
                          "worst case, then exit without touching the GPU. "
                          "Useful to sanity-check before a long run.")
    ap.add_argument('-d', '--device', type=int, default=None)
    ap.add_argument('--bitmap-path', type=str, default=None,
                     help="Where to stream the coverage-bitmap matrix on disk "
                          "(default: <output_base>.bitmap.dat, next to --output). "
                          "Needs roughly candidates x cracked_universe/8 bytes free "
                          "-- the 'estimated' size printed at startup. Put it on "
                          "fast local storage (NVMe/SSD), not network storage. "
                          "Ignored if --in-ram is set.")
    ap.add_argument('--keep-bitmap', action='store_true',
                     help="Don't delete the on-disk bitmap file after a successful "
                          "run. Ignored if --in-ram is set (no file is ever created).")
    ap.add_argument('--no-hybrid', action='store_true',
                     help="Use the old dense, fixed-stride on-disk bitmap format "
                          "instead of the default hybrid dense/sparse format. The "
                          "hybrid format picks whichever representation is smaller "
                          "per-row (from its already-computed popcount) so it is "
                          "never larger than this legacy format and is usually much "
                          "smaller when most candidate rules have sparse coverage "
                          "(the common case: a steep drop-off in Marginal_Gain from "
                          "`rank`/`handler` output is a good sign hybrid will help a "
                          "lot). Ignored if --in-ram is set.")
    ap.add_argument('--in-ram', action='store_true',
                     help="Build the coverage bitmap matrix as a plain in-RAM "
                          "ndarray instead of a disk-backed memmap. Faster overall "
                          "-- no disk I/O for the write pass or for CELF's lazy "
                          "re-validation row reads -- but needs the full "
                          "'estimated' size printed at startup as free RAM. "
                          "--bitmap-path and --keep-bitmap are ignored in this mode.")
    ap.add_argument('--no-parallel-celf', action='store_true',
                     help="Disable the multiprocessing CELF select and use "
                          "the plain single-threaded celf_select() instead. "
                          "By default (disk-backed bitmap, i.e. no --in-ram), "
                          "CELF's greedy-select phase runs across all CPU "
                          "cores (multiprocessing.Pool) with each worker "
                          "additionally issuing several concurrent "
                          "os.pread() reads, to keep many random reads "
                          "against the coverage-bitmap file in flight at "
                          "once -- this is what actually fixes low rules/s "
                          "during lazy revalidation (I/O queue depth).")
    ap.add_argument('--celf-workers', type=int, default=None,
                     help="Worker processes for parallel CELF select "
                          "(default: os.cpu_count()).")
    ap.add_argument('--celf-io-threads', type=int, default=4,
                     help="Concurrent os.pread() calls per worker process "
                          "in parallel CELF select (default 4). Raise this "
                          "on fast NVMe with high native queue depth; total "
                          "concurrent reads in flight is roughly "
                          "--celf-workers x --celf-io-threads.")
    ap.add_argument('--celf-batch-multiplier', type=int, default=8,
                     help="Parallel CELF revalidates up to "
                          "workers x io_threads x this-many candidates per "
                          "round (default 8), to keep all workers fed even "
                          "as individual reads finish at different times.")
    args = ap.parse_args(argv)

    MAX_WORD_LEN = args.max_word_len
    MAX_RULE_LEN = args.max_rule_len
    MAX_OUTPUT_LEN = args.max_output_len
    if MAX_OUTPUT_LEN < MAX_WORD_LEN and not args.auto_max_output_len:
        log(red(f"--max-output-len ({MAX_OUTPUT_LEN}) must be >= --max-word-len "
                f"({MAX_WORD_LEN}) -- rules that only extend words would be "
                f"silently no-op'd otherwise. Aborting. "
                f"(Or pass --auto-max-output-len to size it automatically.)"))
        sys.exit(1)
    log(f"{blue('Buffer limits:')} max-word-len={cyan(MAX_WORD_LEN)} "
        f"max-rule-len={cyan(MAX_RULE_LEN)} max-output-len={cyan(MAX_OUTPUT_LEN)} "
        f"{dim('(entries/rules exceeding these are skipped/truncated -- see --help)')}")

    t0 = time.time()
    rules = load_candidate_rules(args)

    if args.print_output_len_estimate or args.auto_max_output_len:
        worst_len, worst_rule = estimate_worst_case_output_len(rules, MAX_WORD_LEN)
        margin = max(8, worst_len // 16)  # small safety margin, rounded up below
        suggested = worst_len + margin
        log(f"{blue('Static output-length estimate:')} worst case "
            f"{cyan(str(worst_len))} chars for --max-word-len={cyan(MAX_WORD_LEN)} "
            f"{dim(f'(rule: {worst_rule!r})' if worst_rule else '(no rule grows the word)')} "
            f"-- {bold('suggested --max-output-len')} {cyan(str(suggested))} "
            f"{dim('(worst case + margin; conservative, see --help)')}")
        if args.print_output_len_estimate:
            sys.exit(0)
        if MAX_OUTPUT_LEN != suggested:
            log(f"{yellow('--auto-max-output-len:')} overriding --max-output-len "
                f"{cyan(str(MAX_OUTPUT_LEN))} -> {cyan(str(suggested))}")
        MAX_OUTPUT_LEN = suggested
        if MAX_OUTPUT_LEN < MAX_WORD_LEN:
            MAX_OUTPUT_LEN = MAX_WORD_LEN

    cracked_hashes, n_skipped = load_cracked_universe(args.cracked, MAX_WORD_LEN)
    if n_skipped:
        log(f"{yellow('Skipped')} {cyan(f'{n_skipped:,}')} {yellow('cracked entries longer than')} "
            f"{cyan(MAX_WORD_LEN)} {yellow('chars (not counted in coverage universe).')} "
            f"{dim('Raise --max-word-len if this matters for your data.')}")
    if len(cracked_hashes) == 0:
        log(red("Cracked list is empty -- nothing to optimize for. Aborting."))
        sys.exit(1)

    bitmap_path = args.bitmap_path or (os.path.splitext(args.output)[0] + '.bitmap.dat')

    bitmaps, _, initial_gains = compute_coverage_bitmaps(
        rules, args.wordlist, cracked_hashes,
        rule_batch_size=args.rule_batch_size,
        words_per_gpu_batch=args.words_batch_size,
        bitmap_path=bitmap_path,
        device_id=args.device,
        in_ram=args.in_ram,
        hybrid=not args.no_hybrid,
    )

    try:
        budgets = parse_budgets(args.budgets) if args.budgets else []
        run_budget = max(budgets) if budgets else args.budget

        use_parallel = (not args.in_ram) and (not args.no_parallel_celf)
        if use_parallel:
            selected = celf_select_parallel(
                rules, bitmaps, bitmap_path, initial_gains,
                budget=run_budget,
                n_workers=args.celf_workers,
                io_threads=args.celf_io_threads,
                batch_multiplier=args.celf_batch_multiplier,
            )
        else:
            selected = celf_select(rules, bitmaps, initial_gains=initial_gains, budget=run_budget)

        if budgets:
            save_output_multi(selected, args.output, budgets)
        else:
            save_output(selected, args.output)
    finally:
        # Release the reference before deleting the backing file(s)
        # (required on some platforms, e.g. Windows, for memmaps/mmaps)
        # and clean up unless the user wants to keep them (e.g. to
        # re-run celf_select with a different --budget without redoing
        # the GPU pass). None of this applies in --in-ram mode: there's
        # no file. The hybrid format has a second small sidecar index
        # file (<bitmap_path>.index.npz) alongside the data file.
        if hasattr(bitmaps, 'close'):
            bitmaps.close()
        del bitmaps
        if not args.in_ram:
            paths_to_clean = [bitmap_path]
            index_path = HybridRowStore.index_path(bitmap_path)
            if os.path.exists(index_path):
                paths_to_clean.append(index_path)
            if not args.keep_bitmap:
                for p in paths_to_clean:
                    if os.path.exists(p):
                        os.remove(p)
                log(f"{dim('Removed temporary bitmap file(s):')} {', '.join(paths_to_clean)}")
            else:
                log(f"{blue('Kept bitmap file(s) at:')} {', '.join(paths_to_clean)}")

    print(f"\n{green('=' * 60)}")
    print(bold("CELF Post-Processing Complete"))
    print(f"{green('=' * 60)}")
    log(f"{blue('Total time:')} {cyan(f'{time.time() - t0:.1f}s')}")


if __name__ == '__main__':
    main()
