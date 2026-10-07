"""
sparse_coverage.py -- sparse per-rule coverage lists + pure-CPU lazy
greedy (CELF) selection. A third strategy alongside the dense/hybrid
bitmap matrix (ranker_postprocess.py's default) and the bitmap-free
GPU-rescan strategy (celf_recompute_gpu.py, --recompute-gpu).

WHY THIS EXISTS
---------------
The bitmap strategy's coverage matrix is O(n_candidates x
cracked_universe_bits) regardless of how sparse any individual rule's
real coverage is -- a rule that hits 3 cracked passwords out of a
million-entry universe still occupies (roughly) universe_bits/8 bytes
in a dense row, or close to it even in the hybrid dense/sparse format
once enough rules are non-trivially sparse-but-not-THAT-sparse. The
--recompute-gpu strategy avoids the matrix entirely but pays for it
with repeated GPU rescoring every CELF round -- fine when the final
budget is small relative to the candidate pool (most candidates get
pruned before ever reaching a rescore), but nearly always SLOWER than
the bitmap strategy's one-time pass when the budget is close to the
full pool, because early rounds must rescore large surviving
fractions of the pool over and over (see the block comment above
celf_select_recompute_gpu() in celf_recompute_gpu.py).

This strategy sidesteps both trade-offs by storing coverage the way it
actually looks for real hashcat rule pools: SPARSE. Each rule's
coverage is stored as an explicit, small array of the CRACKED-ARRAY
INDICES it hits -- not a bit per possible index. A rule that hits 3 of
a million-entry universe costs 12 bytes (3 x int32), not
universe_bits/8. This is computed via ONE GPU pass (same one-time cost
class as the bitmap strategy's build), then CELF greedy selection runs
ENTIRELY ON THE CPU against the stored sparse arrays -- no GPU
dispatch at all during the greedy rounds, so its cost is independent
of how many rounds/revalidations CELF needs, unlike --recompute-gpu.

Storage is either a plain in-memory dict (InMemorySparseCoverageStore,
used below SPARSE_DISK_THRESHOLD candidates -- zero overhead, same
memory profile as holding the sparse arrays in a dict directly) or a
small SQLite database on disk (SQLiteSparseCoverageStore, above the
threshold) so a candidate pool that wouldn't fit in RAM as a dict of
boxed numpy arrays can still be handled -- SQLite point-lookup cost
(single-digit microseconds once page-cache-warm) is negligible next to
the one-time GPU coverage pass this follows. Both types share the same
Mapping-like interface, so celf_select_sparse() doesn't need to know
or care which one it was handed.

This design (sparse coverage lists in a disk-backed key-value store,
pure-CPU CELF against it) mirrors rulest's selection.py/
coverage_store.py, adapted to this package's coverage model: rulest's
universe is distinct TARGET WORDS (verified exactly via a second GPU
kernel to eliminate bloom-filter false positives); this package's
universe is the CRACKED HASH LIST, deduplicated and sorted, with
binary_search_cracked() providing the same "no wasted storage for a
false match" property directly (an unmatched (word, rule) pair simply
produces no index at all, so there is no separate false-positive class
to filter out the way rulest's GPU bloom-filter prefilter needs to).
"""
import heapq
import math
import os
import sqlite3
import tempfile
import time
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import pyopencl as cl
from tqdm import tqdm

from .ranker_postprocess import (
    MAX_WORD_LEN, MAX_OUTPUT_LEN, MAX_RULE_LEN, LOCAL_WORK_SIZE,
    DEFAULT_WORDS_PER_GPU_BATCH, MAX_DISPATCH_ITEMS,
    log, red, green, yellow, blue, cyan, bold, dim, get_rss_mb,
    load_cracked_universe, load_candidate_rules,
    select_device, save_output, parse_budgets, save_output_multi,
)
# (select_device already imported above -- used by both the coverage
# backend and, now, _SparseCelfGpuBackend for --gpu-celf.)
from .celf_recompute_gpu import (
    _ResidentWordlistMixin, _COMMON_KERNEL_BODY, _build_open_addressing_table,
)

# Default batch size for --gpu-celf's stale-entry revalidation dispatch
# (see celf_select_sparse_gpu / _SparseCelfGpuBackend below).
DEFAULT_GPU_CELF_BATCH = 4096

# Empty-hits sentinel, shared to avoid allocating a fresh empty array
# for every zero-coverage rule (there are often many in a large pool).
EMPTY_HITS = np.empty(0, dtype=np.int32)

# Popcount of every possible byte value (0..255), used to vectorize
# _SparseCelfGpuBackend.covered_count()'s bitset popcount in numpy
# instead of a pure-Python `bin(w).count('1')` loop -- see that
# method's docstring for why the Python-loop version is a real
# bottleneck at hash-table-backed universe sizes (tens of millions of
# bits), not just a style preference.
_POPCOUNT_BYTE_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

# Candidate-pool size above which SparseCoverageStore switches from a
# plain in-memory dict to the SQLite-backed store. rule_ranker's
# typical --candidates default (20,000) sits comfortably below this --
# most runs never touch SQLite at all and pay zero overhead for it.
SPARSE_DISK_THRESHOLD = 100_000


# ============================================================
# --- Storage: dict below the threshold, SQLite above it ---
# ============================================================
class InMemorySparseCoverageStore(dict):
    """Plain-dict coverage store: rule INDEX (int, 0..n_rules-1) ->
    np.ndarray[int32] of the cracked-array indices that rule covers.
    Subclassing dict gives __getitem__/__contains__/__iter__/__len__/
    items()/get() for free at plain-dict speed (no I/O at all)."""

    def put_many(self, rows):
        for idx, arr in rows:
            self[idx] = arr

    def iter_candidates_with_hits(self):
        for idx, arr in self.items():
            n = len(arr)
            if n:
                yield idx, n

    def count_with_hits(self):
        return sum(1 for arr in self.values() if len(arr))

    def get_many(self, indices):
        """Batched counterpart to __getitem__ -- trivial here (no I/O,
        plain dict lookups), but present so callers (recompute_gains())
        can use the SAME code path regardless of store type instead of
        branching on store class."""
        return {int(i): self[int(i)] for i in indices}

    def close(self):
        pass


class SQLiteSparseCoverageStore(Mapping):
    """Disk-backed counterpart to InMemorySparseCoverageStore, used
    above SPARSE_DISK_THRESHOLD candidates -- same rationale as
    ranker_postprocess.HybridRowStore's memmap fallback and
    celf_recompute_gpu's resident-wordlist fallback: keep the run
    working past the point where the natural in-memory representation
    stops comfortably fitting in RAM, at a modest, well-amortized cost
    per access rather than failing outright.

    Backed by a temporary SQLite file by default (auto-deleted on
    close()); pass an explicit path to persist it. A small LRU cache
    sits in front of the per-rule point lookup, since CELF's lazy
    revalidation can re-check the same hot rule's coverage[idx] many
    times as it pops stale heap entries.
    """

    def __init__(self, path=None):
        self._owns_file = path is None
        if path is None:
            fd, path = tempfile.mkstemp(suffix='.db', prefix='rule_ranker_sparse_')
            os.close(fd)
        self._path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute('PRAGMA journal_mode = WAL')
        self._conn.execute('PRAGMA synchronous = OFF')
        self._conn.execute('PRAGMA temp_store = MEMORY')
        self._conn.execute('PRAGMA cache_size = -131072')  # ~128MB page cache
        self._conn.execute(
            'CREATE TABLE IF NOT EXISTS coverage ('
            ' idx    INTEGER PRIMARY KEY,'
            ' hits   BLOB NOT NULL,'   # raw int32 bytes; empty blob = no hits
            ' n_hits INTEGER NOT NULL'
            ')'
        )
        self._conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_coverage_n_hits '
            'ON coverage(n_hits) WHERE n_hits > 0'
        )
        self._conn.commit()
        self._closed = False
        self._len_cache = None
        self._getitem_cache = OrderedDict()
        self._getitem_cache_max = 512

    def put_many(self, rows):
        payload = [
            (int(idx), arr.astype(np.int32, copy=False).tobytes(), int(arr.size))
            for idx, arr in rows
        ]
        if not payload:
            return
        self._conn.executemany(
            'INSERT OR REPLACE INTO coverage (idx, hits, n_hits) VALUES (?,?,?)', payload)
        self._conn.commit()
        self._len_cache = None
        self._getitem_cache.clear()

    def __getitem__(self, idx):
        idx = int(idx)
        cached = self._getitem_cache.get(idx)
        if cached is not None:
            self._getitem_cache.move_to_end(idx)
            return cached
        row = self._conn.execute('SELECT hits FROM coverage WHERE idx = ?', (idx,)).fetchone()
        if row is None:
            raise KeyError(idx)
        blob = row[0]
        hits = np.frombuffer(blob, dtype=np.int32) if blob else EMPTY_HITS
        self._getitem_cache[idx] = hits
        self._getitem_cache.move_to_end(idx)
        if len(self._getitem_cache) > self._getitem_cache_max:
            self._getitem_cache.popitem(last=False)
        return hits

    def __contains__(self, idx):
        return self._conn.execute(
            'SELECT 1 FROM coverage WHERE idx = ?', (int(idx),)).fetchone() is not None

    def get_many(self, indices):
        """Batched counterpart to __getitem__: ONE SQL round trip
        (chunked only by SQLite's parameter-count limit, ~999) for a
        whole list of indices, instead of one `SELECT ... WHERE idx =
        ?` per index. CELF's lazy-revalidation batches (up to
        DEFAULT_GPU_CELF_BATCH=4096 stale heap entries at once) used to
        call __getitem__ in a plain Python loop here -- thousands of
        sequential single-row queries per CELF round, each paying its
        own cursor/round-trip overhead regardless of how fast the
        underlying disk is. This is the dominant cost of a GPU-CELF
        round once covered_count() itself is cheap (see
        _SparseCelfGpuBackend.covered_count()'s vectorized popcount).
        Returns {idx: hits_array}; indices already resident in the
        small getitem LRU cache are served from there without hitting
        SQLite at all, and any fetched rows populate that same cache
        (same eviction policy as __getitem__) so later single-item
        lookups of the same idx stay fast too. Raises KeyError if any
        requested idx isn't present (matching __getitem__'s contract)."""
        idx_list = [int(i) for i in indices]
        result = {}
        missing = []
        for idx in idx_list:
            cached = self._getitem_cache.get(idx)
            if cached is not None:
                self._getitem_cache.move_to_end(idx)
                result[idx] = cached
            else:
                missing.append(idx)

        CHUNK = 900  # stay under SQLite's default ~999 bound-parameter limit
        for cs in range(0, len(missing), CHUNK):
            chunk = missing[cs:cs + CHUNK]
            placeholders = ','.join('?' * len(chunk))
            cur = self._conn.execute(
                f'SELECT idx, hits FROM coverage WHERE idx IN ({placeholders})', chunk)
            for idx, blob in cur:
                hits = np.frombuffer(blob, dtype=np.int32) if blob else EMPTY_HITS
                result[idx] = hits
                self._getitem_cache[idx] = hits
                self._getitem_cache.move_to_end(idx)

        missing_keys = [i for i in idx_list if i not in result]
        if missing_keys:
            raise KeyError(missing_keys[0])

        while len(self._getitem_cache) > self._getitem_cache_max:
            self._getitem_cache.popitem(last=False)

        return result

    def __iter__(self):
        cur = self._conn.execute('SELECT idx FROM coverage')
        for (idx,) in cur:
            yield idx

    def __len__(self):
        if self._len_cache is None:
            self._len_cache = self._conn.execute('SELECT COUNT(*) FROM coverage').fetchone()[0]
        return self._len_cache

    def iter_candidates_with_hits(self):
        cur = self._conn.execute('SELECT idx, n_hits FROM coverage WHERE n_hits > 0')
        yield from cur

    def count_with_hits(self):
        row = self._conn.execute('SELECT COUNT(*) FROM coverage WHERE n_hits > 0').fetchone()
        return row[0] if row else 0

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._conn.close()
        finally:
            if self._owns_file:
                for suffix in ('', '-wal', '-shm'):
                    try:
                        os.unlink(self._path + suffix)
                    except OSError:
                        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ============================================================
# --- GPU kernels: count pass + batched sparse-extract pass ---
# ============================================================
def get_sparse_kernel_source(num_cracked, hash_table_size):
    """Two kernels, built on the exact same rule-transform/hash/
    hash-table-probe device code as celf_recompute_gpu.get_recompute_
    kernel_source() (imported via _COMMON_KERNEL_BODY, not re-derived
    here, so scoring semantics can never drift between strategies).

    This used to look up each transformed word's hash via
    binary_search_cracked() (O(log2(num_cracked)) dependent random
    global-memory reads per word/rule pair -- dominant cost at GPU
    scale once rule transforms themselves are cheap). It now uses the
    same open-addressing lookup_cracked_slot() probe
    celf_recompute_gpu.py's kernels use (<=50% load factor, ~1-2
    probes/lookup). The index each hit resolves to is therefore a
    HASH-TABLE SLOT (0..hash_table_size-1), not a compact
    0..num_cracked-1 cracked-array index -- callers must size any
    bitset/array keyed by these indices to hash_table_size, not
    num_cracked (see _SparseGpuBackend and compute_sparse_coverage_gpu).

    sparse_count_kernel   -- one (word, rule) pair per thread, exact
        hit COUNT per rule against the full cracked universe (static
        for this whole pass -- unlike celf_recompute_gpu's per-round
        shrinking `active` set, there is no "already covered" concept
        yet at coverage-evaluation time). Output is
        (num_rules_in_batch,) uint32, exactly like celf_recompute_gpu's
        score_against_active_kernel's gains[] but without the active-
        bitmap check.

    sparse_extract_kernel -- batched, multi-rule version: given each
        rule's exact count from the pass above (so a tightly-sized,
        contiguous packed output buffer can be allocated first) and
        each rule's prefix-sum offset into that buffer, writes the
        actual matched hash-table SLOT for every hit via atomic_inc on
        a per-rule write-position counter. One dispatch per rule-batch
        covers every rule in that batch in a single pass over the
        wordlist, the same batching shape as the count kernel and as
        celf_coverage_kernel in ranker_postprocess.py -- this is a
        genuinely two-pass computation (count, then extract) but both
        passes are O(n_candidates x wordlist), the SAME one-time order
        as the bitmap strategy's single pass, not something that
        repeats per CELF round the way --recompute-gpu's rescoring
        does.
    """
    return f"""
#define MAX_WORD_LEN {MAX_WORD_LEN}
#define MAX_OUTPUT_LEN {MAX_OUTPUT_LEN}
#define MAX_RULE_LEN {MAX_RULE_LEN}
#define NUM_CRACKED {num_cracked}
#define HASH_TABLE_SIZE {hash_table_size}
#define HASH_TABLE_MASK {hash_table_size - 1}

{_COMMON_KERNEL_BODY}

__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void sparse_count_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned char* rules_in,
    __global const unsigned int* hash_table,
    __global const unsigned int* hash_table_occupied,
    __global unsigned int* hit_counts,
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

    atomic_add(&hit_counts[rule_idx], 1u);
}}

// rule_offsets/rule_capacities/write_pos are all (num_rules_in_batch,);
// out_buffer is (sum(rule_capacities),) -- exactly sized by the caller
// from this same batch's sparse_count_kernel result, so under normal
// operation write_pos[r] never exceeds rule_capacities[r]; the bounds
// check is a safety net against the same rare cross-pass hash-
// collision edge case documented where this pattern is used elsewhere
// in this package (two matched words landing on the same 32-bit hash),
// not an expected occurrence.
__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void sparse_extract_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned char* rules_in,
    __global const unsigned int* hash_table,
    __global const unsigned int* hash_table_occupied,
    __global const unsigned int* rule_offsets,
    __global const unsigned int* rule_capacities,
    __global unsigned int* write_pos,
    __global unsigned int* out_buffer,
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

    unsigned int pos = atomic_inc(&write_pos[rule_idx]);
    if (pos < rule_capacities[rule_idx]) {{
        out_buffer[rule_offsets[rule_idx] + pos] = (unsigned int)slot;
    }}
}}

// Single-pass alternative to count_kernel + extract_kernel: writes
// matched hash-table slots directly, with no prior count pass, into a
// FIXED-STRIDE per-rule row of `out_buffer` (row length = `stride`,
// the caller-supplied total word count across the whole wordlist -- an
// upper bound on any one rule's hit count that can never be exceeded,
// so no count pass is needed to size anything). write_pos[rule_idx] is
// the atomic write cursor into that rule's row; the caller zeroes it
// ONCE per rule-batch, not per word-chunk, since this kernel is
// dispatched once per (rule-batch, word-chunk) pair and cursor
// positions must stay contiguous across chunks. Writes from atomic_inc
// are sequential (0, 1, 2, ...) so each row's valid hits always occupy
// the contiguous prefix [0, write_pos[rule_idx]) -- the same
// bounds-check-and-drop safety net as sparse_extract_kernel covers the
// same rare hash-collision edge case, not expected overflow (stride is
// a true upper bound by construction).
__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void sparse_combined_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned char* rules_in,
    __global const unsigned int* hash_table,
    __global const unsigned int* hash_table_occupied,
    __global unsigned int* write_pos,
    __global unsigned int* out_buffer,
    const unsigned int num_words,
    const unsigned int num_rules_in_batch,
    const unsigned int max_word_len,
    const unsigned int stride,
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

    unsigned int pos = atomic_inc(&write_pos[rule_idx]);
    if (pos < stride) {{
        out_buffer[rule_idx * stride + pos] = (unsigned int)slot;
    }}
}}
"""


class _SparseGpuBackend(_ResidentWordlistMixin):
    """Owns the OpenCL context/buffers for one compute_sparse_coverage_
    gpu() run. Reuses _ResidentWordlistMixin (see celf_recompute_gpu.py)
    for the exact same parse-once/stay-resident wordlist handling
    celf_recompute_gpu.py's _GpuScorer uses -- this backend also issues
    many dispatches over the same wordlist (one count + one extract
    pass per rule-batch), so the same disk-I/O elimination applies
    here.
    """

    def __init__(self, encoded_rules, num_cracked, cracked_hashes_sorted,
                 rule_batch_size, words_per_gpu_batch, device_id=None,
                 wordlist_path=None):
        self.n_rules = encoded_rules.shape[0]
        self.num_cracked = num_cracked
        self.rule_batch_size = rule_batch_size
        self.words_per_gpu_batch = words_per_gpu_batch
        self.encoded = encoded_rules

        # Open-addressed hash table instead of binary-searching the
        # sorted cracked array for every word/rule pair -- see
        # get_sparse_kernel_source()'s docstring and
        # celf_recompute_gpu._GpuScorer, whose __init__ builds the
        # exact same kind of table for the same reason (random global-
        # memory traffic dominates once rule transforms are cheap).
        # Hits now resolve to a SLOT in this table (0..hash_table_size-1),
        # not a compact 0..num_cracked-1 cracked-array index -- any
        # bitset/array keyed by stored hit indices (covered_bitset in
        # _SparseCelfGpuBackend, covered_mask in celf_select_sparse)
        # must be sized to hash_table_size, not num_cracked. The true
        # num_cracked is kept separately (self.num_cracked) for
        # %-coverage reporting, which must stay against the real
        # universe size, not the (sparser) table size.
        table_size = 1
        target_size = max(2, int(math.ceil(num_cracked * 2.0)))
        while table_size < target_size:
            table_size <<= 1
        self.hash_table_size = table_size
        self.hash_table_mask = table_size - 1

        platform, device = select_device(device_id)
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)
        src = get_sparse_kernel_source(num_cracked, self.hash_table_size)
        prg = cl.Program(self.context, src).build()
        self.count_kernel = prg.sparse_count_kernel
        self.extract_kernel = prg.sparse_extract_kernel
        self.combined_kernel = prg.sparse_combined_kernel

        mf = cl.mem_flags
        t_hash0 = time.time()
        hash_table, occupied = _build_open_addressing_table(
            cracked_hashes_sorted, self.hash_table_size, self.hash_table_mask)
        log(f"{dim(f'Hash table built for {num_cracked:,} cracked entries in {time.time() - t_hash0:.2f}s (vectorized)')}")
        self.hash_table_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=hash_table)
        self.hash_table_occupied_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                                hostbuf=occupied)

        words_buffer_size = words_per_gpu_batch * MAX_WORD_LEN * np.uint8().itemsize
        self.base_words_g = cl.Buffer(self.context, mf.READ_ONLY, words_buffer_size)

        rules_buffer_size = rule_batch_size * MAX_RULE_LEN * np.uint8().itemsize
        self.rules_g = cl.Buffer(self.context, mf.READ_ONLY, rules_buffer_size)

        self.hit_counts_g = cl.Buffer(self.context, mf.READ_WRITE,
                                       rule_batch_size * np.uint32().itemsize)
        self.rule_offsets_g = cl.Buffer(self.context, mf.READ_ONLY,
                                         rule_batch_size * np.uint32().itemsize)
        self.rule_capacities_g = cl.Buffer(self.context, mf.READ_ONLY,
                                            rule_batch_size * np.uint32().itemsize)
        self.write_pos_g = cl.Buffer(self.context, mf.READ_WRITE,
                                      rule_batch_size * np.uint32().itemsize)

        self.device = device
        self._word_chunks_gpu = None
        self._word_chunks_host = None
        self._host_fallback_words_g = None
        if wordlist_path is not None:
            self._preload_wordlist(wordlist_path)

        self._out_buffer_g = None
        self._out_buffer_capacity = 0

        self._combined_write_pos_g = cl.Buffer(
            self.context, mf.READ_WRITE, rule_batch_size * np.uint32().itemsize)
        self._combined_out_g = None
        self._combined_out_capacity = 0  # rows (= rule_batch_size); stride tracked separately
        self._combined_stride = 0

    def _ensure_out_buffer(self, capacity):
        if self._out_buffer_g is None or capacity > self._out_buffer_capacity:
            self._out_buffer_g = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE,
                max(1, capacity) * np.uint32().itemsize)
            self._out_buffer_capacity = capacity

    def combined_capacity_bytes(self, stride):
        """Worst-case GPU buffer size (bytes) combined_batch() would
        need for a full rule-batch at the given stride (total word
        count). Callers use this to decide, up front, whether the
        single-pass path fits their memory budget before ever calling
        combined_batch()."""
        return self.rule_batch_size * max(1, stride) * np.uint32().itemsize

    def _ensure_combined_buffer(self, stride):
        needed_rows = self.rule_batch_size
        if (self._combined_out_g is None or stride != self._combined_stride
                or needed_rows > self._combined_out_capacity):
            self._combined_out_g = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE,
                max(1, needed_rows * stride) * np.uint32().itemsize)
            self._combined_out_capacity = needed_rows
            self._combined_stride = stride

    def combined_batch(self, rule_indices, wordlist_path, stride):
        """Single-pass count+extract: for each rule in rule_indices,
        returns its exact sorted int32 array of matched cracked-array
        indices, with only ONE full pass over the wordlist (not two,
        unlike count_batch()+extract_batch()) -- see
        sparse_combined_kernel. `stride` must be >= the total word
        count the backend was preloaded with (self.total_words is the
        natural choice; it's a true upper bound on any single rule's
        hit count). Returns (counts, arrays), counts as int64 ndarray
        and arrays as a list of np.ndarray[int32] in rule_indices order.

        Memory cost is O(rule_batch_size x stride), fixed regardless of
        actual hit density -- callers should check combined_capacity_
        bytes(stride) against their memory budget before choosing this
        over count_batch()+extract_batch() for sparse-hit workloads
        where that product would be excessive.
        """
        self._ensure_combined_buffer(stride)
        n_total = len(rule_indices)
        counts = np.zeros(n_total, dtype=np.int64)
        results = [None] * n_total

        for cs in range(0, n_total, self.rule_batch_size):
            ce = min(cs + self.rule_batch_size, n_total)
            idx_chunk = rule_indices[cs:ce]
            n = len(idx_chunk)
            rules_batch_np = np.zeros((self.rule_batch_size, MAX_RULE_LEN), dtype=np.uint8)
            rules_batch_np[:n] = self.encoded[idx_chunk]
            cl.enqueue_copy(self.queue, self.rules_g, rules_batch_np)
            # Zeroed ONCE for the whole rule-batch (all word-chunks),
            # not per chunk -- write positions must stay contiguous
            # across chunks, see kernel docstring.
            cl.enqueue_fill_buffer(self.queue, self._combined_write_pos_g, np.uint32(0), 0,
                                    self.rule_batch_size * np.uint32().itemsize)

            for words_g, num_words in self._iter_word_chunks(wordlist_path):
                rules_per_sub = max(1, min(n, MAX_DISPATCH_ITEMS // max(num_words, 1)))
                for sub_start in range(0, n, rules_per_sub):
                    sub_end = min(sub_start + rules_per_sub, n)
                    sub_num = sub_end - sub_start
                    sub_rules_g = self.rules_g.get_sub_region(
                        sub_start * MAX_RULE_LEN, sub_num * MAX_RULE_LEN)
                    sub_write_pos_g = self._combined_write_pos_g.get_sub_region(
                        sub_start * np.uint32().itemsize, sub_num * np.uint32().itemsize)
                    sub_out_g = self._combined_out_g.get_sub_region(
                        sub_start * stride * np.uint32().itemsize, sub_num * stride * np.uint32().itemsize)
                    global_size = (int(math.ceil(num_words * sub_num / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
                    self.combined_kernel(
                        self.queue, global_size, (LOCAL_WORK_SIZE,),
                        words_g, sub_rules_g, self.hash_table_g, self.hash_table_occupied_g,
                        sub_write_pos_g, sub_out_g,
                        np.uint32(num_words), np.uint32(sub_num), np.uint32(MAX_WORD_LEN),
                        np.uint32(stride), np.uint32(self.hash_table_mask))

            host_write_pos = np.zeros(self.rule_batch_size, dtype=np.uint32)
            cl.enqueue_copy(self.queue, host_write_pos, self._combined_write_pos_g).wait()
            chunk_counts = np.minimum(host_write_pos[:n].astype(np.int64), stride)
            counts[cs:ce] = chunk_counts

            if int(chunk_counts.sum()) > 0:
                host_rows = np.zeros((n, stride), dtype=np.uint32)
                sub_read_g = self._combined_out_g.get_sub_region(0, n * stride * np.uint32().itemsize)
                cl.enqueue_copy(self.queue, host_rows, sub_read_g).wait()
                for i in range(n):
                    cap = int(chunk_counts[i])
                    if cap == 0:
                        results[cs + i] = EMPTY_HITS
                    else:
                        arr = host_rows[i, :cap].astype(np.int32, copy=True)
                        arr.sort()
                        results[cs + i] = arr
            else:
                for i in range(n):
                    results[cs + i] = EMPTY_HITS

        return counts, results

    def count_batch(self, rule_indices, wordlist_path):
        """Exact hit count for each of rule_indices (row indices into
        self.encoded) against the full cracked universe. Returns
        (len(rule_indices),) int64 array. Same no-intermediate-.wait()
        reasoning as celf_recompute_gpu._GpuScorer.score_batch(): only
        the final host readback blocks."""
        out = np.zeros(len(rule_indices), dtype=np.int64)
        for cs in range(0, len(rule_indices), self.rule_batch_size):
            ce = min(cs + self.rule_batch_size, len(rule_indices))
            idx_chunk = rule_indices[cs:ce]
            n = len(idx_chunk)
            rules_batch_np = np.zeros((self.rule_batch_size, MAX_RULE_LEN), dtype=np.uint8)
            rules_batch_np[:n] = self.encoded[idx_chunk]
            cl.enqueue_copy(self.queue, self.rules_g, rules_batch_np)
            cl.enqueue_fill_buffer(self.queue, self.hit_counts_g, np.uint32(0), 0,
                                    self.rule_batch_size * np.uint32().itemsize)

            for words_g, num_words in self._iter_word_chunks(wordlist_path):
                rules_per_sub = max(1, min(n, MAX_DISPATCH_ITEMS // max(num_words, 1)))
                for sub_start in range(0, n, rules_per_sub):
                    sub_end = min(sub_start + rules_per_sub, n)
                    sub_num = sub_end - sub_start
                    sub_rules_g = self.rules_g.get_sub_region(
                        sub_start * MAX_RULE_LEN, sub_num * MAX_RULE_LEN)
                    sub_counts_g = self.hit_counts_g.get_sub_region(
                        sub_start * np.uint32().itemsize, sub_num * np.uint32().itemsize)
                    global_size = (int(math.ceil(num_words * sub_num / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
                    self.count_kernel(self.queue, global_size, (LOCAL_WORK_SIZE,),
                                       words_g, sub_rules_g,
                                       self.hash_table_g, self.hash_table_occupied_g, sub_counts_g,
                                       np.uint32(num_words), np.uint32(sub_num),
                                       np.uint32(MAX_WORD_LEN), np.uint32(self.hash_table_mask))

            host_counts = np.zeros(self.rule_batch_size, dtype=np.uint32)
            cl.enqueue_copy(self.queue, host_counts, self.hit_counts_g).wait()
            out[cs:ce] = host_counts[:n].astype(np.int64)
        return out

    def extract_batch(self, rule_indices, counts, wordlist_path):
        """Exact sorted int32 array of matched cracked-array indices for
        EVERY rule in rule_indices at once (one packed GPU output
        buffer, split back into per-rule arrays on the host), given
        their exact counts from a prior count_batch() call. Internally
        chunks to rule_batch_size, same as count_batch(). Returns a
        list of np.ndarray[int32], same order as rule_indices."""
        results = [None] * len(rule_indices)
        for cs in range(0, len(rule_indices), self.rule_batch_size):
            ce = min(cs + self.rule_batch_size, len(rule_indices))
            idx_chunk = rule_indices[cs:ce]
            n = len(idx_chunk)
            chunk_counts = np.asarray(counts[cs:ce], dtype=np.uint32)

            if int(chunk_counts.sum()) == 0:
                for i in range(n):
                    results[cs + i] = EMPTY_HITS
                continue

            offsets = np.zeros(n, dtype=np.uint32)
            if n > 1:
                np.cumsum(chunk_counts[:-1], out=offsets[1:])
            total_capacity = int(chunk_counts.sum())

            rules_batch_np = np.zeros((self.rule_batch_size, MAX_RULE_LEN), dtype=np.uint8)
            rules_batch_np[:n] = self.encoded[idx_chunk]
            cl.enqueue_copy(self.queue, self.rules_g, rules_batch_np)

            offsets_np = np.zeros(self.rule_batch_size, dtype=np.uint32)
            offsets_np[:n] = offsets
            capacities_np = np.zeros(self.rule_batch_size, dtype=np.uint32)
            capacities_np[:n] = chunk_counts
            cl.enqueue_copy(self.queue, self.rule_offsets_g, offsets_np)
            cl.enqueue_copy(self.queue, self.rule_capacities_g, capacities_np)
            cl.enqueue_fill_buffer(self.queue, self.write_pos_g, np.uint32(0), 0,
                                    self.rule_batch_size * np.uint32().itemsize)
            self._ensure_out_buffer(total_capacity)

            for words_g, num_words in self._iter_word_chunks(wordlist_path):
                rules_per_sub = max(1, min(n, MAX_DISPATCH_ITEMS // max(num_words, 1)))
                for sub_start in range(0, n, rules_per_sub):
                    sub_end = min(sub_start + rules_per_sub, n)
                    sub_num = sub_end - sub_start
                    sub_rules_g = self.rules_g.get_sub_region(
                        sub_start * MAX_RULE_LEN, sub_num * MAX_RULE_LEN)
                    sub_offsets_g = self.rule_offsets_g.get_sub_region(
                        sub_start * np.uint32().itemsize, sub_num * np.uint32().itemsize)
                    sub_capacities_g = self.rule_capacities_g.get_sub_region(
                        sub_start * np.uint32().itemsize, sub_num * np.uint32().itemsize)
                    sub_write_pos_g = self.write_pos_g.get_sub_region(
                        sub_start * np.uint32().itemsize, sub_num * np.uint32().itemsize)
                    global_size = (int(math.ceil(num_words * sub_num / LOCAL_WORK_SIZE)) * LOCAL_WORK_SIZE,)
                    self.extract_kernel(
                        self.queue, global_size, (LOCAL_WORK_SIZE,),
                        words_g, sub_rules_g, self.hash_table_g, self.hash_table_occupied_g,
                        sub_offsets_g, sub_capacities_g, sub_write_pos_g,
                        self._out_buffer_g,
                        np.uint32(num_words), np.uint32(sub_num), np.uint32(MAX_WORD_LEN),
                        np.uint32(self.hash_table_mask))

            host_packed = np.zeros(total_capacity, dtype=np.uint32)
            sub_out_g = self._out_buffer_g.get_sub_region(0, total_capacity * np.uint32().itemsize)
            cl.enqueue_copy(self.queue, host_packed, sub_out_g).wait()

            for i in range(n):
                start = int(offsets[i])
                cap = int(chunk_counts[i])
                if cap == 0:
                    results[cs + i] = EMPTY_HITS
                else:
                    arr = host_packed[start:start + cap].astype(np.int32, copy=True)
                    arr.sort()
                    results[cs + i] = arr
        return results


# ============================================================
# --- Orchestration: GPU coverage pass -> sparse store ---
# ============================================================
DEFAULT_SPARSE_COMBINED_BUDGET_BYTES = 0  # disabled by default -- see below

# The single-pass combined kernel trades 2 full binary-search passes
# over the wordlist for 1, but at the cost of a FIXED-SIZE
# (rule_batch_size x total_words) device->host transfer every batch,
# regardless of how few actual hits there are. Real hashcat-style
# candidate rules typically match a tiny fraction of the wordlist, so
# in practice that transfer (hundreds of MB to low GBs per batch,
# every batch) dwarfs the one compute pass it saved -- measured ~5x
# SLOWER than the original two-pass path on a real run (1M rules x
# 296K words x 57M cracked), not faster. The two-pass path only ever
# transfers data proportional to actual hit counts (known exactly
# after count_kernel), which is why it stays the default here despite
# doing the binary-search work twice. Left in as an explicit opt-in
# (--sparse-combined-budget-mb) for the narrow case where the wordlist
# is small AND rules are expected to match a large fraction of it, but
# do not enable it without measuring first.


def compute_sparse_coverage_gpu(rules, wordlist_path, cracked_hashes_sorted,
                                 rule_batch_size, words_per_gpu_batch,
                                 device_id=None, disk_threshold=SPARSE_DISK_THRESHOLD,
                                 store_path=None,
                                 combined_budget_bytes=DEFAULT_SPARSE_COMBINED_BUDGET_BYTES):
    """One-time GPU pass (count, then extract, per rule-batch -- see
    get_sparse_kernel_source()) that fills and returns a sparse
    coverage store: rule index -> np.ndarray[int32] of the
    cracked-array indices it covers. Total GPU work is the same O(
    n_candidates x wordlist) order as compute_coverage_bitmaps()'s
    single pass (2x the per-pair work, for the count and extract
    passes, not 2x the wall-clock necessarily since both are GPU-bound
    the same way) -- this cost is paid exactly once, regardless of how
    many CELF rounds/revalidations celf_select_sparse() ends up needing
    afterward, unlike --recompute-gpu.

    Returns (store, initial_counts, universe_size) -- store is an
    InMemorySparseCoverageStore (n_rules <= disk_threshold) or
    SQLiteSparseCoverageStore (above it); initial_counts is an
    (n_rules,) int64 array of each rule's raw hit count (used only to
    log a summary here -- celf_select_sparse() reads its own seed
    counts back out of the store, not from this array, so the two
    strategies can't silently disagree); universe_size is the hash
    table size backing the GPU coverage pass's lookup_cracked_slot()
    probe (see get_sparse_kernel_source()) -- the indices stored in
    `store` are slots in a table of this size (>= num_cracked, not
    equal to it), so callers MUST pass universe_size (not
    len(cracked_hashes_sorted)) as the bitset/covered-array size to
    celf_select_sparse()/celf_select_sparse_gpu(); the true
    len(cracked_hashes_sorted) remains the right value for %-coverage
    reporting and is passed separately as cracked_size.
    """
    n_rules = len(rules)
    num_cracked = len(cracked_hashes_sorted)
    use_disk = n_rules > disk_threshold
    storage_kind = "SQLite disk-backed" if use_disk else "in-memory dict"
    log(f"{blue('Sparse coverage pass:')} {cyan(f'{n_rules:,}')} candidates x "
        f"{cyan(f'{num_cracked:,}')} cracked-universe entries "
        f"{dim(f'(storage: {storage_kind}, threshold {disk_threshold:,})')}")

    encoded = np.zeros((n_rules, MAX_RULE_LEN), dtype=np.uint8)
    for i, r in enumerate(rules):
        rb = r.encode('latin-1', errors='ignore')[:MAX_RULE_LEN]
        encoded[i, :len(rb)] = np.frombuffer(rb, dtype=np.uint8)

    backend = _SparseGpuBackend(
        encoded, num_cracked, cracked_hashes_sorted,
        rule_batch_size, words_per_gpu_batch, device_id=device_id,
        wordlist_path=wordlist_path,
    )

    store = SQLiteSparseCoverageStore(path=store_path) if use_disk else InMemorySparseCoverageStore()
    initial_counts = np.zeros(n_rules, dtype=np.int64)
    all_idx = np.arange(n_rules, dtype=np.int64)

    # Single-pass (count+extract combined) vs. the original two-pass
    # path: the combined kernel needs a fixed-stride (rule_batch_size x
    # total_words) output buffer sized to the worst case up front,
    # since there's no count pass to learn the real, usually much
    # smaller, per-rule sizes from first. That's a straight memory-for-
    # speed trade: half the GPU passes over the wordlist (the dominant
    # cost once the wordlist is resident -- see compute_sparse_
    # coverage_gpu's docstring), at the cost of a larger fixed buffer
    # and a bigger device->host transfer per rule-batch. Below a
    # caller-configurable budget (default 2 GiB) that trade is worth
    # it; above it (huge wordlists and/or huge rule_batch_size) this
    # falls back to the original count_batch()+extract_batch() path
    # automatically, so there's no risk of this regressing memory
    # behavior on runs where the combined buffer wouldn't fit.
    combined_bytes = backend.combined_capacity_bytes(backend.total_words)
    use_combined = combined_bytes <= combined_budget_bytes
    combined_mb = combined_bytes / (1024 ** 2)
    budget_mb = combined_budget_bytes / (1024 ** 2)
    if use_combined:
        detail = f"(fixed-stride buffer {combined_mb:.0f} MB/batch <= budget {budget_mb:.0f} MB)"
        mode_note = green("single-pass (combined count+extract)") + " " + dim(detail)
    else:
        detail = (f"(combined buffer would need {combined_mb:.0f} MB/batch > budget {budget_mb:.0f} MB; "
                   "raise --sparse-combined-budget-mb to use the faster single-pass path if you have the VRAM)")
        mode_note = yellow("two-pass (count, then extract)") + " " + dim(detail)
    log(f"{blue('Coverage pass mode:')} " + mode_note)

    total_batches = math.ceil(n_rules / rule_batch_size)
    pbar = tqdm(total=total_batches, desc=cyan("Sparse coverage (rule batches)"),
                unit="batch", colour="cyan")
    for start in range(0, n_rules, rule_batch_size):
        end = min(start + rule_batch_size, n_rules)
        chunk_idx = all_idx[start:end]
        if use_combined:
            counts, arrays = backend.combined_batch(chunk_idx, wordlist_path, backend.total_words)
        else:
            counts = backend.count_batch(chunk_idx, wordlist_path)
            arrays = backend.extract_batch(chunk_idx, counts, wordlist_path)
        initial_counts[start:end] = counts
        store.put_many(zip((int(i) for i in chunk_idx), arrays))
        pbar.update(1)
        pbar.set_postfix({"rss_mb": f"{get_rss_mb():.0f}"})
    pbar.close()

    n_with_hits = int((initial_counts > 0).sum())
    log(f"{green('Done.')} rules with >=1 hit: {cyan(f'{n_with_hits:,}')}/{cyan(f'{n_rules:,}')}")
    return store, initial_counts, backend.hash_table_size


def _sparse_celf_gpu_kernel_source():
    """Two tiny kernels for the GPU-resident CELF loop (--gpu-celf).
    Unlike get_sparse_kernel_source() above, these don't touch
    apply_hashcat_rule()/hashing at all -- by this point coverage is
    already computed (compute_sparse_coverage_gpu() ran once), so all
    that's left is bit-set bookkeeping against the sparse hit lists:

    celf_gain_kernel     -- one thread per QUERIED candidate (a batch
        of heap entries whose lazy 'gain' bound went stale). Each
        thread walks that candidate's hit-index list once and counts
        how many of those cracked-universe indices are still unset in
        the GPU-resident covered bitset -- i.e. its true marginal gain
        against the CURRENT covered set. This is the GPU replacement
        for the host-side `int(np.count_nonzero(~covered_mask[c]))`
        line in celf_select_sparse().

    celf_mark_covered_kernel -- one thread per hit index of the single
        just-selected candidate; atomically sets that bit in the
        covered bitset. Replaces `covered_mask[store[idx]] = True`.

    Both operate on a packed uint32 bitset (1 bit per cracked-universe
    entry) instead of numpy's 1-byte-per-entry bool array, and both
    read from `hits_flat`/`offsets`/`lengths` -- the sparse store's
    per-candidate hit arrays concatenated ONCE into a single resident
    GPU buffer at setup (see _SparseCelfGpuBackend) -- so no per-round
    host->device transfer of coverage data is needed, only the tiny
    query_indices / gains_out buffers each round.
    """
    return """
// Operates on a small, freshly-built PER-BATCH buffer (offsets/
// lengths index into THIS batch's hits_flat only, not a global
// all-candidates buffer) -- gid IS the batch position, no separate
// query_indices indirection needed. Batch total hits is bounded by
// the caller's hit budget (see _SparseCelfGpuBackend), so plain
// uint32 offsets are safe here even though the FULL candidate pool's
// total hit count can exceed uint32 range.
__kernel void celf_gain_kernel(
    __global const unsigned int* hits_flat,
    __global const unsigned int* offsets,
    __global const unsigned int* lengths,
    __global const unsigned int* covered_bitset,
    __global unsigned int* gains_out,
    const unsigned int n)
{
    unsigned int gid = get_global_id(0);
    if (gid >= n) return;
    unsigned int off = offsets[gid];
    unsigned int len = lengths[gid];
    unsigned int gain = 0;
    for (unsigned int i = 0; i < len; i++) {
        unsigned int bit = hits_flat[off + i];
        unsigned int word = covered_bitset[bit >> 5];
        if (((word >> (bit & 31)) & 1u) == 0u) gain++;
    }
    gains_out[gid] = gain;
}

// hits_flat here is just the SELECTED candidate's own hit array,
// uploaded standalone (length usually a few thousand-tens of
// thousands of entries) -- no offset needed, always starts at 0.
__kernel void celf_mark_covered_kernel(
    __global const unsigned int* hits_flat,
    __global unsigned int* covered_bitset,
    const unsigned int length)
{
    unsigned int gid = get_global_id(0);
    if (gid >= length) return;
    unsigned int bit = hits_flat[gid];
    atomic_or(&covered_bitset[bit >> 5], (1u << (bit & 31)));
}
"""


DEFAULT_GPU_CELF_HIT_BUDGET = 50_000_000  # ~200 MB flat/batch at uint32

# Fraction of total GPU VRAM we're willing to use for the fully-
# resident CSR buffer (hits_flat + offsets + lengths + bitset) before
# falling back to the streaming backend. Leaves headroom for whatever
# else is already allocated on the device (driver overhead, other
# processes, the query/gain scratch buffers).
DEFAULT_GPU_CELF_VRAM_FRACTION = 0.7


def _peek_device_global_mem(device_id=None):
    """Read a GPU device's total VRAM (bytes) WITHOUT creating a
    cl.Context or printing the 'Using GPU: ...' banner that
    select_device() does -- this is just a capacity probe used to
    decide which backend to build, not the actual device selection
    (that still happens once, inside whichever backend gets picked)."""
    try:
        for p in cl.get_platforms():
            try:
                devices = p.get_devices()
            except Exception:
                continue
            gpus = [d for d in devices if d.type == cl.device_type.GPU]
            if gpus:
                dev = gpus[device_id] if device_id is not None and device_id < len(gpus) else gpus[0]
                return int(dev.get_info(cl.device_info.GLOBAL_MEM_SIZE))
    except Exception:
        pass
    return None


def _sparse_celf_gpu_resident_kernel_source():
    """Resident-mode counterpart to the two kernels in
    _sparse_celf_gpu_kernel_source(): same bit-set bookkeeping, but
    indexed through a GLOBAL per-rule offsets/lengths table (sized
    n_rules, uploaded once) plus an explicit query_indices array per
    dispatch, instead of a small per-batch-local buffer. This is what
    lets _SparseCelfGpuResidentBackend keep hits_flat/offsets/lengths
    resident on the GPU for the WHOLE run (built once in __init__)
    and never re-touch the host-side store during CELF rounds at
    all -- the thing that made the old all-resident design fast.
    """
    return """
__kernel void celf_gain_resident_kernel(
    __global const unsigned int* hits_flat,
    __global const unsigned int* offsets,
    __global const unsigned int* lengths,
    __global const unsigned int* covered_bitset,
    __global const unsigned int* query_indices,
    __global unsigned int* gains_out,
    const unsigned int n_queries)
{
    unsigned int gid = get_global_id(0);
    if (gid >= n_queries) return;
    unsigned int cand = query_indices[gid];
    unsigned int off = offsets[cand];
    unsigned int len = lengths[cand];
    unsigned int gain = 0;
    for (unsigned int i = 0; i < len; i++) {
        unsigned int bit = hits_flat[off + i];
        unsigned int word = covered_bitset[bit >> 5];
        if (((word >> (bit & 31)) & 1u) == 0u) gain++;
    }
    gains_out[gid] = gain;
}

__kernel void celf_mark_covered_resident_kernel(
    __global const unsigned int* hits_flat,
    __global unsigned int* covered_bitset,
    const unsigned int offset,
    const unsigned int length)
{
    unsigned int gid = get_global_id(0);
    if (gid >= length) return;
    unsigned int bit = hits_flat[offset + gid];
    atomic_or(&covered_bitset[bit >> 5], (1u << (bit & 31)));
}
"""


class _SparseCelfGpuResidentBackend:
    """Fully GPU-resident CELF backend: builds ONE flat CSR layout
    (hits_flat/offsets/lengths) from `store` ONCE, uploads it as
    read-only GPU buffers that live for the whole run, and never
    touches `store` (never mind SQL) again afterward. Every CELF
    round -- both gain revalidation and mark_covered -- is a pure
    GPU-resident operation against buffers that already live in VRAM.

    This trades the streaming backend's "works regardless of how big
    total_hits gets" guarantee for raw speed: the whole thing only
    works if hits_flat (total_hits * 4 bytes) + offsets/lengths
    (n_rules * 4 bytes each) + the covered bitset fit in VRAM at once.
    celf_select_sparse_gpu() checks that before picking this backend
    (see DEFAULT_GPU_CELF_VRAM_FRACTION) and falls back to
    _SparseCelfGpuBackend (streaming) if it doesn't.
    """

    def __init__(self, store, n_rules, cracked_size, total_hits, device_id=None,
                 universe_size=None):
        self.cracked_size = cracked_size
        self.universe_size = universe_size if universe_size is not None else cracked_size

        platform, device = select_device(device_id)
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)
        prg = cl.Program(self.context, _sparse_celf_gpu_resident_kernel_source()).build()
        self._gain_kernel = prg.celf_gain_resident_kernel
        self._mark_kernel = prg.celf_mark_covered_resident_kernel

        # --- one-time CSR build, straight from the store, into ONE
        # preallocated buffer (no list-of-arrays + concatenate, which
        # would transiently need 2-3x total_hits memory) ---
        offsets = np.zeros(n_rules, dtype=np.uint32)
        lengths = np.zeros(n_rules, dtype=np.uint32)
        hits_flat = np.empty(max(total_hits, 1), dtype=np.uint32)
        if hasattr(store, 'iter_candidates_with_hits'):
            idx_hits = ((idx, store[idx]) for idx, _n in store.iter_candidates_with_hits())
        else:
            idx_hits = ((idx, arr) for idx, arr in store.items() if len(arr))

        pos = 0
        _t0 = time.perf_counter()
        pbar = tqdm(total=total_hits, desc=cyan("GPU-CELF resident buffer build"),
                    unit="hit", unit_scale=True, colour="cyan")
        last_report = 0
        for idx, arr in idx_hits:
            n = len(arr)
            offsets[idx] = pos
            lengths[idx] = n
            if n:
                hits_flat[pos:pos + n] = arr
            pos += n
            if pos - last_report >= 1_000_000:
                pbar.update(pos - last_report)
                pbar.set_postfix({"rss_mb": f"{get_rss_mb():.0f}"})
                last_report = pos
        pbar.update(pos - last_report)
        pbar.close()
        log(f"{dim(f'[PROFILE] resident CSR build (host, one-time): {time.perf_counter() - _t0:.2f}s')}")

        self.offsets_host = offsets  # kept on host too -- mark_covered() needs offset/length by idx
        self.lengths_host = lengths

        _t1 = time.perf_counter()
        mf = cl.mem_flags
        self.hits_flat_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hits_flat)
        self.offsets_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=offsets)
        self.lengths_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lengths)
        hits_flat = None  # free the host copy; GPU has its own now
        log(f"{dim(f'[PROFILE] resident CSR upload to VRAM: {time.perf_counter() - _t1:.2f}s')}")

        n_words = (self.universe_size + 31) // 32
        self.covered_bitset_g = cl.Buffer(self.context, mf.READ_WRITE, size=max(1, n_words) * 4)
        cl.enqueue_fill_buffer(self.queue, self.covered_bitset_g, np.uint32(0), 0, max(1, n_words) * 4)

        self._q_cap = 0
        self._query_g = None
        self._gains_g = None
        self._ensure_query_buffers(4096)

        # profiling accumulators -- per-round cost only (setup is
        # logged separately above)
        self.t_gpu_kernel = 0.0
        self.t_mark_covered = 0.0
        self.t_covered_count = 0.0
        self.n_gain_dispatches = 0
        self.n_mark_calls = 0
        self.n_count_calls = 0

    def _ensure_query_buffers(self, n):
        if n <= self._q_cap:
            return
        mf = cl.mem_flags
        self._query_g = cl.Buffer(self.context, mf.READ_ONLY, size=max(1, n) * 4)
        self._gains_g = cl.Buffer(self.context, mf.READ_WRITE, size=max(1, n) * 4)
        self._q_cap = n

    def recompute_gains(self, candidate_indices):
        n = len(candidate_indices)
        if n == 0:
            return np.empty(0, dtype=np.uint32)
        self.n_gain_dispatches += 1
        _t0 = time.perf_counter()
        self._ensure_query_buffers(n)
        q = np.asarray(candidate_indices, dtype=np.uint32)
        cl.enqueue_copy(self.queue, self._query_g, q)
        self._gain_kernel(self.queue, (n,), None,
                           self.hits_flat_g, self.offsets_g, self.lengths_g,
                           self.covered_bitset_g, self._query_g, self._gains_g,
                           np.uint32(n))
        out = np.empty(n, dtype=np.uint32)
        cl.enqueue_copy(self.queue, out, self._gains_g).wait()
        self.t_gpu_kernel += time.perf_counter() - _t0
        return out

    def mark_covered(self, idx):
        _t0 = time.perf_counter()
        self.n_mark_calls += 1
        offset = int(self.offsets_host[idx])
        length = int(self.lengths_host[idx])
        if length:
            self._mark_kernel(self.queue, (length,), None,
                               self.hits_flat_g, self.covered_bitset_g,
                               np.uint32(offset), np.uint32(length))
            self.queue.finish()
        self.t_mark_covered += time.perf_counter() - _t0

    def covered_count(self):
        _t0 = time.perf_counter()
        self.n_count_calls += 1
        n_words = (self.universe_size + 31) // 32
        buf = np.empty(max(1, n_words), dtype=np.uint32)
        cl.enqueue_copy(self.queue, buf, self.covered_bitset_g)
        byte_view = buf.view(np.uint8)
        result = int(_POPCOUNT_BYTE_TABLE[byte_view].sum(dtype=np.int64))
        self.t_covered_count += time.perf_counter() - _t0
        return result

    def profile_report(self):
        lines = [
            "---- GPU-CELF profile (resident backend) ----",
            f"GPU gain kernel (incl. small upload/download): {self.t_gpu_kernel:8.2f}s  "
            f"({self.n_gain_dispatches} dispatches)",
            f"mark_covered() total:         {self.t_mark_covered:8.2f}s  ({self.n_mark_calls} calls)",
            f"covered_count() total:        {self.t_covered_count:8.2f}s  ({self.n_count_calls} calls)",
            "(no per-round store/SQL access -- hits_flat/offsets/lengths are VRAM-resident)",
            "-----------------------------------------------",
        ]
        return "\n".join(lines)


class _SparseCelfGpuBackend:
    """Owns the OpenCL buffers for one celf_select_sparse_gpu() run.

    STREAMING design -- does NOT build a single flat CSR layout for
    every candidate up front. With a large candidate pool / cracked
    universe, total hits across ALL candidates can run into the tens
    of billions (seen in practice: ~14.2B, ~57GB as uint32), which
    does not fit in either host RAM or GPU VRAM on a typical box (e.g.
    8GB RAM + 8GB VRAM) -- a one-time full-buffer build is simply not
    an option at that scale, not just slow.

    Instead, only `covered_bitset` (packed, cracked_size/32 uint32
    words -- a few MB even for tens of millions of cracked entries) is
    GPU-resident for the whole run. Every other buffer is built fresh,
    on demand, directly from the (kept-open) `store`:

    - recompute_gains(idx_list): pulls each candidate's hit array from
      `store` (point lookups -- the store already has a small LRU
      cache for hot re-checks), concatenates just THIS BATCH into a
      small flat buffer bounded by `hit_budget` (splitting internally
      if a batch's total hits would exceed it), uploads, dispatches,
      downloads the gains, and immediately frees the batch buffer.
    - mark_covered(idx): uploads just the ONE selected candidate's own
      hit array (typically thousands-tens of thousands of entries,
      not billions) and dispatches the mark kernel against it.

    This trades a small amount of redundant store I/O (a hot
    candidate's hit array may be re-fetched across rounds -- mitigated
    by the store's own LRU cache) for never needing more than
    `hit_budget` worth of flat buffer resident at any point, on either
    host or device -- safe on small-VRAM/small-RAM boxes regardless of
    how large total_hits is.
    """

    def __init__(self, store, cracked_size, device_id=None,
                 batch_size=DEFAULT_GPU_CELF_BATCH,
                 hit_budget=DEFAULT_GPU_CELF_HIT_BUDGET,
                 universe_size=None):
        self.store = store
        # cracked_size: true cracked-universe size, kept ONLY for
        # %-coverage reporting (covered_count() is independent of it).
        self.cracked_size = cracked_size
        # universe_size: size of the covered_bitset itself. Must be
        # >= the largest index that can appear in `store`'s hit
        # arrays. When the coverage store was built against the
        # open-addressing hash table (see compute_sparse_coverage_gpu),
        # that's the hash table size, not cracked_size -- defaults to
        # cracked_size for callers/tests that pass already-compact
        # 0..cracked_size-1 indices directly.
        self.universe_size = universe_size if universe_size is not None else cracked_size
        self.batch_size = batch_size
        self.hit_budget = max(1, hit_budget)

        platform, device = select_device(device_id)
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)
        prg = cl.Program(self.context, _sparse_celf_gpu_kernel_source()).build()
        self._gain_kernel = prg.celf_gain_kernel
        self._mark_kernel = prg.celf_mark_covered_kernel

        mf = cl.mem_flags
        n_words = (self.universe_size + 31) // 32
        self.covered_bitset_g = cl.Buffer(self.context, mf.READ_WRITE, size=max(1, n_words) * 4)
        cl.enqueue_fill_buffer(self.queue, self.covered_bitset_g, np.uint32(0), 0, max(1, n_words) * 4)

        # Scratch buffers for the batch pipeline, grown lazily and
        # reused across dispatches -- capped by hit_budget/batch_size
        # so they never need to hold more than one bounded batch.
        self._hits_cap = 0
        self._hits_flat_g = None
        self._off_cap = 0
        self._offsets_g = None
        self._lengths_g = None
        self._gains_g = None

        # --- profiling accumulators (PROFILE patch) ---
        self.t_store_fetch = 0.0      # time inside store.get_many()
        self.t_flat_build = 0.0       # python-side concat of hit arrays into `flat`
        self.t_gpu_upload = 0.0       # enqueue_copy host->device
        self.t_gpu_kernel = 0.0       # kernel dispatch + queue.finish
        self.t_gpu_download = 0.0     # enqueue_copy device->host
        self.t_mark_covered = 0.0     # whole mark_covered() call
        self.t_covered_count = 0.0    # whole covered_count() call
        self.n_gain_dispatches = 0
        self.n_mark_calls = 0
        self.n_count_calls = 0
        self.total_flat_elems = 0     # sum of `total` across all gain dispatches

    def _ensure_hits_buffer(self, n):
        if n <= self._hits_cap:
            return
        mf = cl.mem_flags
        cap = max(n, 1)
        self._hits_flat_g = cl.Buffer(self.context, mf.READ_ONLY, size=cap * 4)
        self._hits_cap = cap

    def _ensure_offset_buffers(self, n):
        if n <= self._off_cap:
            return
        mf = cl.mem_flags
        cap = max(n, 1)
        self._offsets_g = cl.Buffer(self.context, mf.READ_ONLY, size=cap * 4)
        self._lengths_g = cl.Buffer(self.context, mf.READ_ONLY, size=cap * 4)
        self._gains_g = cl.Buffer(self.context, mf.READ_WRITE, size=cap * 4)
        self._off_cap = cap

    def _dispatch_gain_subbatch(self, idx_chunk, arrays):
        """One GPU round-trip for a sub-batch whose combined hit count
        fits comfortably under hit_budget. `arrays` is the list of
        this sub-batch's hit arrays, already fetched from the store,
        in the same order as idx_chunk."""
        n = len(idx_chunk)
        lengths = np.fromiter((len(a) for a in arrays), dtype=np.uint32, count=n)
        offsets = np.zeros(n, dtype=np.uint32)
        if n > 1:
            np.cumsum(lengths[:-1], out=offsets[1:])
        total = int(lengths.sum())

        self._ensure_offset_buffers(n)
        self._ensure_hits_buffer(max(total, 1))

        # --- PROFILE patch: time each phase of one sub-batch dispatch ---
        self.n_gain_dispatches += 1
        self.total_flat_elems += total

        _t0 = time.perf_counter()
        if total:
            flat = np.empty(total, dtype=np.uint32)
            pos = 0
            for a in arrays:
                m = len(a)
                if m:
                    flat[pos:pos + m] = a
                    pos += m
        _t1 = time.perf_counter()
        self.t_flat_build += _t1 - _t0

        if total:
            cl.enqueue_copy(self.queue, self._hits_flat_g, flat, device_offset=0)
        cl.enqueue_copy(self.queue, self._offsets_g, offsets)
        cl.enqueue_copy(self.queue, self._lengths_g, lengths)
        self.queue.finish()
        _t2 = time.perf_counter()
        self.t_gpu_upload += _t2 - _t1

        self._gain_kernel(self.queue, (n,), None,
                           self._hits_flat_g, self._offsets_g, self._lengths_g,
                           self.covered_bitset_g, self._gains_g, np.uint32(n))
        self.queue.finish()
        _t3 = time.perf_counter()
        self.t_gpu_kernel += _t3 - _t2

        out = np.empty(n, dtype=np.uint32)
        cl.enqueue_copy(self.queue, out, self._gains_g)
        self.queue.finish()
        _t4 = time.perf_counter()
        self.t_gpu_download += _t4 - _t3

        return out

    def recompute_gains(self, candidate_indices):
        """Batched replacement for the CPU line
        `int(np.count_nonzero(~covered_mask[c]))` -- fetches each
        candidate's hit array from `store`, dispatches their true
        marginal gain against the CURRENT covered_bitset in as few GPU
        calls as hit_budget allows (almost always one, for the normal
        batch_size=4096-ish stale-revalidation batches), and returns
        results in the SAME order as candidate_indices.

        Fetches the WHOLE batch's hit arrays from `store` in one
        get_many() call (one SQL round trip on SQLiteSparseCoverageStore,
        chunked only by its bound-parameter limit) instead of indexing
        the store once per candidate in a Python loop -- at
        batch_size=4096-ish stale entries per CELF round, that used to
        mean thousands of sequential single-row SQL queries per round
        (each paying its own cursor/round-trip overhead on top of
        whatever the underlying disk costs), which dominates GPU-CELF
        wall-clock once covered_count() itself is cheap (see that
        method's docstring)."""
        idx_list = [int(i) for i in candidate_indices]
        n = len(idx_list)
        if n == 0:
            return np.empty(0, dtype=np.uint32)

        # --- PROFILE patch ---
        _ts0 = time.perf_counter()
        if hasattr(self.store, 'get_many'):
            arrays_by_idx = self.store.get_many(idx_list)
        else:
            # Fallback for a plain dict or any other bare Mapping
            # without a batched get_many() (both of this package's own
            # store types have one).
            arrays_by_idx = {i: self.store[i] for i in idx_list}
        self.t_store_fetch += time.perf_counter() - _ts0

        results = np.empty(n, dtype=np.uint32)
        i = 0
        while i < n:
            chunk_idx = []
            chunk_arrays = []
            chunk_hits = 0
            j = i
            while j < n:
                arr = arrays_by_idx[idx_list[j]]
                m = len(arr)
                # Always take at least one candidate per sub-batch
                # even if it alone exceeds hit_budget (a single huge
                # rule's hit list still has to go somewhere) --
                # otherwise an outlier-large candidate would spin here
                # forever.
                if chunk_idx and chunk_hits + m > self.hit_budget:
                    break
                chunk_idx.append(idx_list[j])
                chunk_arrays.append(arr)
                chunk_hits += m
                j += 1
            gains = self._dispatch_gain_subbatch(chunk_idx, chunk_arrays)
            results[i:j] = gains
            i = j
        return results

    def mark_covered(self, idx):
        """GPU replacement for `covered_mask[store[idx]] = True` --
        uploads just this one selected candidate's hit array (NOT a
        slice of some larger resident buffer) and sets its bits in the
        resident covered bitset."""
        _tm0 = time.perf_counter()
        self.n_mark_calls += 1
        arr = self.store[idx]
        length = len(arr)
        if length == 0:
            self.t_mark_covered += time.perf_counter() - _tm0
            return
        flat = np.ascontiguousarray(arr, dtype=np.uint32)
        self._ensure_hits_buffer(length)
        cl.enqueue_copy(self.queue, self._hits_flat_g, flat, device_offset=0)
        self._mark_kernel(self.queue, (int(length),), None,
                           self._hits_flat_g, self.covered_bitset_g,
                           np.uint32(length))
        self.queue.finish()
        self.t_mark_covered += time.perf_counter() - _tm0

    def covered_count(self):
        """Host-side popcount of the bitset, only for the tqdm
        progress display / final log line -- O(cracked_size/32), not
        O(n_candidates x cracked_size), and only paid once per
        selected rule, not per revalidation.

        Vectorized via a 256-entry byte popcount lookup table (numpy),
        NOT a pure-Python `sum(bin(w).count('1') for w in ...)` loop --
        that loop costs O(n_words) PYTHON-level work, re-paid on every
        single accepted rule (once per CELF round, up to `budget`
        times), which dominates wall-clock at real hashcat scale (the
        hash-table-backed coverage pass's universe_size is ~2x
        cracked_size -- see _SparseGpuBackend -- so this got twice as
        expensive there too). The lookup-table version does the same
        popcount in vectorized numpy C code instead."""
        _tc0 = time.perf_counter()
        self.n_count_calls += 1
        n_words = (self.universe_size + 31) // 32
        buf = np.empty(max(1, n_words), dtype=np.uint32)
        cl.enqueue_copy(self.queue, buf, self.covered_bitset_g)
        byte_view = buf.view(np.uint8)
        result = int(_POPCOUNT_BYTE_TABLE[byte_view].sum(dtype=np.int64))
        self.t_covered_count += time.perf_counter() - _tc0
        return result

    def profile_report(self):
        """PROFILE patch: human-readable breakdown of where time went."""
        lines = [
            "---- GPU-CELF profile ----",
            f"store fetch (get_many):      {self.t_store_fetch:8.2f}s",
            f"flat buffer build (python):  {self.t_flat_build:8.2f}s  "
            f"({self.n_gain_dispatches} dispatches, {self.total_flat_elems:,} elems total)",
            f"GPU upload (h2d):             {self.t_gpu_upload:8.2f}s",
            f"GPU gain kernel:              {self.t_gpu_kernel:8.2f}s",
            f"GPU download (d2h):           {self.t_gpu_download:8.2f}s",
            f"mark_covered() total:         {self.t_mark_covered:8.2f}s  ({self.n_mark_calls} calls)",
            f"covered_count() total:        {self.t_covered_count:8.2f}s  ({self.n_count_calls} calls)",
            "---------------------------",
        ]
        return "\n".join(lines)


def celf_select_sparse_gpu(rules, store, cracked_size, device_id=None,
                            budget=None, batch_size=DEFAULT_GPU_CELF_BATCH,
                            hit_budget=DEFAULT_GPU_CELF_HIT_BUDGET,
                            universe_size=None,
                            vram_fraction=DEFAULT_GPU_CELF_VRAM_FRACTION,
                            force_mode=None):
    """GPU-resident counterpart to celf_select_sparse(): identical
    lazy-greedy (CELF) algorithm and identical selection order/output
    shape, but the covered set and every marginal-gain recomputation
    live on the GPU (a packed bitset + celf_gain_kernel/celf_mark_
    covered_kernel) instead of a numpy bool array + CPU popcount in
    host RAM. The only thing that stays on the host is the heap itself
    (rule index/negated-gain/stamp tuples -- already tiny, O(n_candidates)
    ints, not O(n_candidates x cracked_size)) and the stale-entry
    revalidation queue.

    Where celf_select_sparse() recomputes one candidate's gain per
    stale heap pop (one CPU numpy call each), this pops entries lazily
    but BATCHES consecutive stale pops (up to `batch_size`) into a
    single celf_gain_kernel dispatch before pushing them all back --
    the sequential nature of lazy-greedy (you don't know a pop is
    stale until you pop it) means this can't be a single big dispatch
    like the coverage-computation pass, but batching still amortizes
    kernel-launch overhead across many revalidations per round instead
    of paying it one at a time.

    Returns list[(rule, gain)] best-first, same contract as every
    other strategy's selection output.
    """
    log(f"{blue('Sparse CELF greedy select (GPU):')} "
        f"{dim('covered set + gain recompute resident on GPU, batch=' + str(batch_size))}")

    _tsetup0 = time.perf_counter()  # PROFILE patch
    if hasattr(store, 'iter_candidates_with_hits'):
        log(f"{dim('Querying coverage store for candidates with hits...')}")
        candidates = list(store.iter_candidates_with_hits())
        n_candidates = store.count_with_hits()
        log(f"{dim('Coverage store query done.')}")
    else:
        candidates = [(idx, len(c)) for idx, c in store.items() if len(c)]
        n_candidates = len(candidates)
    _tsetup1 = time.perf_counter()
    log(f"{dim(f'[PROFILE] candidate listing took {_tsetup1 - _tsetup0:.2f}s')}")

    limit = budget if budget else n_candidates
    log(f"{green('Candidates with >=1 hit:')} {cyan(f'{n_candidates:,}')} -- "
        f"{bold('budget')} {cyan(str(limit) if budget else 'unbounded (saturation)')}")

    total_hits = sum(n for _, n in candidates)
    _tsetup2 = time.perf_counter()
    log(f"{dim(f'[PROFILE] total_hits sum took {_tsetup2 - _tsetup1:.2f}s')}")

    n_rules = (max((i for i, _ in candidates), default=-1) + 1)
    uni = universe_size if universe_size is not None else cracked_size
    n_words = (uni + 31) // 32

    # --- Decide: fully GPU-resident CSR (fast, no per-round store/SQL
    # I/O) vs. streaming per-batch (works at any scale, slower). ---
    needed_bytes = (total_hits * 4) + (n_rules * 4 * 2) + (n_words * 4)
    dev_mem = _peek_device_global_mem(device_id)
    use_resident = force_mode == 'resident'
    if force_mode is None and dev_mem is not None:
        use_resident = needed_bytes <= dev_mem * vram_fraction
    budget_bytes = int(dev_mem * vram_fraction) if dev_mem else None

    if dev_mem is not None:
        _vram_msg = (
            f"[PROFILE] VRAM check: need ~{needed_bytes/1e6:,.0f} MB for a fully-resident "
            f"CSR buffer, budget ~{(budget_bytes or 0)/1e6:,.0f} MB "
            f"({vram_fraction:.0%} of {dev_mem/1e6:,.0f} MB total VRAM)"
        )
        log(dim(_vram_msg))
    else:
        log(dim("[PROFILE] Could not probe device VRAM -- defaulting to streaming backend "
                "(pass force_mode='resident' to override)."))

    if use_resident:
        log(f"{green('GPU-CELF:')} {cyan(f'{total_hits:,}')} total hit indices -- "
            f"{dim('fits VRAM budget: using fully-resident CSR backend (no store/SQL access during CELF rounds)')}")
        _tbackend0 = time.perf_counter()
        backend = _SparseCelfGpuResidentBackend(
            store, n_rules=n_rules, cracked_size=cracked_size, total_hits=total_hits,
            device_id=device_id, universe_size=universe_size)
        log(f"{dim(f'[PROFILE] resident backend init total: {time.perf_counter() - _tbackend0:.2f}s')}")
        # Resident backend has copied everything it needs out of
        # `store` into VRAM/host-side offset tables -- safe to close
        # the store now (frees SQLite connection / dict memory early).
        store.close()
    else:
        _fallback_note = (f"exceeds VRAM budget: falling back to streamed per-batch backend "
                           f"(hit budget {hit_budget:,}/dispatch)")
        log(f"{yellow('GPU-CELF:')} {cyan(f'{total_hits:,}')} total hit indices -- "
            f"{dim(_fallback_note)}")
        _tbackend0 = time.perf_counter()  # PROFILE patch
        backend = _SparseCelfGpuBackend(store, cracked_size=cracked_size,
                                         device_id=device_id, batch_size=batch_size,
                                         hit_budget=hit_budget, universe_size=universe_size)
        _init_took = time.perf_counter() - _tbackend0
        log(dim(f"[PROFILE] streaming backend init took {_init_took:.2f}s"))

    heap = [(-int(n_hits), int(idx), 0) for idx, n_hits in candidates]
    heapq.heapify(heap)

    selected = []
    n_covered = 0
    stamp = 0
    _stale_pops_total = 0        # PROFILE patch
    _PROFILE_EVERY = 25          # log a breakdown every N accepted rules

    pbar = tqdm(total=limit, desc=cyan("Sparse CELF greedy select (GPU)"), unit="rule", colour="cyan")
    while heap and len(selected) < limit:
        neg_gain, idx, s = heapq.heappop(heap)
        if s == stamp:
            gain = -neg_gain
            if gain <= 0:
                break
            selected.append((rules[idx], gain))
            backend.mark_covered(idx)
            n_covered = backend.covered_count()
            stamp += 1
            pbar.update(1)
            pbar.set_postfix({
                "recovered": n_covered,
                "rss_mb": f"{get_rss_mb():.0f}",
                "stale_pops": _stale_pops_total,
            })
            if len(selected) % _PROFILE_EVERY == 0:
                log(backend.profile_report())
            continue

        # Drain a batch of consecutive stale entries (bounded by
        # batch_size) before dispatching -- this is the GPU analogue
        # of celf_select_sparse()'s one-at-a-time CPU recompute.
        batch_idx = [idx]
        batch_entries = [(neg_gain, idx, s)]
        while heap and len(batch_idx) < batch_size and heap[0][2] != stamp:
            neg_g2, idx2, s2 = heapq.heappop(heap)
            batch_idx.append(idx2)
            batch_entries.append((neg_g2, idx2, s2))
        _stale_pops_total += len(batch_idx)  # PROFILE patch

        gains = backend.recompute_gains(np.asarray(batch_idx, dtype=np.uint32))
        for (_, cand_idx, _), gain in zip(batch_entries, gains):
            gain = int(gain)
            if gain > 0:
                heapq.heappush(heap, (-gain, cand_idx, stamp))
    pbar.close()

    log(f"{green('Done.')} {bold('Selected')} {cyan(f'{len(selected):,}')} {bold('rules,')} "
        f"{bold('covering')} {cyan(f'{n_covered:,}')}/{cyan(f'{cracked_size:,}')} "
        f"{bold('cracked-universe entries')}")
    log(f"[PROFILE] total stale heap pops requiring recompute: {_stale_pops_total:,}")
    log(backend.profile_report())
    return selected


# ============================================================
# --- Pure-CPU lazy greedy (CELF), no GPU during selection ---
# ============================================================
def celf_select_sparse(rules, store, cracked_size, budget=None, universe_size=None):
    """Lazy greedy (CELF) max-coverage selection against a sparse
    coverage store, entirely on the CPU -- the whole point of this
    strategy. `store` is a Mapping[int, np.ndarray[int32]] (an
    InMemorySparseCoverageStore/SQLiteSparseCoverageStore from
    compute_sparse_coverage_gpu(), or, for tests, any plain dict with
    the same shape) exposing iter_candidates_with_hits()/
    count_with_hits() -- if it doesn't (a raw dict, as the unit tests
    use), this falls back to deriving the same thing from .items().

    Algorithm is the standard lazy-greedy heap: seed a min-heap with
    each candidate's raw hit count (an upper bound on its true marginal
    gain, by submodularity), pop the top; if its "gain" was computed
    against the CURRENT covered set (stamp matches), it's exact and
    gets selected; otherwise recompute its true marginal gain against
    the current covered set and push it back. Ties broken by rule
    INDEX (lower index -- i.e. earlier/better-ranked in the original
    candidate order -- wins), matching celf_select()'s convention in
    ranker_postprocess.py.

    Returns list[(rule, gain)] best-first -- same shape/contract as
    every other strategy's selection output, interchangeable with
    save_output()/save_output_multi().
    """
    log(f"{blue('Sparse CELF greedy select:')} {dim('CPU-only, no GPU dispatch during selection rounds')}")

    if hasattr(store, 'iter_candidates_with_hits'):
        log(f"{dim('Querying coverage store for candidates with hits...')}")
        candidates = store.iter_candidates_with_hits()
        n_candidates = store.count_with_hits()
        log(f"{dim('Coverage store query done.')}")
    else:
        n_candidates = sum(1 for c in store.values() if len(c))
        candidates = ((idx, len(c)) for idx, c in store.items() if len(c))

    limit = budget if budget else n_candidates
    log(f"{green('Candidates with >=1 hit:')} {cyan(f'{n_candidates:,}')} -- "
        f"{bold('budget')} {cyan(str(limit) if budget else 'unbounded (saturation)')}")

    heap = [(-int(n_hits), int(idx), 0) for idx, n_hits in candidates]
    heapq.heapify(heap)

    selected = []
    # covered_mask must be sized to the largest index that can appear
    # in `store`'s hit arrays -- cracked_size itself ONLY when the
    # store holds compact 0..cracked_size-1 indices (e.g. tests that
    # build a store directly). When the store came from the GPU
    # open-addressing coverage pass (compute_sparse_coverage_gpu),
    # indices are hash-table slots and callers must pass the table
    # size as universe_size; cracked_size is still used below, as-is,
    # for %-coverage reporting against the true universe.
    bitset_size = universe_size if universe_size is not None else cracked_size
    covered_mask = np.zeros(bitset_size, dtype=bool)
    n_covered = 0
    stamp = 0

    pbar = tqdm(total=limit, desc=cyan("Sparse CELF greedy select"), unit="rule", colour="cyan")
    while heap and len(selected) < limit:
        neg_gain, idx, s = heapq.heappop(heap)
        if s == stamp:
            gain = -neg_gain
            if gain <= 0:
                break
            selected.append((rules[idx], gain))
            covered_mask[store[idx]] = True
            n_covered = int(covered_mask.sum())
            stamp += 1
            pbar.update(1)
            pbar.set_postfix({"recovered": n_covered})
            continue
        c = store[idx]
        new_gain = int(np.count_nonzero(~covered_mask[c])) if len(c) else 0
        if new_gain <= 0:
            continue
        heapq.heappush(heap, (-new_gain, idx, stamp))
    pbar.close()

    log(f"{green('Done.')} {bold('Selected')} {cyan(f'{len(selected):,}')} {bold('rules,')} "
        f"{bold('covering')} {cyan(f'{n_covered:,}')}/{cyan(f'{cracked_size:,}')} "
        f"{bold('cracked-universe entries')}")
    return selected


# ============================================================
# --- Standalone CLI (parity with celf_recompute_gpu.py's own) ---
# ============================================================
def main(argv=None):
    import argparse
    import time as _time

    ap = argparse.ArgumentParser(
        description="Sparse coverage + pure-CPU lazy-greedy CELF post-stage "
                    "(sparse per-rule hit lists in a dict/SQLite store instead "
                    "of a coverage matrix; no GPU during selection rounds).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('-r', '--ranking-csv')
    src.add_argument('-f', '--rules-file')
    ap.add_argument('-w', '--wordlist', required=True)
    ap.add_argument('-k', '--cracked', required=True)
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('-c', '--candidates', type=int, default=20000)
    ap.add_argument('-b', '--budget', type=int, default=None)
    ap.add_argument('-B', '--budgets', type=str, default=None)
    ap.add_argument('-R', '--rule-batch-size', type=int, default=1024)
    ap.add_argument('-W', '--words-batch-size', type=int, default=DEFAULT_WORDS_PER_GPU_BATCH)
    ap.add_argument('-d', '--device', type=int, default=None)
    ap.add_argument('--sparse-disk-threshold', type=int, default=SPARSE_DISK_THRESHOLD,
                     help=f"Switch from an in-memory dict to a SQLite-backed "
                          f"store above this many candidates (default "
                          f"{SPARSE_DISK_THRESHOLD:,}).")
    ap.add_argument('--sparse-store-path', type=str, default=None,
                     help="Persist the SQLite coverage store at this path "
                          "instead of a temp file that's deleted afterward "
                          "(only relevant once --candidates exceeds "
                          "--sparse-disk-threshold).")
    ap.add_argument('--gpu-celf', action='store_true',
                     help="Run the CELF greedy-select loop itself on the "
                          "GPU (packed covered-bitset + gain-recompute "
                          "kernel) instead of CPU/host RAM. Coverage "
                          "computation is already GPU either way; this "
                          "only changes the SELECTION rounds.")
    ap.add_argument('--gpu-celf-batch', type=int, default=DEFAULT_GPU_CELF_BATCH,
                     help=f"--gpu-celf only: max stale heap entries "
                          f"revalidated per GPU dispatch (default "
                          f"{DEFAULT_GPU_CELF_BATCH:,}).")
    ap.add_argument('--gpu-celf-hit-budget', type=int, default=DEFAULT_GPU_CELF_HIT_BUDGET,
                     help=f"--gpu-celf only: max combined hit-index count "
                          f"held in the per-batch GPU buffer at once "
                          f"(default {DEFAULT_GPU_CELF_HIT_BUDGET:,}, "
                          f"~{DEFAULT_GPU_CELF_HIT_BUDGET * 4 / 1e6:.0f} MB). "
                          f"Lower this on small-VRAM GPUs; raise it on "
                          f"large-VRAM GPUs for fewer, bigger dispatches.")
    ap.add_argument('--gpu-celf-vram-fraction', type=float, default=DEFAULT_GPU_CELF_VRAM_FRACTION,
                     help=f"--gpu-celf only: fraction of total device VRAM the fully-resident "
                          f"CSR backend is allowed to use before falling back to streaming "
                          f"(default {DEFAULT_GPU_CELF_VRAM_FRACTION:.0%}).")
    ap.add_argument('--gpu-celf-mode', choices=['auto', 'resident', 'streaming'], default='auto',
                     help="--gpu-celf only: 'auto' (default) picks the fully-resident CSR "
                          "backend if it fits the VRAM budget, else streams per-batch; "
                          "'resident' forces the fast all-in-VRAM path (fails if it doesn't "
                          "fit); 'streaming' forces the per-batch path regardless of size.")
    args = ap.parse_args(argv)

    t0 = _time.time()
    rules = load_candidate_rules(args)
    cracked_hashes, n_skipped = load_cracked_universe(args.cracked, MAX_WORD_LEN)
    if n_skipped:
        log(f"{yellow('Skipped')} {cyan(f'{n_skipped:,}')} {yellow('cracked entries too long.')}")
    if len(cracked_hashes) == 0:
        log(red("Cracked list is empty -- nothing to optimize for. Aborting."))
        raise SystemExit(1)

    store, initial_counts, universe_size = compute_sparse_coverage_gpu(
        rules, args.wordlist, cracked_hashes,
        rule_batch_size=args.rule_batch_size,
        words_per_gpu_batch=args.words_batch_size,
        device_id=args.device,
        disk_threshold=args.sparse_disk_threshold,
        store_path=args.sparse_store_path,
    )
    try:
        budgets = parse_budgets(args.budgets) if args.budgets else []
        run_budget = max(budgets) if budgets else args.budget
        if args.gpu_celf:
            force_mode = None if args.gpu_celf_mode == 'auto' else args.gpu_celf_mode
            selected = celf_select_sparse_gpu(
                rules, store, len(cracked_hashes), device_id=args.device,
                budget=run_budget, batch_size=args.gpu_celf_batch,
                hit_budget=args.gpu_celf_hit_budget, universe_size=universe_size,
                vram_fraction=args.gpu_celf_vram_fraction, force_mode=force_mode)
        else:
            selected = celf_select_sparse(rules, store, len(cracked_hashes),
                                           budget=run_budget, universe_size=universe_size)

        if budgets:
            save_output_multi(selected, args.output, budgets)
        else:
            save_output(selected, args.output)
    finally:
        store.close()

    print(f"\n{green('=' * 60)}")
    print(bold("CELF Post-Processing Complete (sparse)"))
    print(f"{green('=' * 60)}")
    log(f"{blue('Total time:')} {cyan(f'{_time.time() - t0:.1f}s')}")


if __name__ == '__main__':
    main()
