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
from .celf_recompute_gpu import _ResidentWordlistMixin, _COMMON_KERNEL_BODY

# Default batch size for --gpu-celf's stale-entry revalidation dispatch
# (see celf_select_sparse_gpu / _SparseCelfGpuBackend below).
DEFAULT_GPU_CELF_BATCH = 4096

# Empty-hits sentinel, shared to avoid allocating a fresh empty array
# for every zero-coverage rule (there are often many in a large pool).
EMPTY_HITS = np.empty(0, dtype=np.int32)

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
def get_sparse_kernel_source(num_cracked):
    """Two kernels, built on the exact same rule-transform/hash/
    binary-search device code as celf_recompute_gpu.get_recompute_
    kernel_source() (imported via _COMMON_KERNEL_BODY, not re-derived
    here, so scoring semantics can never drift between strategies):

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
        actual matched cracked-array INDEX for every hit via
        atomic_inc on a per-rule write-position counter. One dispatch
        per rule-batch covers every rule in that batch in a single
        pass over the wordlist, the same batching shape as the count
        kernel and as celf_coverage_kernel in ranker_postprocess.py --
        this is a genuinely two-pass computation (count, then extract)
        but both passes are O(n_candidates x wordlist), the SAME one-
        time order as the bitmap strategy's single pass, not something
        that repeats per CELF round the way --recompute-gpu's rescoring
        does.
    """
    return f"""
#define MAX_WORD_LEN {MAX_WORD_LEN}
#define MAX_OUTPUT_LEN {MAX_OUTPUT_LEN}
#define MAX_RULE_LEN {MAX_RULE_LEN}
#define NUM_CRACKED {num_cracked}

{_COMMON_KERNEL_BODY}

__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void sparse_count_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned char* rules_in,
    __global const unsigned int* cracked_hashes_sorted,
    __global unsigned int* hit_counts,
    const unsigned int num_words,
    const unsigned int num_rules_in_batch,
    const unsigned int max_word_len)
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
    int idx = binary_search_cracked(cracked_hashes_sorted, NUM_CRACKED, h);
    if (idx < 0) return;

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
    __global const unsigned int* cracked_hashes_sorted,
    __global const unsigned int* rule_offsets,
    __global const unsigned int* rule_capacities,
    __global unsigned int* write_pos,
    __global unsigned int* out_buffer,
    const unsigned int num_words,
    const unsigned int num_rules_in_batch,
    const unsigned int max_word_len)
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
    int idx = binary_search_cracked(cracked_hashes_sorted, NUM_CRACKED, h);
    if (idx < 0) return;

    unsigned int pos = atomic_inc(&write_pos[rule_idx]);
    if (pos < rule_capacities[rule_idx]) {{
        out_buffer[rule_offsets[rule_idx] + pos] = (unsigned int)idx;
    }}
}}

// Single-pass alternative to count_kernel + extract_kernel: writes
// matched cracked-array indices directly, with no prior count pass,
// into a FIXED-STRIDE per-rule row of `out_buffer` (row length =
// `stride`, the caller-supplied total word count across the whole
// wordlist -- an upper bound on any one rule's hit count that can
// never be exceeded, so no count pass is needed to size anything).
// write_pos[rule_idx] is the atomic write cursor into that rule's row;
// the caller zeroes it ONCE per rule-batch, not per word-chunk, since
// this kernel is dispatched once per (rule-batch, word-chunk) pair and
// cursor positions must stay contiguous across chunks. Writes from
// atomic_inc are sequential (0, 1, 2, ...) so each row's valid hits
// always occupy the contiguous prefix [0, write_pos[rule_idx]) -- the
// same bounds-check-and-drop safety net as sparse_extract_kernel
// covers the same rare hash-collision edge case, not expected overflow
// (stride is a true upper bound by construction).
__kernel __attribute__((reqd_work_group_size({LOCAL_WORK_SIZE}, 1, 1)))
void sparse_combined_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned char* rules_in,
    __global const unsigned int* cracked_hashes_sorted,
    __global unsigned int* write_pos,
    __global unsigned int* out_buffer,
    const unsigned int num_words,
    const unsigned int num_rules_in_batch,
    const unsigned int max_word_len,
    const unsigned int stride)
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
    int idx = binary_search_cracked(cracked_hashes_sorted, NUM_CRACKED, h);
    if (idx < 0) return;

    unsigned int pos = atomic_inc(&write_pos[rule_idx]);
    if (pos < stride) {{
        out_buffer[rule_idx * stride + pos] = (unsigned int)idx;
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

        platform, device = select_device(device_id)
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)
        src = get_sparse_kernel_source(num_cracked)
        prg = cl.Program(self.context, src).build()
        self.count_kernel = prg.sparse_count_kernel
        self.extract_kernel = prg.sparse_extract_kernel
        self.combined_kernel = prg.sparse_combined_kernel

        mf = cl.mem_flags
        self.cracked_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                    hostbuf=cracked_hashes_sorted)

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
                        words_g, sub_rules_g, self.cracked_g,
                        sub_write_pos_g, sub_out_g,
                        np.uint32(num_words), np.uint32(sub_num), np.uint32(MAX_WORD_LEN),
                        np.uint32(stride))

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
                                       words_g, sub_rules_g, self.cracked_g, sub_counts_g,
                                       np.uint32(num_words), np.uint32(sub_num),
                                       np.uint32(MAX_WORD_LEN))

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
                        words_g, sub_rules_g, self.cracked_g,
                        sub_offsets_g, sub_capacities_g, sub_write_pos_g,
                        self._out_buffer_g,
                        np.uint32(num_words), np.uint32(sub_num), np.uint32(MAX_WORD_LEN))

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

    Returns (store, initial_counts) -- store is an
    InMemorySparseCoverageStore (n_rules <= disk_threshold) or
    SQLiteSparseCoverageStore (above it); initial_counts is an
    (n_rules,) int64 array of each rule's raw hit count (used only to
    log a summary here -- celf_select_sparse() reads its own seed
    counts back out of the store, not from this array, so the two
    strategies can't silently disagree).
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
    return store, initial_counts


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
                 hit_budget=DEFAULT_GPU_CELF_HIT_BUDGET):
        self.store = store
        self.cracked_size = cracked_size
        self.batch_size = batch_size
        self.hit_budget = max(1, hit_budget)

        platform, device = select_device(device_id)
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)
        prg = cl.Program(self.context, _sparse_celf_gpu_kernel_source()).build()
        self._gain_kernel = prg.celf_gain_kernel
        self._mark_kernel = prg.celf_mark_covered_kernel

        mf = cl.mem_flags
        n_words = (cracked_size + 31) // 32
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

        if total:
            flat = np.empty(total, dtype=np.uint32)
            pos = 0
            for a in arrays:
                m = len(a)
                if m:
                    flat[pos:pos + m] = a
                    pos += m
            cl.enqueue_copy(self.queue, self._hits_flat_g, flat, device_offset=0)
        cl.enqueue_copy(self.queue, self._offsets_g, offsets)
        cl.enqueue_copy(self.queue, self._lengths_g, lengths)

        self._gain_kernel(self.queue, (n,), None,
                           self._hits_flat_g, self._offsets_g, self._lengths_g,
                           self.covered_bitset_g, self._gains_g, np.uint32(n))
        out = np.empty(n, dtype=np.uint32)
        cl.enqueue_copy(self.queue, out, self._gains_g)
        return out

    def recompute_gains(self, candidate_indices):
        """Batched replacement for the CPU line
        `int(np.count_nonzero(~covered_mask[c]))` -- fetches each
        candidate's hit array from `store`, dispatches their true
        marginal gain against the CURRENT covered_bitset in as few GPU
        calls as hit_budget allows (almost always one, for the normal
        batch_size=4096-ish stale-revalidation batches), and returns
        results in the SAME order as candidate_indices."""
        idx_list = [int(i) for i in candidate_indices]
        n = len(idx_list)
        if n == 0:
            return np.empty(0, dtype=np.uint32)

        results = np.empty(n, dtype=np.uint32)
        i = 0
        while i < n:
            chunk_idx = []
            chunk_arrays = []
            chunk_hits = 0
            j = i
            while j < n:
                arr = self.store[idx_list[j]]
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
        arr = self.store[idx]
        length = len(arr)
        if length == 0:
            return
        flat = np.ascontiguousarray(arr, dtype=np.uint32)
        self._ensure_hits_buffer(length)
        cl.enqueue_copy(self.queue, self._hits_flat_g, flat, device_offset=0)
        self._mark_kernel(self.queue, (int(length),), None,
                           self._hits_flat_g, self.covered_bitset_g,
                           np.uint32(length))
        self.queue.finish()

    def covered_count(self):
        """Host-side popcount of the bitset, only for the tqdm
        progress display / final log line -- O(cracked_size/32), not
        O(n_candidates x cracked_size), and only paid once per
        selected rule, not per revalidation."""
        n_words = (self.cracked_size + 31) // 32
        buf = np.empty(max(1, n_words), dtype=np.uint32)
        cl.enqueue_copy(self.queue, buf, self.covered_bitset_g)
        return int(sum(bin(w).count('1') for w in buf.tolist()))


def celf_select_sparse_gpu(rules, store, cracked_size, device_id=None,
                            budget=None, batch_size=DEFAULT_GPU_CELF_BATCH,
                            hit_budget=DEFAULT_GPU_CELF_HIT_BUDGET):
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

    if hasattr(store, 'iter_candidates_with_hits'):
        log(f"{dim('Querying coverage store for candidates with hits...')}")
        candidates = list(store.iter_candidates_with_hits())
        n_candidates = store.count_with_hits()
        log(f"{dim('Coverage store query done.')}")
    else:
        candidates = [(idx, len(c)) for idx, c in store.items() if len(c)]
        n_candidates = len(candidates)

    limit = budget if budget else n_candidates
    log(f"{green('Candidates with >=1 hit:')} {cyan(f'{n_candidates:,}')} -- "
        f"{bold('budget')} {cyan(str(limit) if budget else 'unbounded (saturation)')}")

    total_hits = sum(n for _, n in candidates)
    log(f"{blue('GPU-CELF:')} {cyan(f'{total_hits:,}')} total hit indices across all candidates -- "
        f"{dim(f'streamed per-batch (hit budget {hit_budget:,}/dispatch), no full buffer ever built')}")

    # Streaming backend: keeps `store` open and queries it on demand,
    # batch by batch, for the whole run -- see _SparseCelfGpuBackend's
    # docstring for why this replaced the old "build one huge flat
    # buffer up front" design (total_hits can be far larger than
    # available RAM/VRAM).
    backend = _SparseCelfGpuBackend(store, cracked_size=cracked_size,
                                     device_id=device_id, batch_size=batch_size,
                                     hit_budget=hit_budget)

    heap = [(-int(n_hits), int(idx), 0) for idx, n_hits in candidates]
    heapq.heapify(heap)

    selected = []
    n_covered = 0
    stamp = 0

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
            pbar.set_postfix({"recovered": n_covered, "rss_mb": f"{get_rss_mb():.0f}"})
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

        gains = backend.recompute_gains(np.asarray(batch_idx, dtype=np.uint32))
        for (_, cand_idx, _), gain in zip(batch_entries, gains):
            gain = int(gain)
            if gain > 0:
                heapq.heappush(heap, (-gain, cand_idx, stamp))
    pbar.close()

    log(f"{green('Done.')} {bold('Selected')} {cyan(f'{len(selected):,}')} {bold('rules,')} "
        f"{bold('covering')} {cyan(f'{n_covered:,}')}/{cyan(f'{cracked_size:,}')} "
        f"{bold('cracked-universe entries')}")
    return selected


# ============================================================
# --- Pure-CPU lazy greedy (CELF), no GPU during selection ---
# ============================================================
def celf_select_sparse(rules, store, cracked_size, budget=None):
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
    covered_mask = np.zeros(cracked_size, dtype=bool)
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
    args = ap.parse_args(argv)

    t0 = _time.time()
    rules = load_candidate_rules(args)
    cracked_hashes, n_skipped = load_cracked_universe(args.cracked, MAX_WORD_LEN)
    if n_skipped:
        log(f"{yellow('Skipped')} {cyan(f'{n_skipped:,}')} {yellow('cracked entries too long.')}")
    if len(cracked_hashes) == 0:
        log(red("Cracked list is empty -- nothing to optimize for. Aborting."))
        raise SystemExit(1)

    store, initial_counts = compute_sparse_coverage_gpu(
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
            selected = celf_select_sparse_gpu(
                rules, store, len(cracked_hashes), device_id=args.device,
                budget=run_budget, batch_size=args.gpu_celf_batch,
                hit_budget=args.gpu_celf_hit_budget)
        else:
            selected = celf_select_sparse(rules, store, len(cracked_hashes), budget=run_budget)

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
