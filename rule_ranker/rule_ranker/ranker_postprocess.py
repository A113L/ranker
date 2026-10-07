#!/usr/bin/env python3
"""
ranker_postprocess.py -- GPU recompute + lazy-greedy CELF post-stage
=====================================================================
Run this AFTER rank_rules_exhaustive() or rank_rules_mab() (ranker_v5.2.py).

The post-processing stage now has one coverage-selection strategy only:
`recompute-gpu`.  It never materializes the old per-candidate coverage
bitmap matrix (dense, hybrid, memmap, or RAM-backed), and it no longer
ships the former sparse-coverage implementation.

For each greedy round, the GPU re-scores only the candidates that can still
beat the current best marginal gain.  A shrinking `active` bitset tracks
which cracked-password targets have not yet been covered.  The final
selection order is written to the requested `.rule` output and a matching
`_celf.csv` detail file.

Usage
-----
    python3 -m rule_ranker.ranker_postprocess \
        --ranking-csv ranker_output.csv \
        --wordlist rockyou.txt \
        --cracked cracked.txt \
        --candidates 20000 \
        --budget 5000 \
        --output celf_selected.rule

Or feed it a plain rules file instead of a ranking CSV:
    python3 -m rule_ranker.ranker_postprocess --rules-file top_optimized.rule \
        --wordlist rockyou.txt --cracked cracked.txt \
        --budget 5000 --output celf_selected.rule

Memory model
------------
No `(n_candidates x cracked_universe)` coverage matrix is allocated or
written to disk.  The recompute path keeps O(n_candidates) selection state,
plus the bitset of still-uncovered cracked targets and the resident/host
wordlist buffers used by the GPU scorer.  The trade-off is repeated GPU
scoring during lazy re-validation instead of random reads from a stored
coverage matrix.
"""

import argparse
import csv
import mmap
import os
import sys
import time

import numpy as np

# ============================================================
# --- CONSTANTS ---
# ============================================================
# Defaults -- overridable via --max-word-len/--max-rule-len/--max-output-len.
# Smaller, realistic values reduce private-memory pressure on the GPU.
MAX_WORD_LEN = 32
MAX_OUTPUT_LEN = 512
MAX_RULE_LEN = 255
LOCAL_WORK_SIZE = 256
DEFAULT_WORDS_PER_GPU_BATCH = 150000
MAX_DISPATCH_ITEMS = 32 * 1024 * 1024


# ----------------------------------------------------------------------
# Colors & helpers (same palette used by the other pipeline stages)
# ----------------------------------------------------------------------
class C:
    RED = '\033[91m'; GREEN = '\033[92m'; YELLOW = '\033[93m'
    BLUE = '\033[94m'; CYAN = '\033[96m'; MAGENTA='\033[95m'
    BOLD = '\033[1m';  DIM = '\033[2m';   END = '\033[0m'

def red(t): return f"{C.RED}{t}{C.END}"
def green(t): return f"{C.GREEN}{t}{C.END}"
def yellow(t): return f"{C.YELLOW}{t}{C.END}"
def blue(t): return f"{C.BLUE}{t}{C.END}"
def cyan(t): return f"{C.CYAN}{t}{C.END}"
def bold(t): return f"{C.BOLD}{t}{C.END}"
def dim(t): return f"{C.DIM}{t}{C.END}"


def log(msg):
    print(f"{dim('[CELF]')} {msg}", flush=True)


# ============================================================
# --- Loading helpers ---
# ============================================================
def fast_fnv1a_hash_32(data):
    hash_val = 2166136261
    for byte in data:
        hash_val = (hash_val ^ byte) * 16777619 & 0xFFFFFFFF
    return hash_val


def optimized_wordlist_iterator(wordlist_path, max_len, batch_size):
    """Memory-mapped iterator: yields (words_buffer, count) batches."""
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

    Returns ``(arr, n_skipped)`` where ``n_skipped`` counts non-empty
    lines longer than ``max_len`` that were dropped entirely.
    """
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
    """Mirror the GPU kernel's character-to-position mapping."""
    if '0' <= c <= '9':
        return ord(c) - ord('0')
    if 'A' <= c <= 'Z':
        return ord(c) - ord('A') + 10
    if 'a' <= c <= 'z':
        return ord(c) - ord('a') + 10
    return -1


def _tokenize_rule(rule_str):
    """Mirror the GPU kernel's command-length classification."""
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
    """Conservative upper bound for rule output length."""
    L = input_len
    for c, cmd_len, args in _tokenize_rule(rule_str):
        if cmd_len == 1:
            if c in ('d', 'f', 'q'):
                L = L * 2
            elif c in ('[', ']') and L > 1:
                L = L - 1
        elif cmd_len == 2:
            arg = args[0]
            n = _char_to_pos(arg)
            if c == 'D':
                if 0 <= n < L:
                    L -= 1
            elif c == 'L':
                if 0 <= n < L:
                    L -= n
            elif c == 'R':
                if 0 <= n < L:
                    L = n + 1
            elif c == "'":
                if 0 <= n < L:
                    L = n
            elif c in ('^', '$'):
                L += 1
            elif c in ('y', 'Y'):
                if n >= 0:
                    L += min(n, L)
            elif c in ('z', 'Z'):
                if n > 0:
                    L += n
            elif c == 'p':
                if n >= 0:
                    L *= n + 1
            elif c == '[':
                if 0 <= n < L:
                    L -= n
            elif c == ']':
                if 0 <= n < L:
                    L -= n
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
                    L -= end - n1
            elif c == 'i':
                if n1 >= 0:
                    L += 1
    return max(L, 0)


def estimate_worst_case_output_len(rules, max_word_len):
    """Return ``(worst_len, worst_rule)`` for a candidate pool."""
    worst_len = max_word_len
    worst_rule = None
    for r in rules:
        L = estimate_output_len(r, max_word_len)
        if L > worst_len:
            worst_len = L
            worst_rule = r
    return worst_len, worst_rule


def load_candidate_rules(args):
    """Return candidate rules, best-score first, capped at --candidates."""
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
    # Keep the postprocess rule width identical to rank's GPU rule width.
    # Overlong rules are skipped rather than silently truncated because
    # truncation can change Hashcat rule semantics.
    too_long = [r for r in rules if len(r.encode('latin-1', errors='ignore')) > MAX_RULE_LEN]
    if too_long:
        log(f"{yellow('Skipped')} {cyan(f'{len(too_long):,}')} {yellow('rules longer than')} {cyan(str(MAX_RULE_LEN))} {yellow('characters.')}")
        rules = [r for r in rules if len(r.encode('latin-1', errors='ignore')) <= MAX_RULE_LEN]
    log(f"{green('Candidate pool:')} {cyan(f'{len(rules):,}')} {bold('rules')}")
    return rules


# ============================================================
# --- Device selection (minimal; OpenCL imported lazily) ---
# ============================================================
def select_device(device_id=None):
    import pyopencl as cl

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
# --- Output ---
# ============================================================
def save_output(selected, output_path):
    """Write the selected rules to the exact requested output path."""
    if os.path.splitext(output_path)[1]:
        rule_path = output_path
    else:
        rule_path = output_path + '.rule'
    base = os.path.splitext(rule_path)[0]
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
    """'64,250,5000' -> sorted unique list of positive ints."""
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
    """Save one output pair per budget plus the exact --output path."""
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
    save_output(selected, output_path)


# ============================================================
# --- Main ---
# ============================================================
def main(argv=None):
    global MAX_WORD_LEN, MAX_RULE_LEN, MAX_OUTPUT_LEN
    ap = argparse.ArgumentParser(description="GPU recompute + lazy-greedy CELF max-coverage post-stage for ranker_v5.2")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('-r', '--ranking-csv', help="ranker_v5.2 output CSV (legacy or MAB mode)")
    src.add_argument('-f', '--rules-file', help="Plain .rule file (already ranked/optimized)")
    ap.add_argument('-w', '--wordlist', required=True)
    ap.add_argument('-k', '--cracked', required=True, help="Cracked passwords list")
    ap.add_argument('-o', '--output', required=True, help="Output .rule path")
    ap.add_argument('-c', '--candidates', type=int, default=20000,
                    help="How many top-scored rules to feed into greedy selection (default 20000).")
    ap.add_argument('-b', '--budget', type=int, default=None,
                    help="Max rules in final selection (default: run to saturation). Ignored if --budgets is given.")
    ap.add_argument('-B', '--budgets', type=str, default=None,
                    help="Comma-separated list of budget cutoffs to export from one run, e.g. '64,250,5000'.")
    ap.add_argument('-R', '--rule-batch-size', type=int, default=1024,
                    help="Candidate rules evaluated per GPU dispatch batch and used as the upper-bound scan chunk size.")
    ap.add_argument('-W', '--words-batch-size', type=int, default=DEFAULT_WORDS_PER_GPU_BATCH,
                    help="Words per GPU batch when the scorer streams the wordlist.")
    ap.add_argument('--max-word-len', type=int, default=MAX_WORD_LEN,
                    help=f"Words/cracked entries longer than this are skipped (default {MAX_WORD_LEN}).")
    ap.add_argument('--max-rule-len', type=int, default=MAX_RULE_LEN,
                    help=f"Max characters per hashcat rule considered (default {MAX_RULE_LEN}).")
    ap.add_argument('--max-output-len', type=int, default=MAX_OUTPUT_LEN,
                    help=f"Max length of a rule's output word the GPU kernel will produce (default {MAX_OUTPUT_LEN}).")
    ap.add_argument('--auto-max-output-len', action='store_true',
                    help="Estimate the worst-case rule output length before the GPU pass and use that value (+ margin).")
    ap.add_argument('--print-output-len-estimate', action='store_true',
                    help="Print the recommended --max-output-len and exit without touching the GPU.")
    ap.add_argument('-d', '--device', type=int, default=None)
    ap.add_argument('--strategy', choices=['recompute-gpu'], default='recompute-gpu',
                    help="Coverage/selection strategy. Only recompute-gpu is supported: no coverage matrix is materialized.")
    args = ap.parse_args(argv)

    MAX_WORD_LEN = args.max_word_len
    MAX_RULE_LEN = args.max_rule_len
    MAX_OUTPUT_LEN = args.max_output_len
    if MAX_OUTPUT_LEN < MAX_WORD_LEN and not args.auto_max_output_len:
        log(red(f"--max-output-len ({MAX_OUTPUT_LEN}) must be >= --max-word-len "
                f"({MAX_WORD_LEN}) -- rules that only extend words would be silently no-op'd otherwise. Aborting. "
                f"(Or pass --auto-max-output-len to size it automatically.)"))
        sys.exit(1)
    log(f"{blue('Buffer limits:')} max-word-len={cyan(MAX_WORD_LEN)} "
        f"max-rule-len={cyan(MAX_RULE_LEN)} max-output-len={cyan(MAX_OUTPUT_LEN)} "
        f"{dim('(entries/rules exceeding these limits are skipped)')}")

    t0 = time.time()
    rules = load_candidate_rules(args)

    if args.print_output_len_estimate or args.auto_max_output_len:
        worst_len, worst_rule = estimate_worst_case_output_len(rules, MAX_WORD_LEN)
        margin = max(8, worst_len // 16)
        suggested = worst_len + margin
        log(f"{blue('Static output-length estimate:')} worst case "
            f"{cyan(str(worst_len))} chars for --max-word-len={cyan(MAX_WORD_LEN)} "
            f"{dim(f'(rule: {worst_rule!r})' if worst_rule else '(no rule grows the word)')} "
            f"-- {bold('suggested --max-output-len')} {cyan(str(suggested))}")
        if args.print_output_len_estimate:
            sys.exit(0)
        MAX_OUTPUT_LEN = max(MAX_WORD_LEN, suggested)

    cracked_hashes, n_skipped = load_cracked_universe(args.cracked, MAX_WORD_LEN)
    if n_skipped:
        log(f"{yellow('Skipped')} {cyan(f'{n_skipped:,}')} {yellow('cracked entries longer than')} "
            f"{cyan(MAX_WORD_LEN)} {yellow('chars (not counted in coverage universe).')} "
            f"{dim('Raise --max-word-len if this matters for your data.')}")
    if len(cracked_hashes) == 0:
        log(red("Cracked list is empty -- nothing to optimize for. Aborting."))
        sys.exit(1)

    from .celf_recompute_gpu import celf_select_recompute_gpu

    log(f"{blue('--strategy recompute-gpu:')} using memory-light GPU recompute + lazy-greedy selection "
        f"-- {dim('no per-candidate coverage matrix is allocated')}")
    budgets = parse_budgets(args.budgets) if args.budgets else []
    run_budget = max(budgets) if budgets else args.budget

    selected = celf_select_recompute_gpu(
        rules, args.wordlist, cracked_hashes,
        rule_batch_size=args.rule_batch_size,
        words_per_gpu_batch=args.words_batch_size,
        device_id=args.device,
        budget=run_budget,
    )

    if budgets:
        save_output_multi(selected, args.output, budgets)
    else:
        save_output(selected, args.output)

    print(f"\n{green('=' * 60)}")
    print(bold("CELF Post-Processing Complete (recompute-gpu)"))
    print(f"{green('=' * 60)}")
    log(f"{blue('Total time:')} {cyan(f'{time.time() - t0:.1f}s')}")


if __name__ == '__main__':
    main()
