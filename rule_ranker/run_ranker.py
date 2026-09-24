#!/usr/bin/env python3
"""
run_ranker.py -- single entry point for the rule_ranker package
=================================================================
Dispatches to one of the three original tools by subcommand. Everything
after the subcommand is passed straight through to that tool's own
argparse parser unchanged, so existing flags/usage/docs for each script
keep working exactly as before.

Subcommands
-----------
  rank         -> rule_ranker.ranker            (ranker.py, GPU rule ranking v5.2)
  handler      -> rule_ranker.ranker_handler     (ranker_handler.py, fast CSV/rule analysis)
  postprocess  -> rule_ranker.ranker_postprocess (ranker_postprocess.py, GPU/CPU CELF coverage stage)

Examples
--------
    python3 run_ranker.py rank -w wordlist.txt -r rules.rule -c cracked.txt -o out.csv
    python3 run_ranker.py rank --legacy -w wordlist.txt -r rules.rule -c cracked.txt -o out.csv
    python3 run_ranker.py rank --list-devices

    python3 run_ranker.py handler -i out.csv -o summary.txt -r clean.rule -t 1000

    python3 run_ranker.py postprocess -r out.csv -w wordlist.txt -k cracked.txt \\
        -o celf_selected.rule --candidates 20000 --budget 5000

Each subcommand also supports plain '-h'/'--help' to show that tool's
own argument list:

    python3 run_ranker.py rank --help
    python3 run_ranker.py handler --help
    python3 run_ranker.py postprocess --help

Equivalently, each tool remains runnable as a module directly, without
going through this dispatcher:

    python3 -m rule_ranker.ranker ...
    python3 -m rule_ranker.ranker_handler ...
    python3 -m rule_ranker.ranker_postprocess ...
"""

import sys

SUBCOMMANDS = {
    "rank": ("rule_ranker.ranker", "GPU-accelerated Hashcat rule ranking (ranker.py)"),
    "handler": ("rule_ranker.ranker_handler", "Fast ranking-CSV / rule-file analysis (ranker_handler.py)"),
    "postprocess": ("rule_ranker.ranker_postprocess", "GPU/CPU CELF greedy coverage post-stage (ranker_postprocess.py)"),
}


def _print_top_level_usage():
    prog = "run_ranker.py"
    print(f"usage: {prog} <command> [command args...]\n")
    print("Commands:")
    width = max(len(c) for c in SUBCOMMANDS)
    for name, (_, desc) in SUBCOMMANDS.items():
        print(f"  {name.ljust(width)}  {desc}")
    print(f"\nRun '{prog} <command> --help' for that command's own options.")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)

    if not argv or argv[0] in ("-h", "--help"):
        _print_top_level_usage()
        return 0 if argv else 1

    command = argv[0]
    rest = argv[1:]

    if command not in SUBCOMMANDS:
        print(f"run_ranker.py: unknown command '{command}'\n", file=sys.stderr)
        _print_top_level_usage()
        return 1

    module_name, _ = SUBCOMMANDS[command]

    # Import lazily and only the module actually requested: ranker.py
    # depends on pyopencl (a real GPU stack), so a user just wanting
    # `handler` (which has no such dependency) shouldn't be forced
    # to have pyopencl installed to run this dispatcher at all.
    import importlib
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"run_ranker.py: could not import '{module_name}' needed for "
              f"'{command}': {e}", file=sys.stderr)
        print("Check that this command's dependencies are installed "
              "(see requirements.txt).", file=sys.stderr)
        return 1

    result = module.main(rest)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
