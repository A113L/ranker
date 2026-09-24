#!/usr/bin/env python3
"""
run_ranker.py -- single entry point for the rule_ranker package
=================================================================
Dispatches to one of the three original tools based on the first
argument (subcommand).

Usage:
    python3 run_ranker.py rank      [ranker args...]
    python3 run_ranker.py handler   [handler args...]
    python3 run_ranker.py postprocess [postprocess args...]
"""
import sys
import argparse

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="run_ranker",
        description="Dispatcher for rule_ranker tools (rank / handler / postprocess)",
    )
    parser.add_argument(
        "subcommand",
        choices=["rank", "handler", "postprocess"],
        help="Which tool to run",
    )
    # Parse only the subcommand; the rest of argv is forwarded untouched.
    args, remaining = parser.parse_known_args(argv)

    if args.subcommand == "rank":
        from rule_ranker.ranker import main as rank_main
        return rank_main(remaining)
    elif args.subcommand == "handler":
        from rule_ranker.ranker_handler import main as handler_main
        return handler_main(remaining)
    elif args.subcommand == "postprocess":
        from rule_ranker.ranker_postprocess import main as postprocess_main
        return postprocess_main(remaining)
    else:
        parser.error(f"Unknown subcommand: {args.subcommand}")

if __name__ == "__main__":
    sys.exit(main() or 0)
