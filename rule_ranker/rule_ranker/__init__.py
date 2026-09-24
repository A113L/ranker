"""
rule_ranker
===========
Modular package wrapping the three original standalone scripts:

    rule_ranker.ranker              -- ranker.py (GPU rule ranking, v5.2)
    rule_ranker.ranker_handler      -- ranker_handler.py (fast CSV/rule analysis)
    rule_ranker.ranker_postprocess  -- ranker_postprocess.py (GPU/CPU CELF coverage stage)

Each submodule is kept byte-for-byte identical in logic to the original
script -- nothing was rewritten, only relocated into a package and given
a callable `main(argv=None)` entry point so it can be invoked either as:

    python3 -m rule_ranker.ranker --wordlist ... --rules ... --cracked ...

or dispatched through the top-level `run_ranker.py` CLI:

    python3 run_ranker.py rank --wordlist ... --rules ... --cracked ...
    python3 run_ranker.py handler <ranking_csv...>
    python3 run_ranker.py postprocess --ranking-csv ... --wordlist ... --cracked ...

See run_ranker.py --help for the full subcommand list.
"""

__all__ = ["ranker", "ranker_handler", "ranker_postprocess"]
__version__ = "1.0.0"
