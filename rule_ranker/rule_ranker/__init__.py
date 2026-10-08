"""
rule_ranker
===========
Modular package wrapping the three original standalone scripts:

    rule_ranker.ranker              -- ranker.py (GPU rule ranking, v6.0)
    rule_ranker.ranker_handler      -- ranker_handler.py (fast CSV/rule analysis)
    rule_ranker.ranker_postprocess  -- ranker_postprocess.py (GPU recompute + lazy-greedy CELF stage)

The package keeps the original CLI structure, but the ranking/CELF GPU paths
now use independent per-rule 64-bit fingerprinting and sample-based MAB; each
submodule exposes a callable `main(argv=None)` entry point so it can be invoked as:

    python3 -m rule_ranker.ranker --wordlist ... --rules ... --cracked ...

or dispatched through the top-level `run_ranker.py` CLI:

    python3 run_ranker.py rank --wordlist ... --rules ... --cracked ...
    python3 run_ranker.py handler <ranking_csv...>
    python3 run_ranker.py postprocess --ranking-csv ... --wordlist ... --cracked ...

See run_ranker.py --help for the full subcommand list.
"""

__all__ = ["ranker", "ranker_handler", "ranker_postprocess"]
__version__ = "1.1.0"
