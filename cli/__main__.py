"""`python -m cli <command> [args...]` dispatcher.

Commands map 1:1 to modules in this package. Each module exposes a
top-level `main()` (or runs at import via the legacy `if __name__ ==
"__main__"` block — we call it explicitly here so subcommand argv is
forwarded cleanly).
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "generate": "cli.generate",
    "debug_pipeline": "cli.debug_pipeline",
    "balance": "cli.balance",
    "qc": "cli.qc",
    # Per-stage CLIs — each runs one stage of the pipeline against
    # on-disk artifacts (frames.json / pairs.jsonl) so a single stage
    # can be debugged or rerun in isolation.
    "sample": "cli.sample",
    "filter": "cli.filter",
    "pair_gate": "cli.pair_gate",
    "label": "cli.label",
    "perceive": "cli.perceive",
    "match": "cli.match",
    "verify": "cli.verify",
    # Deprecated redirects — `compare_sampling` and `inspect_pair`
    # moved to `python -m viz --mode <name>`. The cli stubs print a
    # migration hint and exit 2. Remove after one release.
    "compare_sampling": "cli.compare_sampling",
    "inspect_pair": "cli.inspect_pair",
}


def _print_usage() -> None:
    print("usage: python -m cli <command> [args...]\n")
    print("commands:")
    for name in sorted(COMMANDS):
        print(f"  {name}")


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        _print_usage()
        return 0
    name = argv[0]
    if name not in COMMANDS:
        print(f"unknown command: {name}", file=sys.stderr)
        _print_usage()
        return 2
    # Forward remaining argv to the subcommand. argparse in the target
    # module reads sys.argv, so reshape it.
    sys.argv = [f"python -m cli {name}"] + argv[1:]
    mod = importlib.import_module(COMMANDS[name])
    if hasattr(mod, "main"):
        rv = mod.main()
        return int(rv or 0)
    # Legacy modules that run at import — already executed.
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
