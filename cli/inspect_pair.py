"""Deprecated location — moved to viz/layer2/inspect_pair.py.

Kept as a thin redirect so existing scripts / muscle memory print a
clear migration message instead of failing with a confusing import
error. Remove after one release.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "inspect_pair has moved to the viz dispatcher. Run:\n"
        "    python -m viz --mode inspect_pair [args...]\n"
        "Note: --cache-root now defaults to 'cache' (parent of "
        "perception/, filter/, ...).\n",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
