"""`python -m viz --mode <name> [args...]` dispatcher.

Modes map to modules under `viz/layer2/` (pipeline-output viz) and
`viz/dataset/` (upstream/downstream dataset explorers). Each module
exposes a top-level `main()` we call after rewiring sys.argv.

Three tiers of layer2 viz:
  * Artifact viz (read-only over caches/jsonl):
      correspondences, perception, gt, filter_rejections,
      compare_sampling, inspect_pair
  * Diagnostic viz (re-runs small CPU pieces of the pipeline to explain
    artifacts — never loads detector/segmenter):
      pairs, pair_match
  * Dataset viz (read-only over published datasets):
      crosspoint, crosspoint_wandb, syn5

`cli debug_pipeline` is intentionally NOT a viz mode — it loads the
detector + segmenter and re-runs Phase 5 of the pipeline.

Flag conventions across all layer2 modes (enforced via viz._args):
  --scene         scene id (repeatable for filter_rejections,
                  compare_sampling; singular elsewhere)
  --cache-root    parent of perception/, filter/, labels/, verifier/
                  (default: cache)
  --adapter       dataset adapter name (default: scannet)
  --model-tag     perception subdir override; default = most-recent
  --scenes-root   raw dataset root (single default in viz/__init__.py)
"""

from __future__ import annotations

import importlib
import sys

LAYER2 = ("correspondences", "perception", "pairs", "gt", "pair_match",
          "filter_rejections", "compare_sampling", "inspect_pair")
DATASET = ("crosspoint", "crosspoint_wandb", "syn5")

MODES: dict[str, str] = {}
MODES.update({m: f"viz.layer2.{m}" for m in LAYER2})
MODES.update({m: f"viz.dataset.{m}" for m in DATASET})


def _print_usage() -> None:
    print("usage: python -m viz --mode <name> [args...]\n")
    print("modes:")
    for name in sorted(MODES):
        print(f"  {name}")


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        _print_usage()
        return 0
    if argv[0] != "--mode" or len(argv) < 2:
        print("expected --mode <name> as the first argument",
              file=sys.stderr)
        _print_usage()
        return 2
    name = argv[1]
    if name not in MODES:
        print(f"unknown mode: {name}", file=sys.stderr)
        _print_usage()
        return 2
    sys.argv = [f"python -m viz --mode {name}"] + argv[2:]
    mod = importlib.import_module(MODES[name])
    if hasattr(mod, "main"):
        rv = mod.main()
        return int(rv or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
