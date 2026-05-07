"""Sample keyframes for one scene and write the frame manifest.

Output: ``--out`` JSON file (default: ``outputs/<run-id>/<scene>/frames.json``)
listing every FrameRef chosen by the sampler. Downstream VLM stages
(``cli/filter``, ``cli/label``) read this file instead of re-running
the adapter and sampler.

Knobs come from ``configs/stages/sample.json`` by default; pass
``--config <path>`` to use a different file. Any explicit CLI flag
overrides the file (precedence: CLI > config).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cli._frames_io import write_frames
from pipeline.config import (
    load_stage_config, merge_cli_with_config, stage_config_path,
)

logger = logging.getLogger("sample")


_KEYS = (
    "sampling", "frame_stride", "min_keyframes",
    "min_translation_m", "min_rotation_deg", "limit_frames",
    "cosmic_base_sampling", "cosmic_union_coverage_min",
    "cosmic_yaw_diff_min_deg",
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=stage_config_path("sample"),
                   help="per-stage config JSON (default: "
                        "configs/stages/sample.json). CLI flags override.")
    p.add_argument("--adapter", default="scannet")
    p.add_argument("--scenes-root", type=Path,
                   default=Path("/home/mila/l/leh/scratch/dataset/scannet_data/scans"))
    p.add_argument("--scene", required=True,
                   help="scene id (e.g. scene0093_00)")
    p.add_argument("--out", type=Path, default=None,
                   help="output frames.json path (default: "
                        "outputs/sample/<scene>/frames.json)")
    # Knob defaults are None so we can detect "user passed it" vs "use config".
    p.add_argument("--sampling", choices=["adaptive", "stride", "cosmic"],
                   default=None)
    p.add_argument("--frame-stride", type=int, default=None)
    p.add_argument("--min-keyframes", type=int, default=None)
    p.add_argument("--min-translation-m", type=float, default=None)
    p.add_argument("--min-rotation-deg", type=float, default=None)
    p.add_argument("--limit-frames", type=int, default=None)
    p.add_argument("--cosmic-base-sampling", choices=["adaptive", "stride"],
                   default=None)
    p.add_argument("--cosmic-union-coverage-min", type=float, default=None)
    p.add_argument("--cosmic-yaw-diff-min-deg", type=float, default=None)
    args = p.parse_args()

    cfg = load_stage_config("sample", args.config)
    merge_cli_with_config(args, cfg, _KEYS)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")

    from cli.generate import make_adapter
    from pipeline.pairs import sample_keyframes

    adapter = make_adapter(args.adapter, args.scenes_root / args.scene)
    sampled_fids, mode = sample_keyframes(
        adapter,
        sampling=args.sampling,
        frame_stride=args.frame_stride,
        min_keyframes=args.min_keyframes,
        min_translation_m=args.min_translation_m,
        min_rotation_deg=args.min_rotation_deg,
        limit_frames=args.limit_frames,
        cosmic_base_sampling=args.cosmic_base_sampling,
        cosmic_union_coverage_min=args.cosmic_union_coverage_min,
        cosmic_yaw_diff_min_deg=args.cosmic_yaw_diff_min_deg,
        log=False,
    )
    frames = [adapter.frame_ref(fid, args.adapter) for fid in sampled_fids]

    out = args.out or Path("outputs/sample") / args.scene / "frames.json"
    write_frames(frames, out)
    logger.info("sampled %d frames (mode=%s) → %s", len(frames), mode, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
