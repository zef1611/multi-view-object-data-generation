"""Upload a stage_1 generation run to W&B as separate runs grouped together.

Creates THREE W&B runs in the same `group` so they appear linked but each
table is the centerpiece of its own run (avoids cramming all tables into
one run, which makes them small in the UI):

  <run_name>__records         BIG records table; one row per correspondence
                              with src/tgt image thumbnails. Job type = data.
  <run_name>__perception_viz  one row per (scene, src, tgt) pair with
                              GD+SAM overlays on src and tgt frames.
                              Job type = viz.
  <run_name>__pairs_viz       one row per (scene, src, tgt) pair with the
                              per-pair match-diagnostic PNG. Job type = viz.

All share `group=<run_name>` so the W&B UI lets you navigate between them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


_RECORD_COLS = [
    "task", "scene_id", "frame_src", "frame_tgt", "visible",
    "src_label", "tgt_label",
    "point_src", "point_tgt",
    "depth_src", "depth_pred_tgt", "depth_obs_tgt",
    "iou_src_to_tgt", "pair_overlap", "seed_retry",
    "X_world", "src_bbox", "tgt_bbox",
    "src_mask_id", "tgt_mask_id",
]


def _make_thumb(path: str, max_side: int):
    """Resize image so longest side = max_side. Returns PIL.Image or None."""
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        w, h = im.size
        if max(w, h) > max_side:
            s = max_side / max(w, h)
            im = im.resize((int(w * s), int(h * s)), resample=Image.BILINEAR)
        return im
    except Exception:
        return None


def _render_frame_overlay(scenes_root, scene_id: str, frame_id: str,
                          masks: list, max_side: int):
    """Draw GD bboxes + SAM masks on top of one color frame; return PIL.Image."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import hashlib
        import matplotlib.pyplot as plt
        palette = plt.get_cmap("tab20").colors
        def color_for(k):
            h = int(hashlib.sha1(k.encode()).hexdigest()[:8], 16)
            r, g, b = palette[h % len(palette)]
            return (int(r * 255), int(g * 255), int(b * 255))

        img_path = scenes_root / scene_id / "color" / f"{frame_id}.jpg"
        im = Image.open(img_path).convert("RGB")
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        import numpy as np
        for i, m in enumerate(masks):
            color = color_for(f"{m.label}|{i}")
            ys, xs = np.where(m.mask)
            if xs.size:
                # mask fill
                rgba_mask = Image.new("RGBA", im.size, (0, 0, 0, 0))
                arr = np.array(rgba_mask)
                arr[m.mask] = (*color, 100)
                rgba_mask = Image.fromarray(arr)
                overlay = Image.alpha_composite(overlay, rgba_mask)
                draw = ImageDraw.Draw(overlay)
            x0, y0, x1, y1 = m.bbox
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            label = f"{m.label} {m.score:.2f}"
            draw.rectangle([x0, max(y0 - 18, 0), x0 + 8 * len(label), y0],
                           fill=(0, 0, 0))
            draw.text((x0 + 2, max(y0 - 16, 0)), label,
                      fill=(255, 255, 255), font=font)
        out = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
        if max(out.size) > max_side:
            s = max_side / max(out.size)
            out = out.resize((int(out.width * s), int(out.height * s)),
                             resample=Image.BILINEAR)
        return out
    except Exception as e:
        logger.warning("frame overlay failed for %s/%s: %s", scene_id, frame_id, e)
        return None


def _split_pair_viz(path: str, max_side: int):
    """The per-pair diagnostic PNG is src|tgt side-by-side. Crop into the
    two halves so each can sit in its own table column. Returns
    (src_thumb, tgt_thumb) PIL.Images or (None, None) on failure."""
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        w, h = im.size
        mid = w // 2
        src = im.crop((0, 0, mid, h))
        tgt = im.crop((mid, 0, w, h))
        if max(src.size) > max_side:
            sc = max_side / max(src.size)
            src = src.resize((int(src.width * sc), int(src.height * sc)),
                             resample=Image.BILINEAR)
        if max(tgt.size) > max_side:
            sc = max_side / max(tgt.size)
            tgt = tgt.resize((int(tgt.width * sc), int(tgt.height * sc)),
                             resample=Image.BILINEAR)
        return src, tgt
    except Exception:
        return None, None


def _record_to_row(rec: dict, tasks_for_record: str) -> list:
    return [
        tasks_for_record,
        rec["scene_id"], rec["frame_src"], rec["frame_tgt"],
        rec.get("visible", True),
        rec.get("src_label", ""), rec.get("tgt_label", ""),
        json.dumps(rec["point_src"]), json.dumps(rec["point_tgt"]),
        rec["depth_src"], rec["depth_pred_tgt"], rec["depth_obs_tgt"],
        rec["iou_src_to_tgt"], rec["pair_overlap"], rec["seed_retry"],
        json.dumps(rec["X_world"]),
        json.dumps(rec["src_bbox"]), json.dumps(rec["tgt_bbox"]),
        rec["src_mask_id"], rec["tgt_mask_id"],
    ]


def _tasks_for_record(rec: dict, task_predicates: dict) -> str:
    """Return comma-separated task names a record satisfies."""
    return ",".join(t for t, p in task_predicates.items() if p(rec))


def upload_run(
    out_root: Path,
    project: str,
    run_name: Optional[str] = None,
    config: Optional[dict] = None,
    stage: str = "stage_1",
    max_table_rows: int = 20000,
    embed_images: bool = True,
    thumb_max_side: int = 2560,
    scenes_root: Optional[Path] = Path("data/scannet/scans"),
    cache_root: Optional[Path] = Path("cache/perception"),
    adapter_name: str = "scannet",
) -> None:
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; skipping W&B upload")
        return
    from .emit import TASK_PREDICATES

    out_root = Path(out_root)
    stage_dir = out_root / stage
    if not stage_dir.exists():
        logger.warning("no %s under %s; nothing to upload", stage, out_root)
        return

    base_name = run_name or out_root.name
    cfg = config or {}

    def _start(suffix: str, job_type: str):
        try:
            return wandb.init(
                project=project, name=f"{base_name}__{suffix}",
                group=base_name, config=cfg, job_type=job_type,
                reinit=True,
            )
        except (wandb.errors.AuthenticationError, wandb.errors.UsageError) as e:
            logger.warning("wandb init failed (%s). Run `wandb login` once, "
                           "or set WANDB_API_KEY. Skipping.", e)
            return None

    # =====================================================================
    # Run 1: records table (the data)
    # =====================================================================
    all_jsonl = stage_dir / "_all" / "correspondences.jsonl"
    if all_jsonl.exists() and all_jsonl.stat().st_size > 0:
        run = _start("records", "data")
        if run is not None:
            with open(all_jsonl) as f:
                recs = [json.loads(l) for l in f]
            kept = recs[:max_table_rows]

            class _Rec:
                def __init__(self, d): self.__dict__.update(d)
                def __getattr__(self, k): return self.__dict__.get(k)
            def tasks(rec):
                r = _Rec(rec)
                return ",".join(t for t, p in TASK_PREDICATES.items() if p(r))

            cols = _RECORD_COLS + (["src_image", "tgt_image"] if embed_images else [])
            tbl = wandb.Table(columns=cols)
            for rec in kept:
                row = _record_to_row(rec, tasks(rec))
                if embed_images:
                    src_thumb = _make_thumb(rec["image_src"], thumb_max_side)
                    tgt_thumb = _make_thumb(rec["image_tgt"], thumb_max_side)
                    row += [
                        wandb.Image(src_thumb,
                                    caption=f"{rec['scene_id']}/{rec['frame_src']}")
                        if src_thumb else None,
                        wandb.Image(tgt_thumb,
                                    caption=f"{rec['scene_id']}/{rec['frame_tgt']}")
                        if tgt_thumb else None,
                    ]
                tbl.add_data(*row)
            wandb.log({"records": tbl})
            logger.info("[wandb] records: %d rows of %d", len(kept), len(recs))

            # Per-task summary metrics on the records run.
            for task, pred in TASK_PREDICATES.items():
                recs_t = [r for r in recs if pred(_Rec(r))]
                wandb.summary[f"{task}/total"] = len(recs_t)
                wandb.summary[f"{task}/visible"] = sum(1 for r in recs_t if r.get("visible", True))
                wandb.summary[f"{task}/occluded"] = sum(1 for r in recs_t if not r.get("visible", True))
            wandb.summary["all/total"] = len(recs)

            # Per-task overview viz uploaded as small images on the records run.
            for sub in sorted(stage_dir.iterdir()):
                if not sub.is_dir() or sub.name in ("perception", "pairs"):
                    continue
                viz = sub / "viz.png"
                if viz.exists():
                    wandb.log({f"viz/{sub.name}": wandb.Image(str(viz))})

            # Rejection counts.
            rejections_jsonl = stage_dir / "_all" / "correspondences.rejections.jsonl"
            if rejections_jsonl.exists():
                from collections import Counter
                reasons = Counter()
                for line in open(rejections_jsonl):
                    try:
                        reasons[json.loads(line)["reason"]] += 1
                    except Exception:
                        continue
                for reason, count in reasons.items():
                    wandb.summary[f"rejections/{reason}"] = count
            run.finish()

    # =====================================================================
    # Run 2: perception viz table (1 row per PAIR — src + tgt side by side)
    # Renders GD+SAM overlays for both frames of each emitted pair from the
    # cached masks, so each pair's detections can be reviewed jointly.
    # =====================================================================
    import pickle
    perception_cache_root = (cache_root / adapter_name) if cache_root else None
    # Collect unique (scene, src, tgt) pairs from the correspondences JSONL.
    pair_keys: list[tuple[str, str, str]] = []
    if all_jsonl.exists() and all_jsonl.stat().st_size > 0:
        seen: set[tuple[str, str, str]] = set()
        with open(all_jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = (r["scene_id"], r["frame_src"], r["frame_tgt"])
                if k not in seen:
                    seen.add(k)
                    pair_keys.append(k)

    # Per-model label cache lookup. After the registry refactor, every
    # labeler writes to `cache/labels/<spec>/<adapter>/<scene>/<frame>.json`.
    # Probe every spec dir we find.
    labels_cache_root = Path("cache/labels")
    labels_available = labels_cache_root.exists() and any(labels_cache_root.iterdir())
    label_spec_dirs = (
        sorted(labels_cache_root.iterdir()) if labels_available else []
    )

    def _labels_for(scene_id: str, frame_id: str,
                    adapter_name: str = "scannet") -> str:
        if not label_spec_dirs:
            return ""
        rel = f"{adapter_name}/{scene_id}/{frame_id}.json"
        for spec_dir in label_spec_dirs:
            cp = spec_dir / rel
            if not cp.exists():
                continue
            try:
                d = json.loads(cp.read_text())
                labels = d.get("labels", [])
                return ", ".join(labels) if labels else "(none)"
            except Exception:
                continue
        return ""

    if pair_keys and perception_cache_root and perception_cache_root.exists():
        run = _start("perception_viz", "viz")
        if run is not None:
            cols = ["scene_id", "src_viz", "tgt_viz"]
            if labels_available:
                cols += ["src_labels", "tgt_labels"]
            ptbl = wandb.Table(columns=cols)
            from viz.cache_io import discover_cfg_dir
            scene_cfg_dir: dict[str, Optional[Path]] = {}
            def _cfg_dir_for(sid: str) -> Optional[Path]:
                if sid not in scene_cfg_dir:
                    scene_cfg_dir[sid] = discover_cfg_dir(
                        perception_cache_root.parent,
                        perception_cache_root.name, sid,
                    )
                return scene_cfg_dir[sid]

            mask_cache: dict[tuple[str, str], list] = {}
            def _load_masks(sid: str, fid: str):
                key = (sid, fid)
                if key in mask_cache:
                    return mask_cache[key]
                cfg_dir = _cfg_dir_for(sid)
                if cfg_dir is None:
                    mask_cache[key] = []
                    return []
                pkl = cfg_dir / f"{fid}.pkl"
                if not pkl.exists():
                    mask_cache[key] = []
                    return []
                try:
                    masks = pickle.load(open(pkl, "rb"))
                except Exception:
                    masks = []
                mask_cache[key] = masks
                return masks

            n = 0
            for sid, fsrc, ftgt in pair_keys:
                src_masks = _load_masks(sid, fsrc)
                tgt_masks = _load_masks(sid, ftgt)
                src_img = _render_frame_overlay(
                    scenes_root, sid, fsrc, src_masks, thumb_max_side
                ) if src_masks else None
                tgt_img = _render_frame_overlay(
                    scenes_root, sid, ftgt, tgt_masks, thumb_max_side
                ) if tgt_masks else None
                if src_img is None and tgt_img is None:
                    continue
                row = [
                    sid,
                    wandb.Image(src_img, caption=f"{sid}/{fsrc} ({len(src_masks)} det)")
                    if src_img else None,
                    wandb.Image(tgt_img, caption=f"{sid}/{ftgt} ({len(tgt_masks)} det)")
                    if tgt_img else None,
                ]
                if labels_available:
                    row += [
                        _labels_for(sid, fsrc),
                        _labels_for(sid, ftgt),
                    ]
                ptbl.add_data(*row)
                n += 1
            wandb.log({"perception_viz": ptbl})
            wandb.summary["pairs_viz_rows"] = n
            logger.info("[wandb] perception_viz: %d rows (per-pair src+tgt)", n)
            run.finish()

    # =====================================================================
    # Run 3: pairs viz table (1 row per pair)
    # =====================================================================
    pairs_dir = stage_dir / "pairs"
    if pairs_dir.exists() and any(pairs_dir.glob("*.png")):
        run = _start("pairs_viz", "viz")
        if run is not None:
            pair_tbl = wandb.Table(columns=["scene_id", "src_viz", "tgt_viz"])
            n = 0
            for png in sorted(pairs_dir.glob("*.png")):
                parts = png.stem.rsplit("_", 2)
                scene = parts[0] if len(parts) == 3 else png.stem
                fsrc = parts[1] if len(parts) == 3 else ""
                ftgt = parts[2] if len(parts) == 3 else ""
                src_thumb, tgt_thumb = _split_pair_viz(str(png), thumb_max_side)
                pair_tbl.add_data(
                    scene,
                    wandb.Image(src_thumb, caption=f"src {fsrc}") if src_thumb else None,
                    wandb.Image(tgt_thumb, caption=f"tgt {ftgt}") if tgt_thumb else None,
                )
                n += 1
            wandb.log({"pairs_viz": pair_tbl})
            wandb.summary["pairs"] = n
            logger.info("[wandb] pairs_viz: %d rows (split src/tgt columns)", n)
            run.finish()

    logger.info("[wandb] DONE: project=%s group=%s", project, base_name)
