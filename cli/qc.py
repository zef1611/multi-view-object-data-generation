"""QC for Phase-1 correspondences against ScanNet ground-truth instance masks.

For each emitted record, look up the ScanNet `instance-filt` objectId at
`point_src` and at `point_tgt`. A correspondence is "correct" if both
pixels resolve to the same non-zero objectId.

Reports:
  - overall pixel agreement rate (% same objectId at both endpoints)
  - per-scene rate
  - per-predicted-label rate
  - count of records where ground-truth instance is missing on one side

Usage:
    python qc_correspondences.py --jsonl out/gpu_smoke.jsonl --scenes-root data/scannet/scans
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets.scannet import ScanNetAdapter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--scenes-root", type=Path, default=Path("/home/mila/l/leh/scratch/dataset/scannet_data/scans"))
    p.add_argument("--out", type=Path, default=None,
                   help="optional JSON report path")
    args = p.parse_args()

    records = [json.loads(l) for l in args.jsonl.read_text().splitlines() if l]
    if not records:
        print("No records.")
        return

    adapters: dict[str, ScanNetAdapter] = {}
    by_scene = defaultdict(list)
    for r in records:
        by_scene[r["scene_id"]].append(r)

    total = 0
    correct = 0
    missing_gt = 0
    per_scene = defaultdict(lambda: [0, 0])
    per_label = defaultdict(lambda: [0, 0])

    for sid, recs in by_scene.items():
        try:
            adapters[sid] = ScanNetAdapter(args.scenes_root / sid)
        except FileNotFoundError:
            print(f"skip {sid}: scene dir missing"); continue
        ad = adapters[sid]
        # Cache instance masks per frame.
        mask_cache: dict[str, tuple] = {}
        for r in recs:
            for fid in (r["frame_src"], r["frame_tgt"]):
                if fid not in mask_cache:
                    qc = ad.qc_instance_mask(fid)
                    mask_cache[fid] = qc
            qc_s = mask_cache[r["frame_src"]]
            qc_t = mask_cache[r["frame_tgt"]]
            if qc_s is None or qc_t is None:
                missing_gt += 1
                continue
            ms, _ = qc_s
            mt, _ = qc_t
            us, vs = r["point_src"]
            ut, vt = r["point_tgt"]
            id_s = int(ms[vs, us]) if 0 <= vs < ms.shape[0] and 0 <= us < ms.shape[1] else 0
            id_t = int(mt[vt, ut]) if 0 <= vt < mt.shape[0] and 0 <= ut < mt.shape[1] else 0
            ok = (id_s != 0 and id_s == id_t)
            total += 1
            correct += int(ok)
            per_scene[sid][0] += int(ok); per_scene[sid][1] += 1
            per_label[r["src_label"]][0] += int(ok); per_label[r["src_label"]][1] += 1

    overall = correct / total if total else 0.0
    print(f"overall: {correct}/{total} = {overall*100:.1f}% same objectId")
    print(f"missing GT (no instance mask): {missing_gt}")
    print("\nper scene:")
    for s, (c, n) in sorted(per_scene.items()):
        print(f"  {s}: {c}/{n} = {100*c/max(n,1):.1f}%")
    print("\nper predicted label:")
    for lbl, (c, n) in sorted(per_label.items(), key=lambda kv: -kv[1][1]):
        print(f"  {lbl:<15s}: {c}/{n} = {100*c/max(n,1):.1f}%")

    if args.out:
        payload = {
            "overall_correct": correct, "overall_total": total,
            "overall_rate": overall, "missing_gt": missing_gt,
            "per_scene": {s: {"correct": c, "total": n, "rate": c/max(n,1)}
                          for s, (c, n) in per_scene.items()},
            "per_label": {l: {"correct": c, "total": n, "rate": c/max(n,1)}
                          for l, (c, n) in per_label.items()},
        }
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
