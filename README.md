# CrossPoint-Objects

Tools for generating, visualizing, and quality-controlling **cross-view object-level correspondences** on indoor-scene datasets (ScanNet today; Matterport / ScanNet++ stubs). Output uses the same JSONL schema as [CrossPoint-378K / CrossPoint-Bench](https://github.com/WangYipu2002/CrossPoint), so it plugs directly into CroPond-7B / Qwen2.5-VL training and evaluation.

Invocation is always `python -m cli <command>` (generation, debugging) or `python -m viz --mode <name>` (visualization). No editable install required — clone, activate the env, and run.

All hyperparams live in `configs/`: per-stage knobs in `configs/stages/<stage>.json`, per-skill gates in `configs/skills/<skill>.json`, and top-level run presets in `configs/runs/<preset>.json`. CLIs read these as defaults; flags override (precedence: **CLI > config**). See [Configuration](#configuration) below.

---

## Setup

```bash
conda activate crosspoint
pip install -r requirements.txt
```

Core deps: `transformers`, `pillow`, `matplotlib`, `numpy`, `google-generativeai`. GPU paths additionally need GroundingDINO + SAM 2.1/3 weights (auto-downloaded the first time their CLI flag is used). Set `HF_HOME` to a scratch path to avoid filling `$HOME`.

The **default detector + segmenter (`scannet-gt + gt-mask`) need no GPU and no API key** — ScanNet GT alone, ~14 s per 40-frame scene. The default VLM specs (`--quality-filter qwen3vl-235B --labeler qwen3vl-235B --verifier qwen3vl-235B-pair`) all need GPUs (tp=4); pass `--quality-filter none` (and skip `--labeler` / `--verifier`) for a fully CPU-only run, or override to `qwen3vl-8B` / `qwen3vl-8B-pair` for cheaper single-GPU runs.

Datasets expected on disk (override with `--scenes-root` etc.):

| Dataset | Default path |
|---|---|
| ScanNet scans | `/home/mila/l/leh/scratch/dataset/scannet_data/scans/<scene_id>/` |
| CrossPoint-378K | `/home/mila/l/leh/scratch/dataset/CrossPoint-378k/` |
| Infinigen `syn_5_types` | `/network/scratch/q/qian.yang/infinigen/.../syn_5_types/` |

---

## Verify your setup (5-minute smoke)

Three commands prove the env, the pipeline, and the GPU path independently:

```bash
# 1. Tests — pure Python, no datasets needed (~70s, 66 tests).
pytest tests/

# 2. CPU smoke — runs the full pipeline against ScanNet GT only, no GPU/API.
python -m cli generate --run-config configs/runs/cpu_smoke.json \
    --scene scene0093_00 --out-root outputs/smoke_cpu

# 3. GPU smoke — same scene with the default 235B labeler+filter+verifier.
#    Skip if you have no GPU.
python -m cli generate --run-config configs/runs/paper_default.json \
    --scene scene0093_00 --out-root outputs/smoke_gpu
```

After `(2)`, expect:

```
outputs/smoke_cpu/stage_1/<skill>/correspondences.pos.jsonl   # 9 skills
outputs/smoke_cpu/stage_1/<skill>/correspondences.neg.jsonl
outputs/smoke_cpu/stage_1/<skill>/pairs.jsonl
outputs/smoke_cpu/stage_1/_all/correspondences.jsonl
outputs/smoke_cpu/stage_1/_all/correspondences.rejections.jsonl
logs/<run-id>/pipeline.log
```

A non-empty `_all/correspondences.jsonl` means the pipeline ran end-to-end. A zero-line file is a real failure — check `pipeline.log` and `_all/correspondences.rejections.jsonl` (every dropped match is logged with a reason).

Visual sanity check on a CPU run:

```bash
python -m viz --mode correspondences \
    --jsonl outputs/smoke_cpu/stage_1/_all/correspondences.jsonl \
    --num 6 --save outputs/smoke_cpu/viz.png
```

---

## Running experiments

The recommended workflow is **edit a JSON, not the CLI**:

1. Pick a starting preset under `configs/runs/` (`qwen3vl_default`, `paper_default`, `qwen3vl_vote3`, `cpu_smoke`).
2. Copy it (`cp configs/runs/paper_default.json configs/runs/myrun.json`) and edit the `stage_overrides` block — model swaps, threshold tweaks, sampling changes all live there.
3. Run with `python -m cli generate --run-config configs/runs/myrun.json --all-scenes --out-root outputs/myrun`.
4. CLI flags still win over the preset, so one-off ablations don't need a new file: `--detector scannet-gt --segmenter gt-mask --quality-filter none` overrides any preset to a CPU-only run.

Common axes you'll want to ablate:

| Axis | Where to change | Typical values |
|---|---|---|
| labeler / filter / verifier model | `stage_overrides.{label,filter,verify}.model` or `--labeler` / `--quality-filter` / `--verifier` | `qwen3vl-235B` / `qwen3vl-8B` / `gemini-2.5-pro` / `none` |
| sampling strategy | `stage_overrides.{sample,pair_gate}.sampling` or `--sampling` | `stride` (default) / `adaptive` / `cosmic` |
| detector + segmenter | `stage_overrides.{perceive,match}.{detector,segmenter}` or `--detector` / `--segmenter` | `scannet-gt + gt-mask` (CPU) / `labeled-gdino + sam2.1` (paper) |
| match thresholds | `configs/stages/match.json` or `--depth-tol` / `--iou-min` | `depth_tol` 0.10–0.20, `iou_min` 0.15–0.25 |
| per-skill gates | `configs/skills/<skill>.json` (no CLI override) | see `SPATIAL_SKILLS.md` |
| labeler majority vote | `qwen3vl_vote3` preset (`n_votes=3`) | reduces label noise; ~3× labeler cost |

**Reusing a previous run's perception cache.** Caches under `cache/{labels,filter,perception,verifier}/` are keyed by registry-spec name + adapter + scene + frame, **not** by output dir. Two runs with the same models on the same scenes share every cache hit; threshold sweeps that don't change models cost only the match/emit phase. To force a recompute, bump the registry spec name in `models/registry.py` or `rm -rf cache/<ns>/<spec>/`.

---

## Layer 2 — Generate correspondences from RGB-D + pose

End-to-end: **frame sampling → per-frame detection + segmentation (cached) → geometric src→tgt matching via depth reprojection → 3D voxel-dedup → JSONL per task.**

### Quickstart (default config)

```bash
# Uses defaults from configs/stages/*.json — no preset needed.
python -m cli generate --scene scene0394_01 --out-root outputs/run

# Or pick a curated run preset:
python -m cli generate --run-config configs/runs/qwen3vl_default.json --scene scene0394_01
python -m cli generate --run-config configs/runs/cpu_smoke.json     --scene scene0394_01
```

Default knobs (loaded from `configs/stages/*.json`; CLI flags override):

| knob | source | default | what |
|---|---|---|---|
| `sampling` | `stages/sample.json` | `stride` | every Nth frame; less pose-cluster duplication than adaptive |
| `frame_stride` | `stages/sample.json` | `50` | 1:N ratio of the scene's raw frames (50 = keep 1 of every 50) |
| `limit_frames` | `stages/sample.json` | `0` | no cap on keyframes (pass any N > 0 for inspection) |
| `quality_filter` | `stages/filter.json` (key: `model`) | `qwen3vl-235B` | per-frame Qwen3-VL usable/unusable filter run on frames-in-pairs after pair-gate (registry name from `models/registry.py`, or `none` to skip) |
| `detector` | `stages/perceive.json` + `stages/match.json` | `scannet-gt` | ScanNet GT instance IDs + labels |
| `segmenter` | `stages/perceive.json` + `stages/match.json` | `gt-mask` | mesh-projected GT mask passthrough |
| `depth_tol` | `stages/match.json` | `0.15` | match-stage visible/occluded boundary (m) |
| `iou_min` | `stages/match.json` | `0.20` | match-stage minimum src→tgt mask reprojection IoU |
| `emit_occlusion_negatives` | `stages/match.json` | `true` | emit occluded matches as NEG records (`--no-emit-occlusion-negatives` to disable) |

Output JSONL has 9 per-skill subfolders, each with `correspondences.pos.jsonl`/`correspondences.neg.jsonl`/`pairs.jsonl`, plus `_all/` aggregate. The default config requires GPU only for the Qwen filter; pass `--quality-filter none` to run end-to-end on CPU.

### Detector / segmenter matrix

| `--detector` | Identity | Box source | Mask source (with `--segmenter sam2.1`) | Speed/40fr |
|---|---|---|---|---|
| **`scannet-gt`** *(default)* | GT instance ID | GT mask bbox | clean SAM (or raw GT with default `gt-mask`) | ~14–22s |
| `scannet-gt-label+gdino` | GT instance ID via mask-IoU | GDino re-grounds | clean SAM | ~50s |
| `labeled-gdino` | VLM canonical (`--labeler`) | GDino re-grounds | clean SAM | depends on labeler |
| `gdino+scannet200` | none | GDino, 96 ScanNet200 classes | clean SAM | ~22s |
| `gdino` | none | GDino DEFAULT_CLASSES | clean SAM | ~22s |
| `noop` | — | 3×3 synthetic grid | bbox-as-mask | <1s |

`gemini+gdino` is a deprecated alias for `labeled-gdino` kept for back-compat.

`--labeler` and `--quality-filter` both take a **registry name** from `models/registry.py::MODELS`:

| Registry name | Backend | Roles | Notes |
|---|---|---|---|
| **`qwen3vl-235B`** | vLLM (auto-launched in-job, tp=4, mem_util=0.9) | **default labeler + filter** | richer labels; ~7 min server load; `recommended_concurrency=16` |
| `qwen3vl-8B` | vLLM (auto-launched in-job, tp=1, mem_util=0.9) | labeler / filter (small, opt-in) | single-GPU alternative; `images_per_prompt=1`, `recommended_concurrency=8` |
| **`qwen3vl-235B-pair`** | vLLM (auto-launched in-job, tp=4, mem_util=0.9) | **default verifier** | 235B-tier verifier; same weights as the labeler spec, `images_per_prompt=2` (sends src+tgt in one prompt) |
| `qwen3vl-8B-pair` | vLLM (auto-launched in-job, tp=1, mem_util=0.9) | verifier (small, opt-in) | single-GPU alternative; `images_per_prompt=2` |
| `gemini-2.5-flash` | Gemini API (server-less) | retained, not wired by default | needs `gemini_api_key.txt` |
| `gemini-2.5-pro` | Gemini API (server-less) | retained, not wired by default | needs `gemini_api_key.txt` |

Every VLM stage runs against vLLM HTTP. Models run **sequentially** — filter → kill → labeler → kill → GDino+SAM (no LLM server up) → optional verifier (`--verifier` on `cli generate`, or `cli balance --verifier ...` / `cli verify` separately) → kill. At most one vLLM server is up at any time, so the default `--labeler qwen3vl-235B --quality-filter qwen3vl-235B --verifier qwen3vl-235B-pair` is fine in a single pipeline run. Same-spec filter+labeler stages collapse into one server lifetime. A pre-flight cache scan skips server launches when every verdict / label / verification is already cached.

**Server reuse across scenes.** `cli generate` opens **one** `launch_server` context per VLM stage that spans the entire scene list, not one per scene. With same-spec filter+labeler the whole batch run is **one** vLLM server lifetime regardless of how many scenes you pass; with the default `--verifier qwen3vl-235B-pair` it's **two**. Concretely: a 50-scene `--all-scenes` run with the default `qwen3vl-235B` filter+labeler loads the model once (~7 min) instead of 50 times. Same pattern as the matterport3d-data-gen scripts. End-to-end command (using all defaults):

```bash
python -m cli generate --all-scenes \
    --detector labeled-gdino --segmenter sam2.1 \
    --verify-concurrency 8 --vllm-concurrency 16 \
    --out-root outputs/full
```

Concurrency: each vLLM stage uses `ThreadPoolExecutor` + `as_completed` over single-image (filter / labeler) or single-pair (verifier) requests; vLLM's continuous batching does the heavy lifting server-side. Default ThreadPool size is `ModelSpec.recommended_concurrency`; override with `--vllm-concurrency` (filter/labeler) or `--verify-concurrency` (verifier). Before each fan-out, a synthetic 1×1 PNG warm-up forces the vision-encoder kernels to compile so the first burst doesn't all stall on the same JIT.

**Multi-GPU perception pre-pass (Phase 4.5).** Between the labeler stage and per-pair match+emit, `cli generate` fans out GDino+SAM perception across every visible GPU using a `multiprocessing.spawn` pool. One worker per GPU holds GDino+SAM resident; each worker batches `--perception-batch-frames` (default 4) frames per `detect_with_labels_multi` + `segment_multi_frame` call. Topology: 4 workers × 4 frames = 16 frames in flight on a 4-GPU node. Workers write the same `cache/perception/<adapter>/<scene>/<model_tag>/<frame_id>.pkl` layout the serial path uses (atomic `.pkl.tmp` + `os.replace`); Phase 5 then runs as a pure cache-read loop. Auto-skips for CPU-only configs (`scannet-gt + gt-mask`), runs below `--perception-prepass-min-frames` (default 40), and cache-complete runs. `--perception-workers 0` forces the legacy serial Phase 5. `--compile-perception` opts into `torch.compile(mode="default")` on SAM 2.1's image encoder (~30 s warmup; pays off on long runs).

**Per-run logs** land at `logs/<run-id>/` (default run-id is `<out-basename>__<timestamp>`):

```
logs/<run-id>/
    pipeline.log              # orchestrator + per-frame [filter:<model>] / [labeler:<model>] entries
    vllm_qwen3vl-235B.log     # full vLLM stdout for the 235B server (default)
    vllm_qwen3vl-235B-pair.log # full vLLM stdout for the verifier server (if --verifier set)
```

Each tagged log line carries per-image inference time (`dt=<sec>`). The same `inference_seconds` value is also persisted in the per-image cache JSON, so post-hoc analysis is just a `find cache/labels -name '*.json' -exec jq .inference_seconds {} \;`.

**Defensive behavior**:
- `parse_labels` is an O(n) bracket-counting scanner (a previous regex-based version had catastrophic backtracking that hung for >5 min on malformed responses). It also **salvages** truncated arrays — when a small-model labeler loops and runs out of `max_new_tokens` mid-array, the scanner truncates to the last complete `{...}` and synthesizes `]`, recovering whatever labels did emit.
- `LabeledGDinoDetector(max_box_frac=0.7)` drops single GDino detections covering > 70 % of the frame *before* NMS. GDino occasionally mis-grounds short generic queries (e.g. `pen .` on a cluttered desk) to a near-frame-spanning box that SAM then segments as the table. Lower the knob if you want to suppress dominant-surface detections too.

| `--segmenter` | Output mask | When to use |
|---|---|---|
| **`gt-mask`** *(default)* | raw mesh-projected GT | deterministic, no model |
| `sam2.1` | clean SAM-refined | image-aligned edges |
| `sam3` | clean SAM3 | newer SAM variant |
| `noop` | bbox-as-mask | smoke tests |

### Sampling

| `--sampling` | Mechanism |
|---|---|
| **`stride`** *(default)* | keep 1 of every `--frame-stride` raw frames (default 50, i.e. **1:50 ratio**); regular temporal spacing — fewer pose-clustered pairs killed by diversity prune |
| `adaptive` | keep frame iff pose moved ≥ `--min-translation-m` (0.20) **or** rotated ≥ `--min-rotation-deg` (15°); content-driven, but motion-event clusters can produce near-duplicates that diversity prune collapses |
| `cosmic` | stride base + COSMIC visibility-set rejection (uses GT instance masks); restricts emitted skills to object-level only |

Cosmic flags: `--cosmic-union-coverage-min 0.3`, `--cosmic-yaw-diff-min-deg 30`, `--cosmic-obj-vis-area-min 0.005`, `--cosmic-obj-vis-depth-pix-min 50`.

### Selection pipeline (frame → pair → skill)

End-to-end stages applied by `pipeline/pairs.py::select_pairs` and the per-skill content gates in `pipeline/skills/` (one file per skill). Default thresholds shown; everything is configurable via `configs/skills/<skill>.json` (per-skill gates) and `configs/pair_selection.json` (global selection floors), or CLI flags. The right-hand column maps each step to the standalone CLI that owns it (the integrated `cli generate` runs the whole sequence in one shot).

```
                                                              ┌─ standalone CLI ─┐
1. FRAME SAMPLING                              (--sampling)   │  cli sample      │
   stride   (default): keep 1 of every N raw frames (--frame-stride 50 → 1:50 ratio)
   adaptive:           keep frame iff Δpose ≥ 0.2 m OR Δrot ≥ 15° from last
   cosmic:             stride base + step 7 visibility-set gate
   [--limit-frames N caps the keyframe list; 0 (default) = no cap]
   → emits frames.json
                          │
                          ▼  e.g. 1068 raw → ~27 keyframes (stride=40)
2. ALL-PAIRS C(N, 2) + POSE PRE-FILTER                                          │  cli pair_gate   │
   • frame_gap ≥ source floor (ScanNet: 40 — see min_frame_gap_by_source)       │  (steps 2–6)     │
   • distance ≤ max_distance_m  (5.0)                                           │  → emits         │
   • angle    ∈ [angle_min_deg, angle_max_deg] = [10°, 80°]                     │  pairs.scored.   │
                          │                                                     │  jsonl           │
                          ▼                                                     │                  │
3. QUALITY GATE  (5×5 corner-grid depth reproject)                              │                  │
   • corner_overlap        ≥ 0.18                                               │                  │
   • quality_score         ≥ 0.12   (= overlap × angle_weight)                  │                  │
                          │                                                     │                  │
                          ▼                                                     │                  │
4. DIVERSITY PRUNE (greedy, quality-sorted)                                     │                  │
   Keep a pair iff its 6-D pose signature (Cx_src,Cy_src,Cz_src,                │                  │
   Cx_tgt,Cy_tgt,Cz_tgt) is ≥ pair_diversity_min_m (0.50) from every            │                  │
   already-kept pair.                                                           │                  │
                          │                                                     │                  │
                          ▼                                                     │                  │
5. PER-TASK POSE-STAGE ASSIGNMENT                                               │                  │
   Tags each pair with the cross_* pose skills it qualifies for:                │                  │
   • cross_spatial_transformation:   angle ≥ 30°                                │                  │
   • cross_depth_variation:          median_depth ratio ≥ 1.3                   │                  │
   • cross_occlusion_visibility:     overlap ≥ 0.40                             │                  │
                                     AND occluded_frac ≥ 0.15                   │                  │
                                     AND frame_gap ≥ 40 + 60                    │                  │
                          │                                                     │                  │
                          ▼                                                     │                  │
6. COSMIC GATE  (only if --sampling cosmic)                                     │                  │
   Builds GT visibility sets, drops pairs whose:                                │                  │
   • union_coverage < cosmic_union_coverage_min (0.30)                          │                  │
   • |Δyaw|         < cosmic_yaw_diff_min_deg  (30°)                            │                  │
   Restricts emitted skills to COSMIC_SKILLS (object-level).
                          │
                          ▼
7. PER-FRAME QUALITY FILTER (frames-in-pairs only)                              │  cli filter      │
   (--quality-filter qwen3vl-235B, default)
   Qwen3-VL emits usable=yes/no per frame in the surviving pair set.
   Pairs whose src or tgt frame is unusable are dropped (logged as
   qwen_filter:<reason> in correspondences.rejections.jsonl). Cached at
   cache/filter/<spec>/<adapter>/<scene>/<frame_id>.json.
   Pass --quality-filter none to bypass (CPU-only run).
                          │
                          ▼
8a. LABELER  (only when --detector labeled-gdino, --labeler qwen3vl-*)         │  cli label       │
    Per frame-in-pair: Qwen3-VL-235B/8B emits {object, canonical} pairs.
    Cached at cache/labels/<spec>/<adapter>/<scene>/<frame>.json. Read by
    the labeled-gdino detector in step 8b (cache-only, no live server).
                          │
                          ▼
8b. PERCEPTION  (detector + segmenter)                                          │  cli perceive    │
    Multi-GPU pre-pass when GPUs are visible; auto-skipped for CPU-only
    combos (scannet-gt + gt-mask, noop) which compute lazily in step 9.
    Cached at cache/perception/<adapter>/<scene>/<detector>+<segmenter>/<frame>.pkl.
                          │
                          ▼
9. GEOMETRIC MATCHING  (pipeline/match.py)                                      │                  │
   Per src mask:                                                                │  cli match       │
   • Erode mask by 5 px so seeds avoid mesh-edge / SAM-halo artifacts.          │  (steps 9–10,    │
   • Depth-reproject the seed pixel into tgt; verify in_bounds + valid          │  emits           │
     tgt depth. Visible iff |depth_pred − depth_obs| ≤ depth_tol (0.15 m).      │  pairs.jsonl +   │
   • Otherwise, if depth_obs < depth_pred (genuine occluder) AND the            │  correspondences │
     projected src mask has non-zero IoU with some tgt mask → keep as           │  .{pos,neg}.     │
     a NEG record (visible=False) when --emit-occlusion-negatives is on.        │  jsonl)          │
   • Whole-mask reprojection IoU must be ≥ iou_min (0.20) to accept POS.        │                  │
   Match-level rejects logged in _all/correspondences.rejections.jsonl:         │                  │
   out_of_bounds, low_iou, bad_depth, no_tgt_mask, no_tgt_depth,                │                  │
   occluded, occluded_no_mask_support, voxel_dup.                               │                  │
                          │                                                     │                  │
                          ▼                                                     │                  │
10. CONTENT-STAGE SKILL GATES  (pipeline/skills/<skill>.py)                     │                  │
    Per content skill: re-check overlap window, viewpoint shift,                │                  │
    min_visible_matches, label score, mask depth coverage, etc.                 │                  │
    Pose-stage skills just attach evidence (already gated in step 6).           │                  │
    Each surviving (pair, skill) → one row in pairs.jsonl, and per-record       │                  │
    rows in correspondences.{pos,neg}.jsonl.                                    │                  │

    Notable per-skill thresholds (configs/skills/<skill>.json):
    • cross_point_correspondence / cross_object_correspondence:
        overlap ∈ [0.15, 0.30],   viewpoint_shift_mode = "and",
        rot ≥ 20°  AND  trans ≥ 0.6 m,    rot ≤ 100°
    • anchor:      overlap ∈ [0.25, 0.70],  scale_ratio_excl = [0.5, 2.0]
    • counting:    overlap ∈ [0.15, 0.60],  unique_total ∈ [3, 15]
    • relative_distance:  ≥ 3 candidates, margin ≥ 0.5 m
    • relative_direction: rot ≥ 20°, trans ≥ 1.0 m, az_sep ≥ 30°
                          │
                          ▼
11. PAIR VERIFIER  (optional, --verifier qwen3vl-235B-pair)                     │  cli verify      │
    Qwen3-VL-8B (images_per_prompt=2) sees src+tgt and validates the
    skill-specific evidence. Drops false positives. Cached at
    cache/verifier/<spec>/<skill>/<scene>/<src>__<tgt>__<evsig>.json;
    emits <input>.verified.jsonl alongside the input pairs.jsonl.
```

The stage counts printed in run logs (`pairs after pose pre-filter`, `pairs after quality gate`, `pairs after diversity prune`, `per-task pair counts`, optional `pairs after cosmic gate`) correspond 1:1 to steps 2–6 above. Step 7's `filter dropped <m>/<n> pairs` lines log the post-pair-gate filter cull.

### Run a single stage

Every pipeline phase has its own subcommand, sharing the same `pipeline/stages.py` body that `cli generate` calls. The chain is **fully decoupled** — each script writes its artifact and the next reads it back via `cli/_io.py::load_inputs`, which auto-detects the input shape (`frames.json` | `pairs.scored.jsonl` | `pairs.jsonl`). All stages are idempotent: re-running with a complete cache returns instantly without launching a server.

```bash
# 1. Sample keyframes (CPU) → frames.json
python -m cli sample    --scene scene0093_00 --frame-stride 50 \
    --out outputs/scene0093_00/frames.json

# 2. Pair-gate: pose pre-filter + quality gate + diversity prune + skill
#    assignment (CPU, pure pose/frustum gating) → pairs.scored.jsonl
python -m cli pair_gate --in outputs/scene0093_00/frames.json \
    --out outputs/scene0093_00/pairs.scored.jsonl

# 3. Per-frame quality filter (GPU). Fed pairs.scored.jsonl so only
#    frames-in-pairs are filtered — no VLM calls on geometrically
#    dropped frames. Default --quality-filter is qwen3vl-235B.
python -m cli filter    --in outputs/scene0093_00/pairs.scored.jsonl \
    --quality-filter qwen3vl-235B

# 4. Labeler (GPU). Same input — labels only frames-in-pairs.
python -m cli label     --in outputs/scene0093_00/pairs.scored.jsonl \
    --labeler qwen3vl-235B

# 5. Perception (GDino + SAM, multi-GPU when available) → cache/perception/...
#    Auto-skips for CPU-only combos (scannet-gt + gt-mask, noop).
python -m cli perceive  --in outputs/scene0093_00/pairs.scored.jsonl \
    --detector labeled-gdino --segmenter sam2.1

# 6. Match + emit (CPU) → <out-root>/stage_1/<skill>/{pairs,correspondences.{pos,neg},rejections}.jsonl
python -m cli match     --in outputs/scene0093_00/pairs.scored.jsonl \
    --detector labeled-gdino --segmenter sam2.1 \
    --out-root outputs/run

# 7. Pair verifier (GPU) → cache/verifier/... + <input>.verified.jsonl
python -m cli verify    --in outputs/run/stage_1/anchor/pairs.jsonl \
    --verifier qwen3vl-235B-pair --verify-concurrency 8
```

Inputs are flexible — `cli filter` / `cli label` / `cli perceive` accept any frame-bearing artifact, so feeding a `pairs.scored.jsonl` reduces them to the union of src/tgt frames (this is the canonical wiring for the new ordering). Caches are model-tagged at `cache/{filter,labels,perception,verifier}/<spec>/...`; switching between the chain and `cli generate` reuses every cached entry. `cli match` enforces a perception-cache pre-flight (with the exact upstream `cli perceive` command in the error) for GPU-heavy combos; CPU-only combos compute lazily.

Note: `cli pair_gate` is pure pose/frustum gating — it does **not** consult the filter cache. The filter post-culls pairs in `cli generate` (Phase 3) or, in the standalone chain, the user simply doesn't feed filter-rejected frames into `cli match` (matches against an unusable frame are still cheap because GDino+SAM caches dominate the cost).

The end-to-end chain is **byte-equivalent** to `cli generate` on the same inputs (pinned by `tests/test_chain_e2e.py`). When you split `pair_gate` and `match` you can re-run pair-gate alone with a different `--pair-quality-min` without re-paying for filter/label/perception, or run the GPU-heavy perception pre-pass on a beefy node and do match/emit on a cheap CPU node.

### Common runs

```bash
# Default (stride=50 + Qwen3-VL-235B filter + scannet-gt/gt-mask)
python -m cli generate --scene scene0394_01 \
    --out-root outputs/run

# CPU-only smoke (noop+noop, no filter, no labeler) — via run preset
python -m cli generate --run-config configs/runs/cpu_smoke.json \
    --scene scene0394_01

# Paper-style preset: adaptive + labeled-gdino + sam2.1 + 235B for filter/labeler/verifier
python -m cli generate --run-config configs/runs/paper_default.json \
    --scene scene0394_01

# Same paper-style run, expressed as raw flags (no preset)
python -m cli generate --scene scene0394_01 \
    --detector labeled-gdino --segmenter sam2.1 \
    --labeler qwen3vl-235B --quality-filter qwen3vl-235B \
    --verifier qwen3vl-235B-pair --verify-concurrency 8 \
    --sampling adaptive \
    --out-root outputs/run_qwen

# CPU-only via flags (skip Qwen filter)
python -m cli generate --scene scene0394_01 \
    --quality-filter none --out-root outputs/run_nofilter

# GT identity + clean SAM masks
python -m cli generate --scene scene0394_01 \
    --detector scannet-gt --segmenter sam2.1 --out-root outputs/run_sam

# GT identity, image-aligned bboxes via GDino re-grounding
python -m cli generate --scene scene0394_01 \
    --detector scannet-gt-label+gdino --segmenter sam2.1 \
    --out-root outputs/run_gtlabel

# VLM-derived open-vocab labels (Gemini API)
python -m cli generate --scene scene0394_01 \
    --detector labeled-gdino --labeler gemini-2.5-pro \
    --segmenter sam2.1 --out-root outputs/run_gemini

# COSMIC sampling (object-level skills only)
python -m cli generate --scene scene0394_01 \
    --sampling cosmic --cosmic-union-coverage-min 0.3 --cosmic-yaw-diff-min-deg 30 \
    --out-root outputs/run_cosmic
```

For batch quota/balance after generation: `python -m cli balance --out-root outputs/run_qwen --verifier qwen3vl-235B-pair --per-scene-cap 40 --per-bucket-cap 200`.

### Debug tooling

Three scripts help inspect what each stage of the pipeline is doing:

```bash
# Per-stage pair-thumbnail dashboard for one (scene, skill).
# Renders a single PNG per scene with rows for stages 1+2 (sampling +
# qwen), 3 (pose pre-filter), 4 (quality gate), 5 (diversity prune),
# 6 (task assignment), 7+8 (perception + match), 9 (skill content gate).
python -m cli debug_pipeline \
    --scene scene0306_00 --scene scene0012_00 \
    --skill cross_point_correspondence \
    --limit-frames 40 \
    --out outputs/pipeline_debug

# Per-pair drill-down for one skill: 5-stage debug PNG (pose / perception
# / match / content gate / final emit) for every pair that the skill
# qualified on. Reads {out-root}/stage_1/<skill>/pairs.jsonl.
python -m viz --mode inspect_pair --out-root outputs/run --skill cross_point_correspondence

# Side-by-side comparison of multiple sampling strategies on the same scenes.
python -m viz --mode compare_sampling \
    --root outputs/run_adaptive:adaptive \
    --root outputs/run_stride:stride=40 \
    --root outputs/run_cosmic:cosmic \
    --skill cross_point_correspondence \
    --scenes scene0306_00 scene0012_00 \
    --out-dir outputs/sampling_compare
```

### Canonical-label pipeline

The labeler (Gemini or Qwen3-VL) emits `{"object", "canonical"}` pairs per detection. Canonicals are persisted on every layer:

- `Detection.canonical` — set by detectors that know one.
- `ObjectMask.canonical` — backfilled in `PerceptionCache.get()` via `detector.canonicalize_mask_label()` post-SAM.
- `CorrespondenceRecord.{src_canonical, tgt_canonical}` — persisted in JSONL.

Skill gates (`counting` especially) match by canonical, not by raw label string. The prompt is editable: `configs/label_prompt.txt` (or override via `configs/stages/label.json::prompt_file`). The same prompt is used by both labeler backends — only the inference server differs. **Note:** the cache key does not include a prompt hash, so editing the prompt does **not** auto-invalidate; either bump the registry spec name or `rm -rf cache/labels/<spec>/` after a prompt change. Same rule applies to any knob edit in `configs/stages/*.json` that affects perception caches.

### Output layout

```
{root}/stage_1/<skill>/correspondences.pos.jsonl       # visible matches (positive)
{root}/stage_1/<skill>/correspondences.neg.jsonl       # occluded matches (negative)
{root}/stage_1/<skill>/pairs.jsonl                     # per-pair skill manifest (Phase 2 input)
{root}/stage_1/_all/correspondences.jsonl              # every record (QC / viz)
{root}/stage_1/_all/correspondences.rejections.jsonl   # per-frame/mask rejection log
```

`<skill>` ∈ {`cross_point_correspondence`, `cross_object_correspondence`, `cross_depth_variation`, `cross_occlusion_visibility`, `cross_spatial_transformation`, `anchor`, `counting`, `relative_distance`, `relative_direction`}.

Per-frame and per-pair caches (model-tagged; see `tests/test_cache_keys.py` for the pinned layout):

```
cache/perception/<adapter>/<scene>/<detector>+<segmenter>/<frame_id>.pkl
cache/labels/<spec.name>/<adapter>/<scene>/<frame_id>.json
cache/filter/<spec.name>/<adapter>/<scene>/<frame_id>.json
cache/verifier/<spec.name>/<skill>/<scene>/<src>__<tgt>__<evsig>.json
```

Detector/segmenter thresholds and label prompts are **not** in the key — bump the registry name (or `rm -rf` the relevant subdir) when changing those. Caches are portable across hosts (no absolute image paths in the key). The verifier path's trailing `<evsig>` is a 10-char sha1 of the canonicalized evidence dict (unavoidable because evidence payloads are arbitrary nested structures); the rest of the path is human-readable. The `(image_path, adapter, scene_id, frame_id)` tuple is plumbed via `models._frame_ref.FrameRef`; every detector / labeler / filter takes a `FrameRef`.

---

## Configuration

All hyperparams live in `configs/`. Three layers:

| File / dir | Owns | Loaded by |
|---|---|---|
| `configs/pair_selection.json` | `selection` floors + `min_frame_gap_by_source` | `pipeline.config.load_skills_config()` |
| `configs/skills/<skill>.json` | per-skill gate (one file per content + pose skill, 9 total) | `pipeline.config.load_skills_config()` |
| `configs/stages/<stage>.json` | per-stage knob defaults (sampler thresholds, models, concurrency, geometric-match knobs, perception batching, viz/W&B) | `pipeline.config.load_stage_config(stage)` |
| `configs/runs/<preset>.json` | top-level preset that picks per-stage configs and applies `stage_overrides` (deep-merged) | `pipeline.config.load_run_config(path)` |

Plus the labeler prompt at `configs/label_prompt.txt` and the GDino vocab at `configs/scannet200_general_objects.txt`.

**Precedence — two tiers, no hidden defaults:**
- Per-stage CLIs (`cli sample / filter / pair_gate / label / perceive / match / verify`) accept `--config <path>` (defaults to `configs/stages/<stage>.json`). Any explicit `--<flag>` overrides the file.
- `cli generate` accepts `--run-config <preset>` instead. The preset's `stages` block selects per-stage files; `stage_overrides` deep-merges on top; CLI flags still win.

**Refusal contract.** `--config` rejects run presets (with a clear error pointing to `--run-config`); `--run-config` rejects per-stage files. No magic auto-detection.

**Editing a knob does *not* invalidate model-tagged caches** (`cache/labels/<spec>/...`, `cache/perception/<adapter>/<scene>/<detector>+<segmenter>/...`). Same rule that already applies to detector thresholds and prompt edits — rename the registry spec or `rm -rf` the relevant subdir to force a recompute.

**Bundled run presets:**
- `configs/runs/qwen3vl_default.json` — slurm runner default (adaptive + labeled-gdino + sam2.1 + 235B labeler + 235B filter, no verifier).
- `configs/runs/paper_default.json` — adds the 235B-pair verifier on top of the qwen3vl_default knobs (235B for filter, labeler, and verifier).
- `configs/runs/qwen3vl_vote3.json` — same as qwen3vl_default but with `n_votes=3` majority voting on the labeler stage; cache lands at `cache/labels/qwen3vl-235B__vote3/`.
- `configs/runs/cpu_smoke.json` — noop + noop + no filter + no labeler + no verifier; CPU-only.

To swap models in a slurm run, copy the preset and edit `stage_overrides.label.model` (or `.filter.model`, `.verify.model`); the slurm runner is a thin wrapper around `--run-config $RUN_CONFIG`.

---

## Layer 1 — Inspect CrossPoint-378K

`viz/dataset/crosspoint.py` overlays `point1`/`point2` (cross_*) or prompt-parsed points (single_*) on referenced images.

```bash
python -m viz --mode crosspoint                                      # random sample
python -m viz --mode crosspoint --type cross_correspondence --num 6
python -m viz --mode crosspoint --index 42 --save out.png
```

Supported `--type` values: `single_spatial_understanding`, `single_fine_grounding`, `cross_correspondence`, `cross_spatial_transformation`, `cross_depth_variation`, `cross_occlusion_visibility`.

A Jupyter version (`viz/notebooks/visualize_crosspoint.ipynb`) and W&B uploader (`python -m viz --mode crosspoint_wandb`) mirror the same logic.

---

## Visualization

All viz modes render **mask outlines (no bboxes)** with **centroid-anchored labels** showing `"specific → canonical"` when they differ.

```bash
python -m viz --mode perception --scene scene0000_00 --num 6      # cached perception
python -m viz --mode gt --scene scene0000_00 --num 6              # raw GT instance masks
python -m viz --mode correspondences --jsonl outputs/run/stage_1/_all/correspondences.jsonl
python -m viz --mode pairs --scene scene0000_00 --sampling adaptive --num 8
python -m viz --mode pair_match --scene scene0000_00 --src 180 --tgt 240
```

`python -m viz --help` lists every mode.

---

## Adding a new dataset

Subclass `BaseSceneAdapter` in `datasets/`, implement `list_frames` + `load_frame`, register in `cli/generate.py::make_adapter`. See `datasets/README.md` for the full contract, camera conventions (OpenCV), and optional overrides (`image_path`, `load_pose`, `reproject` for depthless datasets, `qc_instance_mask`). `tests/test_mock_adapter.py` is a working in-memory template.

`scannet-gt`, `gt-mask`, and `cosmic` paths require `qc_instance_mask` (per-frame GT instance IDs + label dict). Other detector/segmenter combos work on any RGB-D adapter.

---

## Repo map

```
.
├── cli/                    # python -m cli <command> dispatch
│   ├── generate.py                                     # end-to-end (delegates per-stage to pipeline/stages.py)
│   ├── sample.py / pair_gate.py / filter.py            # Phase 1–3
│   ├── label.py / perceive.py / match.py / verify.py   # Phase 4–6
│   ├── _frames_io.py / _io.py                          # frames.json + load_inputs (auto-detect frames | scored pairs | pair manifests)
│   └── debug_pipeline.py / balance.py / qc.py
│
├── viz/                    # python -m viz --mode <name> dispatch
│   ├── palette.py / overlays.py / cache_io.py    # shared helpers
│   ├── layer2/             # pipeline-output viz
│   │   └── correspondences.py / perception.py / pairs.py / gt.py / pair_match.py
│   ├── dataset/            # upstream/downstream dataset explorers
│   │   └── crosspoint.py / crosspoint_wandb.py / syn5.py
│   └── notebooks/visualize_crosspoint.ipynb
│
├── pipeline/               # core pipeline (was crosspoint_gen/core/)
│   ├── pairs.py            # select_pairs orchestrator + ViewPair
│   ├── pairs_io.py         # ScoredPair + read/write_scored_pairs (Phase 3 artifact)
│   ├── stages.py           # stage_filter / stage_label / stage_pair_gate / stage_perceive / stage_match / stage_verify
│   ├── perception_workers.py / match.py / project.py / geometry.py / dedup.py / rng.py
│   ├── emit.py / manifest.py / config.py / wandb_uploader.py
│   ├── cosmic.py / label_blocklist.py / label_matcher.py
│   ├── sampling/           # one file per strategy: adaptive, stride, cosmic
│   └── skills/             # one file per skill (9 of them) + base.py
│
├── models/                 # role subfolders (was crosspoint_gen/models/)
│   ├── registry.py / base.py / _vlm_base.py / _frame_ref.py / _json_salvage.py / noop.py
│   ├── labelers/           # gemini.py + qwen3vl.py
│   ├── filters/qwen.py
│   ├── verifiers/qwen_pair.py
│   ├── detectors/          # gdino.py + labeled_gdino.py
│   ├── segmenters/         # sam21.py + sam3.py + gt.py
│   └── gt/                 # base.py + scannet.py + scannet_gdino.py
│
├── datasets/               # was crosspoint_gen/adapters/
│   └── base.py + scannet.py + matterport.py + scannetpp.py
│
├── configs/                # see "Configuration" section above
│   ├── pair_selection.json
│   ├── stages/<stage>.json # 7 files: sample, filter, pair_gate, label, perceive, match, verify
│   ├── skills/<skill>.json # 9 files (one per content + pose skill)
│   ├── runs/<preset>.json  # qwen3vl_default, paper_default, cpu_smoke
│   ├── label_prompt.txt
│   └── scannet200_general_objects.txt
├── scripts/                # slurm launchers
│   ├── generate_qwen3vl.sh # end-to-end `cli generate --run-config $RUN_CONFIG`
│   ├── label_qwen3vl.sh    # label-only stage launcher
│   └── run_vlm_stage.sh    # unified resumable launcher for cli filter | label | verify
├── tests/                  # 66 tests
├── pyproject.toml          # project metadata only
└── CrossPoint/             # gitignored upstream reference
```

Adding new things is one file + one registry line — see CLAUDE.md for the full table.

---

## Testing

```bash
pytest tests/                     # 66 tests in ~70s
```

Coverage:
- `test_cache_keys.py` — pins labeler / filter / verifier cache-path layouts (`cache/<ns>/<spec>/<adapter>/<scene>/<frame>.json`, plus the verifier's `<src>__<tgt>__<evsig>.json` evidence-digest suffix) so consumers (viz, debug, downstream tooling) can rely on them.
- `test_mock_adapter.py` — Layer-2 pair selection + matching against a synthetic in-memory adapter (no datasets or models required).
- `test_pipeline_smoke.py` — end-to-end smoke run via `python -m cli generate` with `--detector noop --segmenter noop`.
- `test_chain_e2e.py` — runs the per-stage chain (`cli sample → pair_gate → match`) on the noop pipeline and asserts the final `correspondences.jsonl` is byte-equivalent to `cli generate`'s output. Pins the contract that every stage script integrates correctly with `pipeline/stages.py`.
- `test_project_roundtrip.py` — depth-reprojection invariants.
- `test_stages.py` — `pipeline/stages.py` smoke (cache-completeness probes, cache-only fast paths, fail-closed verifier, manifest collection / verified-jsonl truncation).
- `test_perception_batching.py` — Phase-4.5 multi-frame GDino + SAM batching parity vs the single-frame path, plus orchestration auto-skip rules.
- `test_pairs_io.py` — `pairs.scored.jsonl` round-trip; pins the `ViewPair.tasks` frozenset → sorted-list invariant.
- `test_io_loader.py` — `cli/_io.py::load_inputs` auto-detection across `frames.json`, `pairs.scored.jsonl`, `pairs.jsonl`, and directory inputs.
- `test_stage_pair_gate.py` — `pipeline.stages.stage_pair_gate` parity vs `select_pairs` called directly.
- `test_config_loader.py` — unified config loader: skills-split assembles correctly, `load_stage_config` rejects run presets, `load_run_config` deep-merges `stage_overrides`, `merge_cli_with_config` enforces CLI > config precedence, every `STAGE_NAMES` entry has a shipped `configs/stages/<name>.json`.
