# Spatial Skills — pair-selection guide

This document lists the seven spatial-reasoning skills used to route
CrossPoint-style frame pairs into training data, and what makes a frame
pair "good" for each one. It also records where each skill's gate runs
in the pipeline (pose-stage vs. content-stage) and the threshold keys in
`configs/skills/<skill>.json`.

## Pipeline context

Pair selection is a two-stage gate:

1. **Pose stage** — cheap, runs before perception. Uses only camera poses,
   frame gap, and the 5×5 corner-reprojection probe (overlap + occluded
   fraction). Handled by `pipeline/pairs.py::_assign_tasks`, with global
   floors in `configs/pair_selection.json` and per-skill knobs in
   `configs/skills/<pose-skill>.json` (the three `cross_*` skills below).
2. **Content stage** — runs after detector + segmenter + `match_pair`.
   Uses labels, masks, and cross-view matches. Handled by the per-skill
   gates in `pipeline/skills/<skill>.py` (registered in
   `pipeline/skills/__init__.py::SKILL_GATES`), configured by
   `configs/skills/<content-skill>.json`.

Both stages contribute to the same `pair.tasks` set; emit writes one
JSONL per skill under `out_root/stage_1/<skill>/correspondences.jsonl`
(see `pipeline/emit.py`).

## Skill catalog

| Skill                          | Stage       | Core question                                                               |
| ------------------------------ | ----------- | --------------------------------------------------------------------------- |
| `cross_point_correspondence`   | content     | Given a marked point in view 1, where is the same real-world spot in view 2? |
| `cross_object_correspondence`  | content     | Point to an object in view 2 that also appears in view 1 (frame-level).      |
| `relative_distance`            | content     | Which target object is farthest from a reference?                           |
| `relative_direction`           | content     | In what direction is object X from the other camera's viewpoint?            |
| `cross_occlusion_visibility`   | pose        | Is object X visible from view 2 or occluded?                                |
| `cross_depth_variation`        | pose        | Does the same object appear closer/farther across the two views?            |
| `cross_spatial_transformation` | pose        | How did the object's 2D footprint change under the viewpoint shift?         |

Context: `cross_point_correspondence` is the point-level task.
`cross_object_correspondence` is the frame-level lift: instead of asking
"find *this pixel* in the other view," it asks the model to itself
identify what is shared and point to an object that appears in both.
The other skills are downstream question formats built on top of the
same pair + match evidence.

## What makes a good frame pair per skill

### Cross Point Correspondence — "find the same spot in the other view"

- **Non-trivial viewpoint shift (AND-gated)** — rotation ≥ 20° **AND**
  translation ≥ 0.6 m (`viewpoint_shift_mode: "and"`). Both axes are
  required so rotation-dominated near-view pairs (where the answer
  collapses to "same pixel") are rejected.
- **Rotation cap** — reject rotation > 100°; beyond that the viewpoints
  are near-opposite and matching often fails.
- **Tight overlap window** — overlap ∈ [0.15, 0.30] so the task is
  never trivial appearance matching.
- **≥ 1 visible match on a labeled whole object** — require `m.visible
  and src_label non-empty and score ≥ 0.25`, where `src_label` is a
  parent-object category (chair, sofa, lamp, refrigerator). Sub-part
  labels (armrests, handles, knobs, switches) are intentionally not
  produced by the detector vocabulary — the marked point can still sit
  on a specific feature of the object, but the label stays at the whole
  object level.
- **Depth-reliable at the source point** — src-mask depth coverage ≥ 0.6
  so the reprojection ground truth is stable (not sampled from a hole in
  the depth map).
- **Point inside the src mask**, not at a seed centroid — already
  enforced by `match_pair`; worth re-asserting when picking the question
  point in phase 2.
### Cross Object Correspondence — "point to a shared object in view 2"
Frame-level lift of the point-level task. No query point is marked in
view 1; the model must identify a shared object itself and point at it
in view 2. Answer format: `{"point_2d": [u, v]}`, where `(u, v)` is any
point inside a tgt mask that corresponds to a src mask in the pair.

- **Same viewpoint-shift and overlap criteria as point correspondence**
  — rotation ≥ 20° AND translation ≥ 0.6 m, rotation ≤ 100°, overlap ∈
  [0.15, 0.30]. A shared frame-level concept is only meaningful under a
  real viewpoint change.
- **≥ 1 matched visible object** with a tgt mask whose centroid lies
  inside the tgt image bounds (pointable answer). Depth-coverage is
  *not* required on the src side, because the question is about the
  object in view 2 — the model is not asked to reason about a specific
  src pixel's 3D position.
- **Labeled tgt object** — tgt mask has a non-empty label and detector
  score ≥ 0.25. The label is used in phase-2 prompt generation
  ("point to an object visible in both views") and in the ground-truth
  answer metadata for eval.
- **Tgt mask area ≥ 0.5% of the image** — enough pixels that "point on
  the object" has room to be unambiguous. Excludes tiny far-away
  detections where even a correct answer lands close to the mask edge.
- **Avoid near-duplicate pairs** — already handled by the pair-selection
  diversity prune, but worth asserting: if view 2 is ~identical to view
  1, every visible object is trivially shared and the frame-level task
  collapses to "point to any salient object."

### Relative Distance — "farthest from the reference"
- **≥ 3 matched objects with valid 3D world points**, and at least one
  visible from both views to serve as the reference.
- **Discriminative margin** — farthest vs. runner-up ≥ 0.5 m
  (configurable). Without margin the answer is noise.
- **Reference object is unambiguous** — unique category or a category
  with exactly one matched instance. Avoid "the chair" when there are
  three.
- **Objects span meaningful depth range** — object extents should be
  small vs. the inter-object distances (otherwise "distance" depends on
  which surface point you sample).
- **Depth maps reliable** — reject pairs where a candidate object's
  mask has <60% valid depth pixels or high depth variance.
- **Avoid colinear layouts** from both cameras — if all objects are
  along both optical axes, 2D projection destroys depth ordering;
  prefer lateral spread.

### Relative Direction — "which way is X from the other viewpoint?"
- **Sufficient baseline** (≥ 1 m) so azimuth buckets
  (front/back/left/right) are stable under small pose noise.
- **Target object's bearing from the *other* camera is well-defined** —
  elevation within ±60° of horizontal so the azimuth answer is
  linguistically natural (not "directly above").
- **Discriminative azimuth spread** across candidate objects — at
  least 30° separation between candidates, otherwise distractors are
  interchangeable.
- **Scene has a consistent up-direction** (gravity-aligned).
  Matterport/ScanNet are; freeform captures may not be.
- **Target object anchored in 3D** with low localization uncertainty —
  small mask, dense depth, not on a reflective/transparent surface.
- **Non-trivial case**: prefer targets that are OUT of the querying
  camera's FOV or near the FOV boundary; if the target is plainly
  visible, the model doesn't need direction reasoning.

### Cross Occlusion Visibility
- **Same pair yields both visible and occluded matches** — single-class
  labels are useless.
- **Clear occluder geometry** — depth gap ≥ 0.15 m between predicted
  and observed tgt depth; distinct occluder category from the occluded
  object.
- **Foreground occluder is itself confidently detected** (has a mask,
  valid label).
- **Not marginal reprojections** — reject when the reprojected point
  lands within a few pixels of the image border (can't tell "occluded"
  from "out of frame").

### Cross Depth Variation
- **Axial baseline** — translation component along the viewing
  direction ≥ 0.8 m. Pure lateral motion rarely produces depth
  variation worth asking about.
- **Per-match depth delta ≥ 0.5 m** on at least one object, and
  pair-median depth delta ≥ 0.2 m (whole pair exhibits the effect, not
  a single outlier).
- **Object near but not adjacent to camera** — depth 0.5–4 m.
  Very-near objects have noisy depth; far objects have near-zero
  parallax.
- **Depth-variation direction is linguistically clean**
  (closer/farther) — avoid tangential motion where the object is "to
  the side" in both views.

### Cross Spatial Transformation
- **Large viewpoint change** — rotation ≥ 45° *or* translation ≥ 1.2 m,
  with camera-center shift ≥ 0.3 m (not pure in-place rotation).
- **Object-level evidence of the transformation** — ≥ 1 matched object
  whose src/tgt bbox scale or aspect ratio sits outside [0.6, 1.67].
  Without this, the "transformation" is scene-level but invisible on
  any specific object.
- **Overlap 0.15–0.55** — enough shared content to ground the
  question, little enough that naive appearance matching fails.
- **Rigid-body-consistent matches** — filter pairs where multiple
  matched objects' relative 3D arrangement is inconsistent (indicates a
  dynamic scene or bad poses).

## Threshold reference (`configs/skills/<skill>.json`)

Each skill has its own JSON file under `configs/skills/`; global
pair-selection floors and per-source frame-gap minima live in
`configs/pair_selection.json`. The values below are mirrored from those
files — keep this section in sync if you edit the configs.

`configs/pair_selection.json` (global, applies to every pair before any
skill gate runs):

```json
{
  "selection": {
    "pair_quality_min": 0.12, "pair_diversity_min_m": 0.50,
    "corner_overlap_min": 0.18,
    "angle_min_deg": 10.0, "angle_max_deg": 80.0,
    "max_distance_m": 5.0, "min_yaw_diff_deg": 30.0
  },
  "min_frame_gap_by_source": {
    "scannet": 40, "scannet++": 10, "matterport": 0, "unknown": 0
  }
}
```

Pose-stage skills (consumed by `pipeline/pairs.py::_assign_tasks`;
each skill may add a `min_frame_gap_bonus_by_source` on top of the
global floor):

```json
// configs/skills/cross_spatial_transformation.json
{ "angle_deg_min": 30.0,
  "min_frame_gap_bonus_by_source": {"scannet": 0, "scannet++": 0, ...} }

// configs/skills/cross_depth_variation.json
{ "median_depth_ratio_min": 1.3,
  "min_frame_gap_bonus_by_source": {"scannet": 0, "scannet++": 0, ...} }

// configs/skills/cross_occlusion_visibility.json
{ "overlap_min": 0.40, "occluded_fraction_min": 0.15,
  "min_frame_gap_bonus_by_source": {"scannet": 60, "scannet++": 20, ...} }
```

Content-stage skills (consumed by the `gate_<skill>` functions in
`pipeline/skills/<skill>.py`):

```json
// configs/skills/cross_point_correspondence.json
{ "overlap": [0.15, 0.30], "viewpoint_shift_mode": "and",
  "min_rot_deg": 20.0, "min_trans_m": 0.6, "max_rot_deg": 100.0,
  "min_visible_matches": 1, "min_label_score": 0.25,
  "mask_depth_coverage_min": 0.6 }

// configs/skills/cross_object_correspondence.json
{ "overlap": [0.15, 0.30], "viewpoint_shift_mode": "and",
  "min_rot_deg": 20.0, "min_trans_m": 0.6, "max_rot_deg": 100.0,
  "min_visible_matches": 1, "min_label_score": 0.25,
  "min_tgt_mask_area_frac": 0.005 }

// configs/skills/relative_distance.json
{ "overlap": [0.20, 0.70], "min_objects": 3, "min_margin_m": 0.5,
  "mask_depth_coverage_min": 0.6, "min_ref_visibility": "both" }

// configs/skills/relative_direction.json
{ "overlap": [0.05, 0.40],
  "min_trans_m": 1.0, "min_rot_deg": 20.0,
  "max_elev_deg": 60.0, "min_azimuth_sep_deg": 30.0,
  "bucket_hysteresis_deg": 10.0 }
```

## Where each gate lives in code

- Pose stage: `pipeline/pairs.py::_assign_tasks` (reads the three
  pose-skill JSONs via `pipeline/config.py::load_skills_config`).
- Content stage: `pipeline/skills/<skill>.py` — each file exports a
  `gate_<skill>` function, registered in
  `pipeline/skills/__init__.py::SKILL_GATES` and dispatched by
  `assign_content_skills`.
- Wiring: `pipeline/stages.py::stage_match` calls
  `assign_content_skills(...)` after `match_pair(...)` and merges the
  result into `pair.tasks` before `pipeline/emit.py` writes one JSONL
  shard per skill under `<out_root>/stage_1/<skill>/correspondences.jsonl`.
