# CLAUDE.md

Guidance for Claude Code working in this repo. The project generates cross-view object correspondences from posed RGB-D scenes for VLM spatial-reasoning training data.

## Onboarding (read this first)

If you're picking this repo up cold:

1. **Verify the env.** `pytest tests/` (66 tests, ~70s, no GPU/dataset needed) → then `python -m cli generate --run-config configs/runs/cpu_smoke.json --scene scene0093_00 --out-root outputs/smoke_cpu`. A non-empty `outputs/smoke_cpu/stage_1/_all/correspondences.jsonl` means the full pipeline runs end-to-end on CPU.
2. **Pick a starting preset, don't write CLI flags from scratch.** `configs/runs/{cpu_smoke, qwen3vl_default, paper_default, qwen3vl_vote3}.json` cover the common configurations. Copy → edit `stage_overrides` → run with `--run-config <path>`. CLI flags still win for one-off ablations.
3. **Read the input/output contract before changing a stage.** Each stage in `pipeline/stages.py` reads / writes one of three artifact shapes (`frames.json` | `pairs.scored.jsonl` | `<out_root>/<skill>/pairs.jsonl`); they auto-detect via `cli/_io.py::load_inputs`. Round-trip invariants are pinned by `tests/test_pairs_io.py`, `tests/test_io_loader.py`, `tests/test_chain_e2e.py`. If you break one, those tests will say so.
4. **Caches are king.** Every VLM verdict and every perception result is on disk under `cache/{labels,filter,perception,verifier}/<spec>/<adapter>/<scene>/...`. Two runs with the same models on the same scenes share every cache hit — threshold sweeps that don't change models cost only the match/emit phase. To force a recompute, bump the registry spec name in `models/registry.py` or `rm -rf cache/<ns>/<spec>/`. **Editing JSON config values does not invalidate caches.**
5. **All artifacts go under `outputs/`, never `/tmp/`.** Smoke runs, scratch viz, log dirs — everything (see Conventions below).
6. **Don't reach for new abstractions.** Adding a labeler / detector / skill / sampler / dataset is one file + one registry line, see "Adding a new …" table below. The pipeline is intentionally stage-shaped — a new behavior almost always belongs inside one existing `stage_*` body, not a new orchestrator.

Common gotchas:
- The default `--sampling` is `stride` (`configs/stages/sample.json`); the `qwen3vl_default` and `paper_default` presets override it to `adaptive`. Check both before debugging "wrong" frame counts.
- `cli pair_gate` is pure pose/frustum gating — it does **not** run the quality filter. The filter only runs in Phase 3 of `cli generate` (or as the standalone `cli filter` stage).
- Verifier specs need `images_per_prompt>=2` (sends src+tgt in one prompt). Don't pair a `*-pair` spec with a single-image cache namespace.
- Same-spec filter+labeler stages **collapse** onto one vLLM server lifetime; mixing specs doubles the load time. Default config (`qwen3vl-235B` for both) loads the model once for an N-scene run.
- `cli generate` writes per-run logs to `logs/<run-id>/pipeline.log` (orchestrator + per-frame entries) and `logs/<run-id>/vllm_<spec>.log` (raw server stdout). Check both when debugging a stuck run.

## Repository scope

- **Correspondence generation pipeline** (`pipeline/` + `cli/generate.py`) — generates cross-view object correspondences from posed RGB-D scenes (ScanNet today, Matterport / ScanNet++ stubs). Output is per-skill JSONL. **All active work happens here.**
- **`CrossPoint/` subdirectory** — vendored upstream repo, **gitignored**, read-only reference.

## Invocation

```bash
# End-to-end generation pipeline (chains every stage internally; one
# vLLM server lifetime per VLM stage spans the whole batch — so a 50-scene
# run loads each model exactly once, not 50 times).
python -m cli generate --scene scene0093_00 --detector scannet-gt --segmenter gt-mask
python -m cli generate --scene s1 --scene s2 --detector labeled-gdino \
    --segmenter sam2.1 --labeler qwen3vl-235B --quality-filter qwen3vl-235B \
    --verifier qwen3vl-235B-pair  # filter+labeler collapse to one server, then verifier
python -m cli generate --help              # full flag list

# Per-stage CLIs — every Phase has its own runner, sharing helpers in
# pipeline/stages.py with `cli generate`. Each accepts `--in` (or
# `--frames` for filter/label back-compat) and routes through
# `cli/_io.py::load_inputs`, so any stage can consume the output of
# any prior stage (frames.json | pairs.scored.jsonl | pairs.jsonl).
# All are idempotent — cached entries trigger no server launch.
python -m cli sample    --scene scene0093_00 --out frames.json
python -m cli pair_gate --in frames.json --out pairs.scored.jsonl
python -m cli filter    --in pairs.scored.jsonl --quality-filter qwen3vl-235B
python -m cli label     --in pairs.scored.jsonl --labeler qwen3vl-235B
python -m cli perceive  --in pairs.scored.jsonl --detector labeled-gdino --segmenter sam2.1
python -m cli match     --in pairs.scored.jsonl --out-root outputs/<run>
python -m cli verify    --in outputs/<run>/stage_1/cross_point_correspondence/pairs.jsonl --verifier qwen3vl-235B-pair

# Phase 2 (pair_gate) is pure pose/frustum gating — no vLLM, no quality
# filter. Phase 3 (filter) runs on the union of frames-in-pairs only,
# saving VLM calls on geometrically dropped frames. `cli generate` then
# drops pairs whose endpoint frame is unusable. Standalone `cli pair_gate`
# does NOT consult the filter cache by design.

# Other CLIs (compute / pipeline ops)
python -m cli debug_pipeline ...           # per-stage tracer (re-runs Phase 5)
python -m cli balance --out-root outputs/run --verifier qwen3vl-235B-pair
python -m cli qc ...

# Visualization (single dispatcher: `python -m viz --mode <name> [args...]`)
# Two tiers — see viz/__main__.py:
#   artifact viz   read-only over caches/jsonl
#   diagnostic viz re-runs CPU-only pipeline pieces (no models loaded)
python -m viz --mode pairs --scene scene0093_00
python -m viz --mode perception --scene scene0093_00 --num 6
python -m viz --mode correspondences --jsonl outputs/.../correspondences.jsonl
python -m viz --mode filter_rejections --scene s1 --scene s2 \
    --filter-spec qwen3vl-235B --out-dir outputs/<run>/viz_filter_rejections
python -m viz --mode inspect_pair --out-root outputs/<run> --skill cross_point_correspondence
python -m viz --mode compare_sampling --root <a>:adaptive --root <b>:stride \
    --scenes scene0093_00 --out-dir outputs/<compare>
python -m viz --help                       # all modes
```

`scripts/generate_qwen3vl.sh` is the slurm runner; its body invokes `python -m cli generate`.

`scripts/run_vlm_stage.sh <filter|label|verify> [...cli args]` is the unified resumable launcher for the three VLM stages. Same script under `bash` (interactive), `sbatch` (single job), or `sbatch --array=1-N%1` (auto-resume on preemption); per-frame/per-pair caches make every rerun a resume.

## Current default pipeline

```
adaptive frame sampling
  └─► pair-gate (pose pre-filter + quality gate + diversity prune + skill tags)
       └─► per-frame quality filter (frames-in-pairs only) → drops unusable pairs
            └─► labeler (frames-in-pairs only)
                 └─► ScanNet GT instance bbox  (= "scannet-gt" detector)
                      └─► GT mask passthrough   (= "gt-mask" segmenter)
                           └─► geometric matching → 3D voxel dedup → JSONL
                                └─► optional pair verifier
```

Stage defaults (from `configs/stages/*.json`): `sampling=stride`, `detector=scannet-gt`, `segmenter=gt-mask`, `filter=qwen3vl-235B`, `labeler=qwen3vl-235B`, `verifier=qwen3vl-235B-pair`. Swap labelers/filters/verifiers via registry name (`--labeler` / `--quality-filter` / `--verifier`) — `qwen3vl-8B` / `qwen3vl-8B-pair` for cheaper single-GPU runs, `gemini-2.5-flash` / `gemini-2.5-pro` for the API path. Run presets in `configs/runs/*.json` override these per stage (e.g. `qwen3vl_default` flips sampling to `adaptive`).

Pass `--quality-filter none` (and skip `--labeler` / `--verifier`) for a fully **CPU-only** run on `scannet-gt + gt-mask` — no GPU and no API key required, ~14s per 40-frame scene.

### Detector matrix

| `--detector` | Identity source | Box source | Speed/40fr | When to use |
|---|---|---|---|---|
| **`scannet-gt`** *(default)* | GT instance ID | GT mask bbox | ~14s | fastest, deterministic, ScanNet-only |
| `scannet-gt-label+gdino` | GT instance ID via mask-IoU | GDino re-grounds | ~50s | clean image-aligned bboxes + GT identity |
| `labeled-gdino` | VLM canonical (set via `--labeler`) | GDino re-grounds | depends on labeler | rich Q&A wording, off-ScanNet portable |
| `gdino+scannet200` | none (closed vocab) | GDino, 96 classes | ~22s | label-driven recall only, no GT |
| `gdino` | none | GDino DEFAULT_CLASSES | ~22s | legacy fallback |
| `noop` | — | 3×3 synthetic grid | <1s | smoke tests |

`gemini+gdino` is a deprecated CLI alias kept permanently — maps to `labeled-gdino`.

### Model registry (labeler / quality-filter / verifier)

`--labeler`, `--quality-filter`, and `--verifier` all accept any **registry name** from `models/registry.py::MODELS`. Adding a new model = one line.

| Registry name | Backend | tp | mem_util | images_per_prompt | rec_concurrency | Roles |
|---|---|---|---|---|---|---|
| `qwen3vl-8B` | vLLM HTTP (in-job) | 1 | 0.9 | 1 | 8 | labeler / filter (small, opt-in) |
| **`qwen3vl-235B`** *(default filter + labeler)* | vLLM HTTP (in-job) | 4 | 0.9 | 1 | 16 | labeler / filter |
| `qwen3vl-8B-pair` | vLLM HTTP (in-job) | 1 | 0.9 | **2** | 8 | verifier (small, opt-in) |
| **`qwen3vl-235B-pair`** *(default verifier)* | vLLM HTTP (in-job) | 4 | 0.9 | **2** | 16 | verifier (needs 2 images per prompt) |
| `gemini-2.5-flash` | Gemini API | — | — | — | — | retained, not wired by default |
| `gemini-2.5-pro`   | Gemini API | — | — | — | — | retained, not wired by default |

> **Memory utilization caveat.** All vLLM specs target `gpu_memory_utilization=0.9` for a generous KV-cache budget. If a back-to-back kill→relaunch of the same spec on the same GPU(s) (e.g., generate's filter→labeler when not collapsed, or `balance` immediately after `generate` sharing a spec) races with vLLM's free-memory check on startup, the second launch can fail with `Free memory on device cuda:0 (X/Y GiB) on startup is less than desired GPU memory utilization (0.90, Z GiB)`. Drop the offending spec to `0.55` if you hit it; the prior incident is documented in `JOURNAL.md` 2026-04-29.

vLLM-backend specs are auto-launched by the pipeline (`models.registry.launch_server` is a context manager). `--limit-mm-per-prompt` is set per-spec from `ModelSpec.images_per_prompt`. Gemini-backend specs are server-less and kept in the tree for legacy use; the orchestrator no longer falls back to them implicitly.

**Sequential execution** is the design contract: only **one model is loaded at a time**. The end-to-end run is pair-gate (CPU) → filter → kills server → labeler → kills server → GDino+SAM (all GPUs free) → verifier (optional, via `--verifier` on `cli generate`, or as a separate `cli balance` / `cli verify` invocation) → kills server. So you can pair any filter with any labeler with any verifier without GPU contention.

**Server reuse across scenes.** Every vLLM stage in `cli generate` opens **one** `launch_server` context that spans the entire scene list — not one per scene. Phase 1 (sample) is per-scene with no GPU; Phase 2 (pair-gate) is per-scene with no server (pure pose/frustum gating); Phase 3 (filter) drains the union of frames-in-pairs against one filter server lifetime, then drops pairs whose endpoints are unusable; Phase 4 (labeler) drains the reduced frames-in-pairs against one labeler server; Phase 5 (perception+match+emit) runs per scene with no LLM server (GDino+SAM are alive throughout, lazy-loaded on first use); optional Phase 6 (verifier) runs against every emitted manifest in one server lifetime. With `filter_spec == labeler_spec`, Phases 3+4 collapse onto a **single** server lifetime — total server count for an N-scene run with the default `qwen3vl-235B` filter+labeler is **1** (or **2** with the default `--verifier qwen3vl-235B-pair`). This is the same pattern matterport3d-data-gen uses.

If `--labeler` and `--quality-filter` resolve to the same spec, the filter and labeler stages **collapse** into one server lifetime (pair-gate sits in Phase 2 with no server, so it's outside the collapse). Verifier always runs in its own server lifetime since it lives downstream of pair selection.

**Filter-as-post-cull**: filter runs *after* pair-gate, on the smaller frames-in-pairs set. After verdicts populate, `pipeline.stages.apply_filter_to_pairs` drops pairs whose src or tgt endpoint is unusable; each scene's `frames_for_pairs` is rebuilt from survivors before the labeler runs, so the labeler also sees the reduced set. Dropped pairs are logged to `correspondences.rejections.jsonl` as `qwen_filter:src:<reason>` or `qwen_filter:tgt:<reason>`. `cli pair_gate` (standalone) is pure pose/frustum gating and does **not** accept `--quality-filter` — the filter only ever runs as Phase 3.

**Pre-flight cache scan**: before launching a server, the orchestrator checks if every needed verdict / label is already cached. If so, the server launch is skipped entirely.

**Phase 4.5 — multi-GPU perception pre-pass.** Between Phase 4 (labeler) and Phase 5 (per-pair match+emit), `cli generate` can fan-out GDino+SAM perception across all visible GPUs. One worker per GPU, each holding GDino+SAM resident, processes frames in micro-batches via `LabeledGDinoDetector.detect_with_labels_multi` + `SAM21Segmenter.segment_multi_frame`. Workers write the same `cache/perception/<adapter>/<scene>/<model_tag>/<frame_id>.pkl` layout the serial path uses (atomic `.pkl.tmp` + `os.replace`); Phase 5 then runs as a pure cache-read loop.

- `--perception-workers N` — default = `torch.cuda.device_count()`; `0` disables.
- `--perception-batch-frames K` — default 4. Frames per batched forward inside a worker. Topology: 4 workers × 4 frames = 16 frames in flight on a 4-GPU node.
- `--perception-prepass-min-frames M` — default 40. Below this, skip the pre-pass (worker startup ~30–60s; not worth it for tiny runs). `0` forces.
- `--compile-perception` — opt-in `torch.compile(mode="default", dynamic=False)` on SAM 2.1's image encoder only (~30s warmup). GDino is not compiled; its variable text/image shapes cause inductor recompile thrash.

Auto-skip cases (logged): `(scannet-gt + gt-mask)` or `(noop + noop)` are CPU-only and never enter the pre-pass; cache-complete and below-threshold runs short-circuit before any spawn. The slurm runner enables the pre-pass by default via `PERCEPTION_WORKERS=$(nvidia-smi -L | wc -l)` (override with the env var).

Both labelers produce the same `[{"object","canonical"}, ...]` JSON via the prompt at `configs/label_prompt.txt`, so the cache schema is shared. Caches are **model-tagged**, no hashes in the per-frame caches:
- `cache/labels/<spec.name>/<adapter>/<scene>/<frame_id>.json` — keys: `valid, labels, canonicals, raw, items, attempts, inference_seconds`
- `cache/filter/<spec.name>/<adapter>/<scene>/<frame_id>.json` — keys: `usable, reason, raw, inference_seconds`
- `cache/verifier/<spec.name>/<skill>/<scene>/<src>__<tgt>__<evsig>.json` — keys: `usable, reason, raw, attempts, inference_seconds`. The trailing `<evsig>` is a 10-char sha1 of the canonicalized evidence dict (unavoidable because evidence payloads are arbitrary nested structures); everything before it is human-readable.

The `(image_path, adapter, scene_id, frame_id)` tuple flows through every detector / labeler / filter as a `models._frame_ref.FrameRef`. Cache paths are derived from `FrameRef.cache_subpath`; bumping prompts or detector thresholds does **not** auto-invalidate — rename the registry spec or `rm -rf cache/<ns>/<spec>/` after a behavior change.

**Per-run logs**: every invocation (`generate`, `balance`, `sample`, `filter`, `label`, `verify`) writes to `logs/<run-id>/`:
- `pipeline.log` — orchestrator + per-frame/per-pair tagged entries: `[filter:<model>] image=… usable=… dt=…`, `[labeler:<model>] image=… labels=N attempts=K dt=…`, `[verifier:<model>] <skill>/<scene>/<src>__<tgt> usable=… dt=…`. ThreadPool fan-out is via `as_completed` with a `progress N/M` line every 50 items.
- `vllm_<spec.name>.log` — full vLLM server stdout for each launched spec.
`--run-id` defaults to `<command>__<timestamp>` (e.g. `generate__…`, `verify__…`); override with `--run-id`. Logs dir is configurable with `--logs-dir`.

**Concurrency + warm-up**: every vLLM stage uses `pipeline.stages._fan_out` — `ThreadPoolExecutor` + `as_completed` over single-image (or single-pair, for the verifier) requests, with vLLM's continuous batching doing the heavy lifting server-side. Default ThreadPool size comes from `ModelSpec.recommended_concurrency` (override with `--vllm-concurrency` for filter/labeler or `--verify-concurrency` for the verifier). Before each fan-out, `_VLMBase.warmup()` fires one synthetic 1×1 PNG request to compile the vision-encoder kernels; without it the first burst all stalls on the same JIT and cascades into retry storms.

**Defensive parsing in the labeler** (`models/labelers/gemini.py::parse_labels` + shared `models/_json_salvage.py::find_json_array`):
- O(n) bracket-counting scanner instead of regex (a previous regex `\[\s*(?:\{.*?\}\s*,?\s*)*\]` had catastrophic backtracking — observed burning >5 min of CPU on a 600-char malformed response). Don't reintroduce regex for nested-structure parsing.
- **Salvage** when the outer `[` never closes (small models can loop and run out of `max_new_tokens` mid-array): scanner truncates to the last complete `{...}` at array depth 1 and synthesizes `]`. Recovers ~all labels even from a degenerate response.
- Cache-only mode (labeler/filter constructed with `endpoint=None`) accepts a recorded `valid=False` cache as "no labels available" instead of unlinking and retrying — required for the cache-only consumer in stage 3.

**Frame-area filter in `LabeledGDinoDetector.detect_with_labels`**: drop any single GDino bbox covering > `max_box_frac` (default 0.7) of the frame *before* NMS. GDino occasionally mis-grounds short generic queries (e.g. `pen .` on a cluttered desk) to a near-frame-spanning box; SAM then segments the table within. The filter catches that obvious failure mode. Knob: `LabeledGDinoDetector(max_box_frac=0.7)`. Lowering risks suppressing legitimate dominant-surface detections.

### Segmenter matrix

| `--segmenter` | Output mask | Speed | When to use |
|---|---|---|---|
| **`gt-mask`** *(default)* | raw mesh-projected GT | <1s/frame | deterministic, no model |
| `sam2.1` | clean SAM-refined | ~1.5s/frame | clean edges, image-aligned |
| `sam3` | clean SAM3 | ~2s/frame | newer SAM variant |
| `noop` | bbox-as-mask | <1s | smoke tests |

### Sampling matrix

| `--sampling` | Mechanism | When to use |
|---|---|---|
| **`stride`** *(stage default)* | every Nth frame (`--frame-stride 50`) | regular spacing; default in `configs/stages/sample.json` |
| `adaptive` | pose-thresholded keyframes (`--min-translation-m 0.20`, `--min-rotation-deg 15.0`) | paper-faithful — used by the `qwen3vl_default` / `paper_default` run presets |
| `cosmic` | stride/adaptive base + COSMIC visibility-set rejection | object-level skills only |

`cosmic` mode adds three GT-driven gates (`--cosmic-union-coverage-min`, `--cosmic-yaw-diff-min-deg`, area + depth visibility) and restricts emitted skills to `{cross_object_correspondence, relative_distance, relative_direction}`.

## Canonical-label pipeline

The labeler (Gemini or Qwen3-VL) emits per-detection `{"object", "canonical"}` pairs (e.g. `{"office chair", "chair"}`). Canonicals are persisted on every layer:

- `Detection.canonical` — set by detectors that know one (`labeled-gdino`, `scannet-gt`, `scannet-gt-label+gdino`).
- `ObjectMask.canonical` — backfilled in `PerceptionCache.get()` via `detector.canonicalize_mask_label()` after segmentation. SAM stays label-agnostic.
- `CorrespondenceRecord.{src_canonical, tgt_canonical}` — persisted in JSONL.

Skill gates and visualizations use canonical-first identity. Falls back to `label` when canonical is empty (e.g. plain `gdino`).

The labeler prompt lives in `configs/label_prompt.txt` (editable; shared by all labeler backends). The prompt is **not** in the cache key — after editing it, bump the registry spec name or `rm -rf cache/labels/<spec>/` to force a recompute.

`LabeledGDinoDetector(labeler=...)` accepts any labeler matching `LabelerProtocol` (`label` / `label_with_canonical` / `config`) — see `models/detectors/labeled_gdino.py`. Don't add new labeler-specific branches in the detector; widen the protocol instead.

## Repository layout

```
.
├── cli/                              # python -m cli <command>
│   ├── __main__.py                   # dispatcher
│   ├── generate.py                   # end-to-end pipeline (chains every stage; delegates per-stage bodies to pipeline/stages.py)
│   ├── sample.py / filter.py / pair_gate.py / label.py / perceive.py / match.py / verify.py   # per-stage runners
│   ├── _frames_io.py                 # frames.json read/write
│   ├── _io.py                        # load_inputs(path) → InputBundle (frames.json | pairs.scored.jsonl | pairs.jsonl, auto-detect)
│   ├── debug_pipeline.py / balance.py / qc.py
│   │     # inspect_pair.py + compare_sampling.py are deprecation shims;
│   │     # the real implementations live under viz/layer2/
│
├── viz/                              # python -m viz --mode <name>
│   ├── __main__.py                   # dispatcher
│   ├── _args.py                      # add_scene_args / add_cache_args / add_scenes_root_arg
│   ├── palette.py / overlays.py / cache_io.py   # shared rendering helpers
│   └── layer2/                       # pipeline-output viz (8 modes)
│       └── correspondences.py / perception.py / pairs.py / gt.py / pair_match.py
│         filter_rejections.py / compare_sampling.py / inspect_pair.py
│
├── pipeline/                         # core pipeline
│   ├── pairs.py                      # select_pairs orchestrator + ViewPair
│   ├── pairs_io.py                   # ScoredPair + read/write_scored_pairs (pairs.scored.jsonl, Phase 3 artifact)
│   ├── stages.py                     # stage_filter / stage_label / stage_pair_gate / stage_perceive / stage_match / stage_verify (shared by cli/generate + per-stage CLIs)
│   ├── match.py / project.py / geometry.py / dedup.py / rng.py
│   ├── emit.py / manifest.py / config.py / wandb_uploader.py
│   ├── cosmic.py                     # COSMIC visibility-set computation
│   ├── label_blocklist.py            # DEFAULT_LABEL_BLOCKLIST (shared by sampling + GT detectors)
│   ├── label_matcher.py              # CLIP-text paraphrase matcher
│   ├── sampling/                     # add a sampler = drop a file here
│   │   └── __init__.py + base.py + adaptive.py + stride.py + cosmic.py
│   └── skills/                       # 7 per-skill files + base.py
│       └── __init__.py (SKILL_GATES + POSE_EVIDENCE registries) + base.py
│           + cross_point_correspondence.py + cross_object_correspondence.py
│           + relative_distance.py + relative_direction.py
│           + cross_spatial_transformation.py + cross_depth_variation.py + cross_occlusion_visibility.py
│
├── models/
│   ├── registry.py                   # ModelSpec + MODELS + launch_server + cache paths
│   ├── base.py                       # Detection / ObjectMask / Detector / Segmenter ABCs
│   ├── _vlm_base.py                  # shared lifecycle + caching for VLM clients
│   ├── _json_salvage.py              # find_json_array (O(n) bracket scanner + salvage)
│   ├── noop.py                       # NoopDetector + NoopSegmenter (smoke tests)
│   ├── labelers/                     # gemini.py + qwen3vl.py (subclass _VLMBase)
│   ├── filters/qwen.py               # quality filter (subclass _VLMBase)
│   ├── verifiers/qwen_pair.py        # pair-level skill verifier (subclass _VLMBase, vLLM HTTP)
│   ├── detectors/                    # gdino.py + labeled_gdino.py (LabeledGDinoDetector)
│   ├── segmenters/                   # sam21.py + sam3.py + gt.py
│   └── gt/                           # base.py + scannet.py + scannet_gdino.py
│
├── datasets/
│   └── base.py + scannet.py + matterport.py + scannetpp.py
│
├── configs/                          # see "Configuration" section below
│   ├── pair_selection.json           # selection floors + per-source frame-gap bonuses
│   ├── stages/<stage>.json           # per-stage knob defaults (sample/filter/pair_gate/label/perceive/match/verify)
│   ├── skills/<skill>.json           # per-skill gates (one file per content + pose skill)
│   ├── runs/<preset>.json            # top-level run presets (qwen3vl_default / paper_default / qwen3vl_vote3 / cpu_smoke)
│   ├── label_prompt.txt              # labeler prompt
│   └── scannet200_general_objects.txt
├── scripts/
│   ├── generate_qwen3vl.sh           # slurm runner — thin wrapper around `cli generate --run-config`
│   ├── label_qwen3vl.sh              # slurm runner — label stage only
│   └── run_vlm_stage.sh              # unified resumable launcher for cli filter | label | verify (bash or sbatch)
├── tests/                            # pytest suite (66 tests)
└── pyproject.toml                    # metadata only — no console_scripts; use python -m
```

### Adding a new …

| Thing | Where | Edits |
|---|---|---|
| dataset adapter | `datasets/<name>.py` | subclass `BaseSceneAdapter`; register in `cli/generate.py::make_adapter` |
| labeler | `models/labelers/<name>.py` | subclass `_VLMBase`; add 1 line in `models/registry.py::MODELS` |
| quality filter | `models/filters/<name>.py` | same as labeler |
| pair verifier | `models/verifiers/<name>.py` | subclass `_VLMBase` with `cache_namespace = "verifier"`; spec must have `images_per_prompt>=2` |
| detector | `models/detectors/<name>.py` | subclass `Detector`; register in `cli/generate.py::make_detector` |
| segmenter | `models/segmenters/<name>.py` | subclass `Segmenter`; register in `cli/generate.py::make_segmenter` |
| sampling strategy | `pipeline/sampling/<name>.py` | add to `pipeline/sampling/__init__.py::SAMPLERS` |
| skill gate | `pipeline/skills/<name>.py` | implement `gate_<name>`, register in `pipeline/skills/__init__.py::SKILL_GATES`, add `configs/skills/<name>.json` (will be picked up automatically by `load_skills_config`) |
| viz mode | `viz/layer2/<name>.py` | call `add_scene_args` / `add_cache_args` / `add_scenes_root_arg` from `viz._args` for the standard flags; add to `viz/__main__.py::LAYER2`. All viz modes share `--cache-root` (parent of `perception/` / `filter/` / ...), `--adapter`, `--scenes-root`. Modes that load detector/segmenter belong in `cli/`, not viz. |
| CLI subcommand | `cli/<name>.py` | add to `cli/__main__.py::COMMANDS`. CLI is for compute / pipeline ops (generate, balance, qc, debug_pipeline, sample/filter/pair_gate/label/perceive/match/verify); pure-viz commands belong under `viz/layer2/`. Per-stage runners route inputs through `cli/_io.py::load_inputs` and call a single `stage_*` body in `pipeline/stages.py` so `cli generate` can delegate to the same code. |

## Conventions

- **All run artifacts live under `outputs/`, never `/tmp/`.** Every `python -m cli generate / sample / filter / label / verify / balance` invocation, every `python -m viz --mode <X> --save ...` call, every ad-hoc smoke or scratch test, must point its `--out-root` / `--logs-dir` / `--out` / `--out-dir` / `--save` paths under the project's `outputs/` tree. Use `outputs/<descriptive_name>/` (e.g. `outputs/smoke_perception_scene0069/`); the default `logs/<run-id>/` is already inside the project. The slurm runner `scripts/generate_qwen3vl.sh` already follows this convention. Throwaway tests still go under `outputs/scratch_<name>/`, not `/tmp/`.
- **Camera convention**: OpenCV (+X right, +Y down, +Z forward). Adapters that load OpenGL/Blender poses must convert in `load_frame`: `pose_opencv = pose_opengl @ diag([1, -1, -1, 1])`.
- **Cache layout** (model-tagged, no hashes in per-frame caches; the pair-keyed verifier cache appends a 10-char evidence digest — pinned by `tests/test_cache_keys.py`):
  - `cache/perception/<adapter>/<scene>/<detector>+<segmenter>/<frame_id>[__<canon_suffix>].pkl`
  - `cache/labels/<spec.name>/<adapter>/<scene>/<frame_id>.json` — labeler verdicts (labels + canonicals + raw).
  - `cache/filter/<spec.name>/<adapter>/<scene>/<frame_id>.json` — per-frame quality verdicts.
  - `cache/verifier/<spec.name>/<skill>/<scene>/<src>__<tgt>__<evsig>.json` — pair-level verifier verdicts.
  Detector/segmenter thresholds and label prompts are **not** in the cache key — bump the registry name (or `rm -rf` the relevant subdir) when changing those. Caches are portable across hosts (no absolute image paths in the key). The `(adapter, scene_id, frame_id, image_path)` tuple is plumbed via `models._frame_ref.FrameRef`; every detector / labeler / filter takes a `FrameRef`.
- **Inter-stage artifacts** (per-stage CLI handoff format; the pipeline never stores ViewPair / FrameRef in pickle):
  - `frames.json` (`cli sample` → `cli filter|pair_gate|label|perceive`): list of FrameRef dicts (`adapter`, `scene_id`, `frame_id`, `image_path`). Schema in `cli/_frames_io.py`.
  - `pairs.scored.jsonl` (`cli pair_gate` → `cli label|perceive|match`): one JSON object per surviving `ViewPair`, self-describing (carries `adapter`, `scene_id`, `image_src`, `image_tgt`) so consumers reconstruct FrameRefs adapter-free. `tasks` is a sorted list (frozenset on disk would be JSON-incompatible). Schema in `pipeline/pairs_io.py`. Round-trip pinned by `tests/test_pairs_io.py`.
  - `<out_root>/<skill>/pairs.jsonl` (`cli match` → `cli verify`): per-skill `PairManifest` (full pose + intrinsics + objects + evidence). Schema in `pipeline/manifest.py`.
  All three round-trip through `cli/_io.py::load_inputs(path) -> InputBundle`, which auto-detects shape and (for pair files) materializes the unique-frame `FrameRef` list — so every stage CLI accepts any prior-stage output via `--in` without manual reshaping.
- Mask-NMS in SAM2.1/SAM3 defaults to `mask_nms_iou=0.4`.
- ScanNet `_aggregation()` shifts keys by +1 to align with `instance-filt` PNG convention (1-based, 0=background).
- Visualizations use **mask outlines via `ax.contour`** (via `viz.draw_mask_outline`), never bboxes; labels at `mask_centroid(mask)` with `ha="center"`.
- Default content-skill identity uses `mask.canonical or mask.label`. Don't add new label-equality checks elsewhere — extend the canonical accessor pattern.

## Configuration

Hyperparams live in `configs/`, split into three layers — never sprinkled across argparse defaults. The loader is `pipeline/config.py` (no Hydra, no OmegaConf, just JSON + dataclasses).

| File / dir | Owns | Loaded by |
|---|---|---|
| `configs/pair_selection.json` | `selection` floors + `min_frame_gap_by_source` | `load_skills_config()` |
| `configs/skills/<skill>.json` | per-skill gate (one file per name in `CONTENT_SKILLS` ∪ `POSE_SKILLS`) | `load_skills_config()` |
| `configs/stages/<stage>.json` | per-stage knobs (sampler thresholds, model name, concurrency, geometric-match knobs, perception batching, viz/W&B) | `load_stage_config(stage)` |
| `configs/runs/<preset>.json` | top-level preset that picks per-stage configs and applies `stage_overrides` (deep-merged) | `load_run_config(path)` |

**Precedence — two tiers, no hidden defaults:**
- Per-stage CLIs: `--<flag>` wins over `--config <path>` (defaults to `configs/stages/<stage>.json`). Flags default to `None`; the loader copies in config values for any None field.
- `cli generate`: `--<flag>` wins over `--run-config <preset>` wins over per-stage default JSONs. The `_apply_run_config` helper does this in one place.

**No third tier of "built-in defaults" inside Python.** Every committed `configs/stages/*.json` must carry every knob the stage needs; missing keys raise `KeyError` (fail-loud).

Editing a JSON value does **not** invalidate model-tagged caches (`cache/labels/<spec>/...`, `cache/perception/<adapter>/<scene>/<detector>+<segmenter>/...`). To force a recompute, rename the registry spec or `rm -rf` the relevant subdir — same rule that already applies when editing `label_prompt.txt` or detector thresholds.

**`--config` vs `--run-config`:** stage CLIs accept `--config` (per-stage file). `cli generate` accepts `--run-config` (preset). Each loader rejects the other shape with a clear error — no auto-detection.

## Tests

```bash
pytest tests/
```

66 tests cover: cache-key invariants for labels/filter/verifier, mock-adapter pair selection, end-to-end noop pipeline via `python -m cli generate`, end-to-end per-stage chain (`cli sample → pair_gate → match`) byte-equivalent to `cli generate`, projection roundtrip, `pipeline/stages.py` smoke (`_fan_out` ordering, cache-completeness probes, cache-only fast path, fail-closed verifier, `collect_pair_manifests` skip-malformed + filter-by-skill, `write_verified_per_skill` truncate-on-rerun + drop-None-verdicts), Phase-4.5 perception batching, `pairs.scored.jsonl` round-trip + frozenset/sorted-list invariant, `cli/_io.py::load_inputs` auto-detection across frames.json / scored pairs / pair manifests / directory variants, `stage_pair_gate` parity vs `select_pairs`, and the unified config loader (deep-merge of `stage_overrides`, CLI > config precedence, `--config` ↔ `--run-config` cross-rejection, `STAGE_NAMES` ↔ `configs/stages/*.json` coverage).
