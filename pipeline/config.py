"""Config loading for the Layer 2 pipeline.

Three flavors of config feed into the CLIs:

  * **Skills + pair-selection** (assembled from
    ``configs/pair_selection.json`` + ``configs/skills/<name>.json``).
    Drives the pose pre-filter and per-skill content gates. Loaded by
    ``load_skills_config()`` and resolved per-source via ``resolve()``.
    ``load_config(path=None)`` is a back-compat alias.

  * **Per-stage** (``configs/stages/<stage>.json``). One file per pipeline
    stage holds the knob defaults for that stage (sampler thresholds,
    model name, concurrency, geometric-match knobs, etc.). Loaded by
    ``load_stage_config(stage)``.

  * **Run preset** (``configs/runs/<preset>.json``). Top-level file used by
    ``python -m cli generate --run-config <preset>``: picks the per-stage
    config files and applies a deep-merged ``stage_overrides`` block.
    Loaded by ``load_run_config(path)``.

CLI flags always win over config values; the precedence helper is
``merge_cli_overrides()``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

CONFIGS_ROOT = Path(__file__).resolve().parents[1] / "configs"
PAIR_SELECTION_PATH = CONFIGS_ROOT / "pair_selection.json"
SKILLS_DIR = CONFIGS_ROOT / "skills"
STAGES_DIR = CONFIGS_ROOT / "stages"

# Kept as a sentinel for callers that want "the default skills config" —
# now it's a directory rather than a file, but the symbol remains for
# back-compat with `cli match --task-config <path>` etc.
DEFAULT_CONFIG_PATH = SKILLS_DIR

# Stage names with a shipped configs/stages/<name>.json. Used by
# load_stage_config() to validate names and by tests.
STAGE_NAMES = ("sample", "filter", "pair_gate", "label",
               "perceive", "match", "verify")


# --------------------------------------------------------------------- #
# skills + pair-selection                                                #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class TaskGate:
    """Resolved per-task gates for a single source."""
    name: str
    min_frame_gap: int
    angle_deg_min: float = 0.0
    median_depth_ratio_min: float = 0.0
    overlap_min: float = 0.0
    occluded_fraction_min: float = 0.0


@dataclass(frozen=True)
class PairConfig:
    """Top-level pair-selection config, resolved for a single source."""
    source: str
    pair_quality_min: float
    pair_diversity_min_m: float
    corner_overlap_min: float
    angle_min_deg: float
    angle_max_deg: float
    max_distance_m: float
    min_yaw_diff_deg: float
    source_floor: int
    tasks: dict[str, TaskGate]

    @property
    def min_frame_gap_pre(self) -> int:
        if not self.tasks:
            return self.source_floor
        return min(g.min_frame_gap for g in self.tasks.values())


def _strip_comments(d: dict[str, Any]) -> dict[str, Any]:
    """Return ``d`` minus keys starting with '_'. Recursive on nested dicts."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(k, str) and k.startswith("_"):
            continue
        out[k] = _strip_comments(v) if isinstance(v, dict) else v
    return out


def load_skills_config() -> dict[str, Any]:
    """Assemble the merged skills/pair-selection dict from
    ``configs/pair_selection.json`` + ``configs/skills/<name>.json``.

    Output shape (matches what ``resolve()`` and ``load_content_skills()``
    expect):

        {
          "selection": {...},
          "min_frame_gap_by_source": {...},
          "tasks":          {<pose_skill>: {...}, ...},
          "content_skills": {<content_skill>: {...}, ...}
        }
    """
    # Lazy import to avoid a pipeline.config <-> pipeline.skills cycle at
    # module load time.
    from pipeline.skills.base import CONTENT_SKILLS, POSE_SKILLS

    if not PAIR_SELECTION_PATH.exists():
        raise FileNotFoundError(
            f"missing {PAIR_SELECTION_PATH} — required for pair selection"
        )
    base = json.loads(PAIR_SELECTION_PATH.read_text())
    cfg: dict[str, Any] = {
        "selection": base["selection"],
        "min_frame_gap_by_source": base["min_frame_gap_by_source"],
        "tasks": {},
        "content_skills": {},
    }

    for name in (*POSE_SKILLS, *CONTENT_SKILLS):
        p = SKILLS_DIR / f"{name}.json"
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p} — every skill listed in CONTENT_SKILLS / "
                f"POSE_SKILLS must have a configs/skills/<name>.json"
            )
        spec = _strip_comments(json.loads(p.read_text()))
        bucket = "tasks" if name in POSE_SKILLS else "content_skills"
        cfg[bucket][name] = spec

    return cfg


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Back-compat alias for ``load_skills_config``.

    The legacy ``--task-config <path>`` flag passed a path to ``tasks.json``;
    that file is gone. If ``path`` is supplied it must point at a JSON file
    with the same shape ``load_skills_config()`` produces (e.g. a snapshot
    saved by a downstream tool); otherwise the assembled-from-split config
    is returned.
    """
    if path is None or Path(path) == SKILLS_DIR:
        return load_skills_config()
    p = Path(path)
    if p.is_dir():
        return load_skills_config()
    return json.loads(p.read_text())


def resolve(cfg: dict, source: str) -> PairConfig:
    """Resolve the merged skills/pair-selection dict against one source."""
    sel = cfg["selection"]
    src_floors = cfg["min_frame_gap_by_source"]
    floor = int(src_floors.get(source, src_floors.get("unknown", 0)))

    tasks: dict[str, TaskGate] = {}
    for name, spec in cfg.get("tasks", {}).items():
        bonus_map = spec.get("min_frame_gap_bonus_by_source", {})
        bonus = int(bonus_map.get(source, bonus_map.get("unknown", 0)))
        tasks[name] = TaskGate(
            name=name,
            min_frame_gap=floor + bonus,
            angle_deg_min=float(spec.get("angle_deg_min", 0.0)),
            median_depth_ratio_min=float(spec.get("median_depth_ratio_min", 0.0)),
            overlap_min=float(spec.get("overlap_min", 0.0)),
            occluded_fraction_min=float(spec.get("occluded_fraction_min", 0.0)),
        )

    return PairConfig(
        source=source,
        pair_quality_min=float(sel["pair_quality_min"]),
        pair_diversity_min_m=float(sel["pair_diversity_min_m"]),
        corner_overlap_min=float(sel["corner_overlap_min"]),
        angle_min_deg=float(sel["angle_min_deg"]),
        angle_max_deg=float(sel["angle_max_deg"]),
        max_distance_m=float(sel["max_distance_m"]),
        min_yaw_diff_deg=float(sel.get("min_yaw_diff_deg", 0.0)),
        source_floor=floor,
        tasks=tasks,
    )


# --------------------------------------------------------------------- #
# per-stage configs                                                      #
# --------------------------------------------------------------------- #


def stage_config_path(stage: str) -> Path:
    """Default JSON path for a stage's config. ``stage`` must be one of
    the known stage names (validated against ``STAGE_NAMES``)."""
    if stage not in STAGE_NAMES:
        raise ValueError(
            f"unknown stage {stage!r}; expected one of {STAGE_NAMES}"
        )
    return STAGES_DIR / f"{stage}.json"


def load_stage_config(stage: str, path: Path | None = None) -> dict[str, Any]:
    """Load and validate a per-stage config file.

    Raises ``ValueError`` if the file looks like a run preset (has a
    top-level 'stages' key) — that catches the common "passed the wrong
    --config" mistake loudly.
    """
    p = Path(path) if path is not None else stage_config_path(stage)
    if not p.exists():
        raise FileNotFoundError(
            f"stage config {p} does not exist — restore from git or pass "
            f"--config <path>"
        )
    raw = json.loads(p.read_text())
    if "stages" in raw:
        raise ValueError(
            f"{p} looks like a run preset (has top-level 'stages' key) — "
            f"did you mean --run-config? --config expects a per-stage file."
        )
    return _strip_comments(raw)


# --------------------------------------------------------------------- #
# run presets                                                            #
# --------------------------------------------------------------------- #


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``over`` onto ``base``. Returns a new dict. Nested dicts
    are merged recursively; non-dict values in ``over`` replace those in
    ``base`` outright (lists are *not* concatenated)."""
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class RunPreset:
    """Resolved run preset: per-stage dicts already deep-merged with
    stage_overrides, plus the top-level run knobs.

    ``stages`` maps stage_name → resolved dict. Use it instead of touching
    the raw preset.
    """
    path: Path
    adapter: str
    scenes_root: Path
    out_root: Path
    logs_dir: Path
    stages: dict[str, dict[str, Any]]
    extras: dict[str, Any]


def load_run_config(path: Path) -> RunPreset:
    """Load a run preset and resolve every per-stage dict.

    Refuses to load a per-stage file (no top-level ``stages`` block) — the
    error mirrors ``load_stage_config``'s "wrong --config" check.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"run preset {p} does not exist")
    raw = _strip_comments(json.loads(p.read_text()))
    if "stages" not in raw:
        raise ValueError(
            f"{p} has no top-level 'stages' block — that looks like a "
            f"per-stage config, not a run preset. Did you mean --config?"
        )
    overrides_all = raw.get("stage_overrides", {})
    stages: dict[str, dict[str, Any]] = {}
    for stage, stage_path in raw["stages"].items():
        stage_cfg = load_stage_config(stage, Path(stage_path))
        if stage in overrides_all:
            stage_cfg = _deep_merge(stage_cfg, overrides_all[stage])
        stages[stage] = stage_cfg
    extras = {k: v for k, v in raw.items()
              if k not in {"adapter", "scenes_root", "out_root", "logs_dir",
                           "stages", "stage_overrides"}}
    return RunPreset(
        path=p,
        adapter=raw.get("adapter", "scannet"),
        scenes_root=Path(raw["scenes_root"]) if "scenes_root" in raw
                    else Path("/home/mila/l/leh/scratch/dataset/scannet_data/scans"),
        out_root=Path(raw.get("out_root", "outputs/run")),
        logs_dir=Path(raw.get("logs_dir", "logs")),
        stages=stages,
        extras=extras,
    )


# --------------------------------------------------------------------- #
# CLI override helper                                                    #
# --------------------------------------------------------------------- #


def merge_cli_with_config(args: argparse.Namespace,
                          cfg: dict[str, Any],
                          keys: Iterable[str]) -> None:
    """Two-tier precedence: CLI flag > config file.

    For each ``key`` in ``keys``:
      * If ``args.<key>`` is not None, the CLI flag wins — leave args alone.
      * Otherwise, copy ``cfg[key]`` (when present) onto ``args.<key>``.

    A key absent from both ``cfg`` and ``args`` is left at None and the
    downstream code is responsible for raising — fail-loud is the contract.
    """
    for k in keys:
        if getattr(args, k, None) is not None:
            continue
        if k in cfg:
            setattr(args, k, cfg[k])
