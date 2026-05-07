#!/bin/bash
# =============================================================================
# CrossPoint Layer 2: generate correspondences via run-preset.
#
# All knobs (sampler, detector/segmenter, models, perception batching,
# match thresholds, viz) live in configs/runs/<preset>.json. Edit that
# file (or copy it) to change behavior — this wrapper is intentionally
# thin.
#
# Usage:
#   sbatch scripts/generate_qwen3vl.sh                   # all scenes
#   sbatch scripts/generate_qwen3vl.sh scene0093_00      # single-scene smoke
#
# Environment overrides:
#   RUN_CONFIG=configs/runs/<preset>.json   default qwen3vl_default.json
#   PERCEPTION_WORKERS=<int>                default $(nvidia-smi -L | wc -l)
# =============================================================================
#SBATCH --job-name=cp_qwen3vl
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH -c 24
#SBATCH --mem=200Gb
#SBATCH --time=3:00:00
#SBATCH --output=/home/mila/l/leh/projects/CrossPoint-Objects/logs/slurm/cp_qwen3vl_%j.out
#SBATCH --error=/home/mila/l/leh/projects/CrossPoint-Objects/logs/slurm/cp_qwen3vl_%j.err

set -e

SCENE_ID=${1:-}

REPO=/home/mila/l/leh/projects/CrossPoint-Objects
SCRATCH=/home/mila/l/leh/scratch
LOG_DIR=$REPO/logs/slurm
OUT_ROOT=$SCRATCH/dataset/cross-points/qwen3vl_run_${SLURM_JOB_ID:-local}

mkdir -p $LOG_DIR $OUT_ROOT

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

# Loud deprecation: the LABELER / FILTER env vars used to swap models.
# Now you edit configs/runs/<preset>.json instead. Fail fast if either is
# set so users don't think the env var still does something.
if [ -n "${LABELER:-}" ] || [ -n "${FILTER:-}" ]; then
    log "ERROR: \$LABELER / \$FILTER env vars are no longer supported."
    log "       Edit configs/runs/<preset>.json (e.g. copy qwen3vl_default.json"
    log "       to qwen3vl_8B.json and change stage_overrides.label.model)."
    exit 2
fi

# Environment. The conda env is on Lustre /home (not BeeGFS /scratch) —
# Triton's per-kernel JIT shells out to gcc and was hitting NFS attribute-
# cache races when 4 vLLM TP workers compiled concurrently against headers
# under /network/scratch. CC/CXX point at the conda-shipped gcc so its
# sysroot is consistent; TRITON_CACHE_DIR puts compiled .so files on
# node-local disk so subsequent kernels skip the compile entirely.
unset ROCR_VISIBLE_DEVICES
source /network/scratch/l/leh/miniconda3/etc/profile.d/conda.sh
conda activate /home/mila/l/leh/envs/cp
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton-$USER
mkdir -p "$TRITON_CACHE_DIR"

RUN_CONFIG="${RUN_CONFIG:-configs/runs/qwen3vl_default.json}"

log "run_config=$RUN_CONFIG scene=${SCENE_ID:-<all>}"

# vLLM workers from a prior crashed job can outlive their parent. Sweep
# before the pipeline starts, so launch_server() gets a clean port.
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "VLLM::"           2>/dev/null || true
sleep 3

cd "$REPO"

SCENE_ARGS=()
if [ -n "$SCENE_ID" ]; then
    SCENE_ARGS+=(--scene "$SCENE_ID")
else
    SCENE_ARGS+=(--all-scenes)
fi

# Default = visible GPU count; override with PERCEPTION_WORKERS=0 to force
# the legacy serial Phase 5.
PERCEPTION_WORKERS="${PERCEPTION_WORKERS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

python -m cli generate \
    --run-config "$RUN_CONFIG" \
    "${SCENE_ARGS[@]}" \
    --perception-workers "$PERCEPTION_WORKERS" \
    --out-root "$OUT_ROOT" \
    --out      correspondences.jsonl

log "Done. Output: $OUT_ROOT/correspondences.jsonl"
