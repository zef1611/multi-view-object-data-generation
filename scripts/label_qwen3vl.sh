#!/bin/bash
# =============================================================================
# CrossPoint Layer 2: run the labeler stage on its own.
#
# Two-step flow (idempotent — both stages skip cached entries):
#   1. python -m cli sample  --scene <id> ...   → frames.json
#   2. python -m cli label   --in frames.json --labeler qwen3vl-235B
#
# Usage:
#   sbatch scripts/label_qwen3vl.sh scene0270_00
#   sbatch scripts/label_qwen3vl.sh scene0270_00 scene0093_00
#
# Environment overrides:
#   LABELER=qwen3vl-235B           default; alternatives: qwen3vl-8B
#   OUT_ROOT=outputs/<dir>         default outputs/label_<jobid>
#   SCENES_ROOT=<path>             default scannet_data/scans
# =============================================================================
#SBATCH --job-name=cp_label
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH -c 24
#SBATCH --mem=200Gb
#SBATCH --time=3:00:00
#SBATCH --output=/home/mila/l/leh/projects/CrossPoint-Objects/logs/slurm/cp_label_%j.out
#SBATCH --error=/home/mila/l/leh/projects/CrossPoint-Objects/logs/slurm/cp_label_%j.err

set -e

if [ "$#" -lt 1 ]; then
    echo "usage: sbatch $0 <scene_id> [<scene_id> ...]" >&2
    exit 2
fi

REPO=/home/mila/l/leh/projects/CrossPoint-Objects
LABELER="${LABELER:-qwen3vl-235B}"
SCENES_ROOT="${SCENES_ROOT:-/home/mila/l/leh/scratch/dataset/scannet_data/scans}"
OUT_ROOT="${OUT_ROOT:-$REPO/outputs/label_${SLURM_JOB_ID:-local}}"
LOG_DIR=$REPO/logs/slurm

mkdir -p "$LOG_DIR" "$OUT_ROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

# Same env setup as generate_qwen3vl.sh — see comments there.
unset ROCR_VISIBLE_DEVICES
source /network/scratch/l/leh/miniconda3/etc/profile.d/conda.sh
conda activate /home/mila/l/leh/envs/cp
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton-$USER
mkdir -p "$TRITON_CACHE_DIR"

# Sweep orphaned vLLM workers from prior crashed jobs.
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "VLLM::"           2>/dev/null || true
sleep 3

cd "$REPO"

# Phase 1 — sample frames per scene (CPU-only, fast).
FRAME_FILES=()
for SCENE in "$@"; do
    OUT=$OUT_ROOT/$SCENE
    mkdir -p "$OUT"
    log "sample: $SCENE"
    python -m cli sample \
        --scene "$SCENE" \
        --scenes-root "$SCENES_ROOT" \
        --out "$OUT/frames.json"
    FRAME_FILES+=("$OUT/frames.json")
done

# Phase 2 — label every frames.json. The labeler launches one vLLM server
# lifetime (cli label loops over frames internally); to span multiple
# scenes on a single server lifetime, concatenate frames.json files first.
COMBINED=$OUT_ROOT/all_frames.json
python -c "
import json, sys
out = []
for p in sys.argv[1:]:
    out.extend(json.load(open(p)))
json.dump(out, open('$COMBINED', 'w'))
print(f'combined {len(out)} frames from {len(sys.argv)-1} scenes -> $COMBINED')
" "${FRAME_FILES[@]}"

log "label: ${#FRAME_FILES[@]} scene(s), labeler=$LABELER"
python -m cli label \
    --in "$COMBINED" \
    --labeler "$LABELER" \
    --logs-dir "$OUT_ROOT/logs" \
    --run-id "label_${LABELER}"

log "Done. Cache: cache/labels/$LABELER/scannet/<scene>/<frame>.json"
log "      Run logs: $OUT_ROOT/logs/label_${LABELER}/"
