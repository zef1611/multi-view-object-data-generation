#!/bin/bash
# =============================================================================
# Unified launcher for cli filter / label / verify.
#
# Resumes from per-frame/per-pair cache regardless of how it's invoked
# (sbatch, sbatch array, interactive bash, nested call from another
# script). Cache lives at:
#   cache/filter/<spec>/...
#   cache/labels/<spec>/...
#   cache/verifier/<spec>/...
# with atomic per-item writes — kill-safe at any point. Re-running the
# same command is the resume operation.
#
# Usage:
#   bash   scripts/run_vlm_stage.sh filter --in frames.json
#   bash   scripts/run_vlm_stage.sh label  --in pairs.scored.jsonl
#   bash   scripts/run_vlm_stage.sh verify --in outputs/<run>/anchor/pairs.jsonl
#   sbatch scripts/run_vlm_stage.sh filter --in frames.json
#   sbatch --array=1-20%1 scripts/run_vlm_stage.sh verify --in <pairs.jsonl>
# =============================================================================
#SBATCH --job-name=cp_vlm
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:4
#SBATCH -c 24
#SBATCH --mem=200Gb
#SBATCH --time=12:00:00
#SBATCH --array=1-10%1
#SBATCH --output=/home/mila/l/leh/projects/CrossPoint-Objects/logs/slurm/cp_vlm_%A_%a.out
#SBATCH --error=/home/mila/l/leh/projects/CrossPoint-Objects/logs/slurm/cp_vlm_%A_%a.err

set -e

STAGE="${1:?usage: $0 <filter|label|verify> [...cli args]}"
shift
case "$STAGE" in
  filter|label|verify) ;;
  *) echo "unknown stage: $STAGE (expected filter|label|verify)"; exit 2 ;;
esac

REPO=/home/mila/l/leh/projects/CrossPoint-Objects
mkdir -p "$REPO/logs/slurm"

unset ROCR_VISIBLE_DEVICES
source /network/scratch/l/leh/miniconda3/etc/profile.d/conda.sh
conda activate /home/mila/l/leh/envs/cp
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton-$USER
mkdir -p "$TRITON_CACHE_DIR"

pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "VLLM::"           2>/dev/null || true
sleep 3

cd "$REPO"

DEFAULT_RUN_ID="${STAGE}__${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-$$}"
if [[ "$*" != *"--run-id"* ]]; then
    set -- --run-id "$DEFAULT_RUN_ID" "$@"
fi

exec python -m cli "$STAGE" "$@"
