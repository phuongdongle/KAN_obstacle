#!/bin/bash -l

#SBATCH --job-name=kan_effects
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/kan_effects_%j.out
#SBATCH --error=logs/kan_effects_%j.err

set -euo pipefail

# ============================================================
# Project paths
# ============================================================

PROJECT="$HOME/kan_project/KAN_obstacle"
PYTHON="$HOME/airquality_graphdyn/.venv/bin/python"

cd "$PROJECT"
mkdir -p logs

echo "============================================================"
echo "KAN EFFECTS BENCHMARK"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Start time:   $(date)"
echo "Project:      $(pwd)"
echo "Python:       $PYTHON"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "============================================================"
echo

# ============================================================
# Check Python environment
# ============================================================

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python executable not found:"
    echo "$PYTHON"
    exit 1
fi

"$PYTHON" --version

# ============================================================
# Check PyTorch and CUDA
# ============================================================

echo
echo "PyTorch / CUDA check"
echo "------------------------------------------------------------"

"$PYTHON" - <<'PY'
import torch

print("PyTorch version :", torch.__version__)
print("CUDA build      :", torch.version.cuda)
print("CUDA available  :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Check the Slurm GPU allocation "
        "and CUDA-enabled PyTorch installation."
    )

print("GPU count       :", torch.cuda.device_count())
print("GPU name        :", torch.cuda.get_device_name(0))
print("Current device  :", torch.cuda.current_device())

props = torch.cuda.get_device_properties(0)
print("GPU memory (GB) :", round(props.total_memory / 1024**3, 2))
PY

echo
echo "nvidia-smi"
echo "------------------------------------------------------------"
nvidia-smi

# ============================================================
# Run benchmark
# ============================================================

echo
echo "============================================================"
echo "Starting KAN benchmark"
echo "============================================================"
echo

"$PYTHON" -u benchmarks/run_kan_effects_benchmark.py \
    --problems all \
    --device cuda \
    --gpu 0 \
    --widths 8,16,32,48 \
    --grids 5,10,15,20

echo
echo "============================================================"
echo "KAN benchmark completed"
echo "End time: $(date)"
echo "============================================================"
