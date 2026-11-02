#!/bin/bash
#SBATCH --job-name=qr8_v2_a6000
#SBATCH --partition=P2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=c03,c04
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
JOB_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/runs/a6000_v2_${JOB_ID}}"
MODE="${MODE:-validation}"
CUDA_ARCH="${CUDA_ARCH:-sm_86}"
GPU_NAME_PATTERN="${GPU_NAME_PATTERN:-A6000}"
WARMUPS="${WARMUPS:-5}"
TRIALS="${TRIALS:-30}"
MAX_REFINEMENTS="${MAX_REFINEMENTS:-5}"
TOLERANCE="${TOLERANCE:-1e-12}"
OUTPUT="${OUTPUT:-both}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$RUN_ROOT" results

source /etc/profile.d/modules.sh 2>/dev/null || true
source /usr/share/Modules/init/bash 2>/dev/null || true
if type module >/dev/null 2>&1; then
    module load cuda-12.1.module 2>/dev/null || module load cuda/12.1 2>/dev/null || \
    module load cuda-12.1 2>/dev/null || module load cuda-12.2 2>/dev/null || \
    module load cuda-12.3 2>/dev/null || module load cuda-12.4 2>/dev/null || true
fi
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc unavailable"; exit 20; }

GPU_LIST="$(nvidia-smi -L 2>&1 || true)"
{
    echo "job_id=$JOB_ID"
    echo "mode=$MODE"
    echo "cuda_arch=$CUDA_ARCH"
    echo "gpu_name_pattern=$GPU_NAME_PATTERN"
    echo "date=$(date)"
    echo "hostname=$(hostname)"
    echo "gpus=$GPU_LIST"
} | tee "$RUN_ROOT/job_config.txt"
if ! grep -qi "$GPU_NAME_PATTERN" <<< "$GPU_LIST"; then
    echo "ERROR: allocated GPU does not match pattern '$GPU_NAME_PATTERN'" | tee -a "$RUN_ROOT/job_config.txt"
    exit 40
fi

nvidia-smi 2>&1 | tee "$RUN_ROOT/environment_before_build.txt"
make clean
make CUDA_ARCH="$CUDA_ARCH"
make smoke
python3 tests/validate_results.py results/smoke.csv

if [ "$MODE" = "validation" ]; then
    ./qr8_benchmark --m 513 --n 31 --warmups 1 --trials 2 --maximum-refinements 3 \
        --raw-csv "$RUN_ROOT/odd_synthetic_validation.csv"
    python3 tests/validate_results.py "$RUN_ROOT/odd_synthetic_validation.csv"
    ./qr8_benchmark --m 4096 --n 256 --condition 100 --output q --warmups 1 --trials 3 \
        --ns-inverse-iterations 20 --ns-polar-iterations 20 --ns-inverse-tolerance 1e-10 \
        --raw-csv "$RUN_ROOT/ns_validation.csv"
    python3 tests/validate_results.py "$RUN_ROOT/ns_validation.csv"
    make suite-smoke
    LATEST_RAW="$(find results/suite-smoke -name raw_all.csv -type f | sort | tail -n 1)"
    python3 tests/validate_results.py "$LATEST_RAW"
elif [ "$MODE" = "full" ]; then
    python3 scripts/run_suite.py --warmups "$WARMUPS" --trials "$TRIALS" \
        --maximum-refinements "$MAX_REFINEMENTS" --tolerance "$TOLERANCE" \
        --output "$OUTPUT" --results "$RUN_ROOT/primary"
else
    echo "ERROR: unsupported MODE=$MODE" | tee -a "$RUN_ROOT/job_config.txt"
    exit 30
fi

PACKAGE_DIR="${PACKAGE_DIR:-$PROJECT_DIR/packages}"
mkdir -p "$PACKAGE_DIR"
PACK="$PACKAGE_DIR/qr8_v2_a6000_${JOB_ID}.tar.gz"
tar -C "$RUN_ROOT" -czf "$PACK" .
echo "package=$PACK"
