#!/bin/bash
#SBATCH --job-name=build_mmcv
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/datastore/aniketag/mmdetection3d/outputs/build_mmcv_%j.log
#SBATCH --error=/cluster/datastore/aniketag/mmdetection3d/outputs/build_mmcv_%j.log

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

PIP=/cluster/datastore/aniketag/allEnv/mmdet3d/bin/pip
PYTHON=/cluster/datastore/aniketag/allEnv/mmdet3d/bin/python

echo "=== Node: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== CUDA: $(nvcc --version | grep release) ==="
echo "=== Torch: $($PYTHON -c 'import torch; print(torch.__version__, torch.version.cuda)') ==="

# Remove any prebuilt mmcv
echo "Uninstalling prebuilt mmcv..."
$PIP uninstall -y mmcv 2>/dev/null || true

# Build mmcv from source (compiles against whatever CUDA/torch is installed)
echo "Building mmcv 2.1.0 from source..."
MMCV_WITH_OPS=1 $PIP install "mmcv==2.1.0" \
    --no-binary mmcv \
    --cache-dir /cluster/datastore/aniketag/.pip_cache \
    -v 2>&1

echo "=== Build done ==="
$PYTHON -c "
import mmcv
from mmcv.ops import box_iou_rotated
print('mmcv:', mmcv.__version__, '— ops OK')
"
