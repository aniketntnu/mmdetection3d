#!/bin/bash
#SBATCH --job-name=build_mmcv2
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/cluster/datastore/aniketag/mmdetection3d/outputs/build_mmcv2_%j.log
#SBATCH --error=/cluster/datastore/aniketag/mmdetection3d/outputs/build_mmcv2_%j.log

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Isolate from ~/.local pip packages that would interfere with build
export PYTHONNOUSERSITE=1
unset PYTHONPATH

ENV=/cluster/datastore/aniketag/allEnv/mmdet3d
PIP=$ENV/bin/pip
PYTHON=$ENV/bin/python

echo "=== Node: $(hostname) ==="
echo "=== CUDA: $(nvcc --version | grep release) ==="
echo "=== Torch from ENV: $($PYTHON -c 'import sys; sys.path = [p for p in sys.path if ".local" not in p]; import torch; print(torch.__file__, torch.__version__, torch.version.cuda)') ==="

# Remove any prebuilt mmcv
$PIP uninstall -y mmcv 2>/dev/null || true

# Upgrade to torch 2.4 + cu124 (matches CUDA 12.x better)
echo "Installing PyTorch 2.4+cu124..."
$PIP install torch==2.4.0+cu124 torchvision==0.19.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --cache-dir /cluster/datastore/aniketag/.pip_cache -q

echo "Torch version: $($PYTHON -c 'import torch; print(torch.__version__, torch.version.cuda)')"

# Install matching mmcv prebuilt wheel for cu124/torch2.4
echo "Installing mmcv for cu124+torch2.4..."
$PIP install mmcv==2.2.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu124/torch2.4/index.html \
    --cache-dir /cluster/datastore/aniketag/.pip_cache

echo "=== Verifying mmcv ops ==="
$PYTHON -c "
import sys
sys.path = [p for p in sys.path if '.local' not in p]
import mmcv
from mmcv.ops import box_iou_rotated, points_in_boxes_all
print('mmcv:', mmcv.__version__, '— ops OK')
import torch
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
"
