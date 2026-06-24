#!/bin/bash
#SBATCH --job-name=build_mmcv_normal
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/cluster/datastore/aniketag/mmdetection3d/logs/build_mmcv_normal_%j.log
#SBATCH --error=/cluster/datastore/aniketag/mmdetection3d/logs/build_mmcv_normal_%j.log

# hpc1-8 (RTX 8000, sm_75): system CUDA toolkit is runtime-only (no nvcc).
# nvcc 12.1 is installed in a conda prefix at allEnv/nvcc121.
# Strategy: use that nvcc + torch cu121 to compile mmcv 2.1.0 from source for sm_75.

# nvcc wrapper adds --allow-unsupported-compiler (GCC 13 > nvcc12.1's GCC 12 limit)
export CUDA_HOME=/cluster/datastore/aniketag/allEnv/nvcc121
export PATH=/cluster/datastore/aniketag/allEnv/nvcc121_wrap:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# RTX 8000 = sm_75
export TORCH_CUDA_ARCH_LIST="7.5"
export MMCV_WITH_OPS=1

ENV=/cluster/datastore/aniketag/allEnv/mmdet3d_normal
PIP=$ENV/bin/pip
PYTHON=$ENV/bin/python
CACHE=/cluster/datastore/aniketag/.pip_cache

echo "=== Node: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ') ==="
echo "=== nvcc: $(nvcc --version | grep release) ==="

# Reinstall torch to cu121 to match nvcc 12.1
echo "Installing torch 2.4.0+cu121..."
$PIP install torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --no-cache-dir -q
echo "Torch: $($PYTHON -c 'import torch; print(torch.__version__, torch.version.cuda)')"

# Build mmcv 2.1.0 from source for sm_75
echo "Building mmcv 2.1.0 from source for sm_75 (RTX 8000)..."
$PIP uninstall -y mmcv 2>/dev/null || true
MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST="7.5" $PIP install "mmcv==2.1.0" \
    --no-binary mmcv \
    --no-cache-dir \
    -v 2>&1

echo ""
echo "=== Verifying — running actual CUDA kernels on RTX 8000 ==="
$PYTHON -c "
import sys
sys.path = [p for p in sys.path if '.local' not in p]
import torch, mmcv, mmdet, mmdet3d, mmengine

print('torch    :', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('GPU name :', torch.cuda.get_device_name(0))
print('mmcv     :', mmcv.__version__)
print('mmdet    :', mmdet.__version__)
print('mmdet3d  :', mmdet3d.__version__)
print('mmengine :', mmengine.__version__)

from mmcv.utils import ext_loader
ext_mod = ext_loader.load_ext('_ext', ['hard_voxelize_forward'])
pts     = torch.randn(1000, 4).cuda()
voxels  = torch.zeros(16000, 32, 4, dtype=torch.float32).cuda()
coors   = torch.zeros(16000, 3,  dtype=torch.int32).cuda()
num_pts = torch.zeros(16000,     dtype=torch.int32).cuda()
vox_num = torch.zeros(1,         dtype=torch.int32).cuda()
ext_mod.hard_voxelize_forward(
    pts, voxels, coors, num_pts, vox_num,
    [0.16, 0.16, 4.0], [0.0, -40.0, -3.0, 70.4, 40.0, 1.0],
    32, 16000, 3, True)
print('hard_voxelize_forward : OK — RTX 8000 (sm_75) kernel works!')
from mmcv.ops import box_iou_rotated, points_in_boxes_all, dynamic_scatter
print('other ops             : OK')
"

echo ""
echo "=== Build complete ==="
