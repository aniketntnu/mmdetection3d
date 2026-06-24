#!/bin/bash
#SBATCH --job-name=build_mmcv3
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/cluster/datastore/aniketag/mmdetection3d/outputs/build_mmcv3_%j.log
#SBATCH --error=/cluster/datastore/aniketag/mmdetection3d/outputs/build_mmcv3_%j.log

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
unset PYTHONPATH

ENV=/cluster/datastore/aniketag/allEnv/mmdet3d
PIP=$ENV/bin/pip
PYTHON=$ENV/bin/python

echo "=== $(hostname) | CUDA: $(nvcc --version | grep release) ==="

# Replace 2.2.0 with 2.1.0 (compatible with mmdet 3.3.0)
$PIP uninstall -y mmcv 2>/dev/null || true

# Use the wheel already built and cached if available
CACHED_WHEEL=$(ls /cluster/datastore/aniketag/.pip_cache/wheels/*/*/mmcv-2.1.0*.whl 2>/dev/null | head -1)
if [ -n "$CACHED_WHEEL" ]; then
    echo "Found cached mmcv 2.1.0 wheel: $CACHED_WHEEL"
    $PIP install "$CACHED_WHEEL"
else
    echo "Building mmcv 2.1.0 from source..."
    MMCV_WITH_OPS=1 $PIP install "mmcv==2.1.0" \
        --no-binary mmcv \
        --cache-dir /cluster/datastore/aniketag/.pip_cache \
        -v
fi

echo "=== Verifying ==="
$PYTHON -c "
import mmcv, mmdet, mmdet3d, torch
print('mmcv     :', mmcv.__version__)
print('mmdet    :', mmdet.__version__)
print('mmdet3d  :', mmdet3d.__version__)
print('torch    :', torch.__version__, '| CUDA:', torch.cuda.is_available())
from mmcv.ops import box_iou_rotated, points_in_boxes_all
print('mmcv ops : OK')
"
