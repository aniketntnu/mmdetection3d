#!/bin/bash
#SBATCH --job-name=pvrcnn_a2d2
#SBATCH --partition=a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10-00:00:00
#SBATCH --output=/cluster/datastore/aniketag/mmdetection3d/logs/train_a2d2_%j.log
#SBATCH --error=/cluster/datastore/aniketag/mmdetection3d/logs/train_a2d2_%j.log

mkdir -p /cluster/datastore/aniketag/mmdetection3d/logs

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
unset PYTHONPATH

REPO=/cluster/datastore/aniketag/mmdetection3d
PYTHON=/cluster/datastore/aniketag/allEnv/mmdet3d/bin/python
CONFIG=$REPO/configs/pv_rcnn/pv_rcnn_a2d2.py
WORK_DIR=$REPO/work_dirs/pv_rcnn_a2d2

mkdir -p $WORK_DIR/videos
mkdir -p $REPO/outputs/gt_video
mkdir -p $REPO/outputs/pred_videos

echo "=== Node: $(hostname) ==="
echo "=== GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ') ==="
echo "=== Start: $(date) ==="

cd $REPO

PYTHONUNBUFFERED=1 $PYTHON -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29500 \
    $REPO/tools/train.py \
    $CONFIG \
    --launcher pytorch \
    --work-dir $WORK_DIR \
    --resume

echo "=== Training finished at $(date) ==="
