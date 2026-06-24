#!/bin/bash
#SBATCH --job-name=pvrcnn_debug
#SBATCH --partition=a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/datastore/aniketag/mmdetection3d/logs/debug_%j.log
#SBATCH --error=/cluster/datastore/aniketag/mmdetection3d/logs/debug_%j.log

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
unset PYTHONPATH

REPO=/cluster/datastore/aniketag/mmdetection3d
PYTHON=/cluster/datastore/aniketag/allEnv/mmdet3d/bin/python

cd $REPO

echo "=== DEBUG RUN: 16 samples, 30 epochs on A100 ==="
echo "=== Start: $(date) ==="

PYTHONUNBUFFERED=1 $PYTHON -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29501 \
    $REPO/tools/train.py \
    $REPO/configs/pv_rcnn/pv_rcnn_a2d2.py \
    --launcher pytorch \
    --work-dir $REPO/work_dirs/debug \
    --cfg-options \
        train_dataloader.dataset.ann_file=a2d2_infos_debug.pkl \
        val_dataloader.dataset.ann_file=a2d2_infos_debug.pkl \
        val_evaluator.ann_file=data/a2d2/a2d2_infos_debug.pkl \
        train_cfg.max_epochs=30 \
        train_cfg.val_interval=9999 \
        default_hooks.checkpoint.interval=10 \
        default_hooks.logger.interval=5

echo "=== Debug finished: $(date) ==="
