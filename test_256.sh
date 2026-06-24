#!/bin/bash
#SBATCH --job-name=pvrcnn_256
#SBATCH --partition=a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/datastore/aniketag/mmdetection3d/logs/test256_%j.log
#SBATCH --error=/cluster/datastore/aniketag/mmdetection3d/logs/test256_%j.log

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
unset PYTHONPATH

REPO=/cluster/datastore/aniketag/mmdetection3d
PYTHON=/cluster/datastore/aniketag/allEnv/mmdet3d/bin/python

cd $REPO
echo "=== 256-sample test: batch=16, 30 epochs ==="

PYTHONUNBUFFERED=1 $PYTHON -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29502 \
    $REPO/tools/train.py \
    $REPO/configs/pv_rcnn/pv_rcnn_a2d2.py \
    --launcher pytorch \
    --work-dir $REPO/work_dirs/test256 \
    --cfg-options \
        train_dataloader.dataset.ann_file=a2d2_infos_256.pkl \
        val_dataloader.dataset.ann_file=a2d2_infos_256.pkl \
        val_evaluator.ann_file=data/a2d2/a2d2_infos_256.pkl \
        train_cfg.max_epochs=5 \
        train_cfg.val_interval=9999 \
        default_hooks.checkpoint.interval=10 \
        default_hooks.logger.interval=2

echo "=== 256-sample test done: $(date) ==="
