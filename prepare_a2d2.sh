#!/bin/bash
#SBATCH --job-name=a2d2_prep
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/datastore/aniketag/mmdetection3d/logs/a2d2_prep_%j.log
#SBATCH --error=/cluster/datastore/aniketag/mmdetection3d/logs/a2d2_prep_%j.log

mkdir -p /cluster/datastore/aniketag/mmdetection3d/logs \
         /cluster/datastore/aniketag/mmdetection3d/outputs/gt_video \
         /cluster/datastore/aniketag/mmdetection3d/outputs/pred_videos \
         /cluster/datastore/aniketag/mmdetection3d/work_dirs/pv_rcnn_a2d2/videos

export PYTHONNOUSERSITE=1
unset PYTHONPATH

PYTHON=/cluster/datastore/aniketag/allEnv/mmdet3d/bin/python
REPO=/cluster/datastore/aniketag/mmdetection3d

cd $REPO
echo "=== Node: $(hostname) | CPUs: $(nproc) | Start: $(date) ==="
PYTHONUNBUFFERED=1 $PYTHON tools/prepare_a2d2.py

if [ $? -eq 0 ]; then
    echo "=== Preprocessing done at $(date) — submitting training ==="
    sbatch $REPO/train_a2d2_h100.sh
else
    echo "=== Preprocessing FAILED at $(date) ==="
    exit 1
fi
