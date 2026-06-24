#!/bin/bash
#SBATCH --job-name=pp_video
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/cluster/datastore/aniketag/mmdetection3d/logs/gen_video_pp_%j.log
#SBATCH --error=/cluster/datastore/aniketag/mmdetection3d/logs/gen_video_pp_%j.log

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
unset PYTHONPATH

REPO=/cluster/datastore/aniketag/mmdetection3d
PYTHON=/cluster/datastore/aniketag/allEnv/mmdet3d/bin/python

cd $REPO
echo "Generating PointPillars prediction video from epoch_1000.pth on $(hostname)..."

$PYTHON tools/make_pred_video.py \
  --config configs/pointpillars/pointpillars_a2d2.py \
  --checkpoint work_dirs/pointpillars_a2d2/epoch_1000.pth \
  --epoch 1000 \
  --data-root data/a2d2/ \
  --num-frames 60 \
  --out-dir outputs/pred_videos_pp \
  --score-thr 0.3

echo "Done: $(date)"
ls -lh outputs/pred_videos_pp/pred_epoch_1000.mp4 2>/dev/null || echo "MP4 not found"
