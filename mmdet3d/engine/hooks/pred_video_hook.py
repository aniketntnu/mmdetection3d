"""Hook that generates a prediction video after specific training epochs."""
import os
import subprocess
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class PredVideoHook(Hook):
    # Priority 95 > VERY_LOW (90) so this runs AFTER CheckpointHook saves epoch_N.pth
    priority = 95
    """Generate a prediction video on the val set after key epochs.

    Submits a self-contained sbatch job so it never conflicts with training.

    Args:
        script (str): Path to make_pred_video.py.
        config (str): Path to the detector config used for inference.
        data_root (str): Absolute path to the dataset root.
        num_frames (int): Number of val frames to render.
        score_thr (float): Score threshold for predictions.
        out_dir (str): Output directory for videos.
        python (str): Python binary to use inside the video job.
        partition (str): SLURM partition for the video job.
        epochs (list[int]): Explicit list of epochs to fire at.
            If None, uses default schedule: 5, 25, 50, 75, 100, 200,
            400, 600, 800, 1000.
    """

    def __init__(self,
                 script,
                 config,
                 data_root,
                 num_frames=60,
                 score_thr=0.3,
                 out_dir=None,
                 python='/cluster/datastore/aniketag/allEnv/mmdet3d/bin/python',
                 partition='h100',
                 nodelist=None,
                 epochs=None):
        self.script     = script
        self.config     = config
        self.data_root  = data_root
        self.num_frames = num_frames
        self.score_thr  = score_thr
        self.out_dir    = out_dir or os.path.join(
            os.path.dirname(config), '../../outputs/pred_videos')
        self.python     = python
        self.partition  = partition
        self.nodelist   = nodelist
        self._epochs    = set(epochs) if epochs is not None else {
            5, 25, 50, 75, 100, 200, 400, 600, 800, 1000}

    def _should_run(self, epoch):
        return epoch in self._epochs

    def after_train_epoch(self, runner):
        epoch = runner.epoch + 1          # 1-indexed
        if not self._should_run(epoch):
            return

        ckpt = os.path.join(runner.work_dir, f'epoch_{epoch}.pth')
        if not os.path.exists(ckpt):
            last = os.path.join(runner.work_dir, 'last_checkpoint')
            if os.path.exists(last):
                with open(last) as f:
                    candidate = f.read().strip()
                # Only use if it belongs to this work_dir (not a previous run)
                if candidate.startswith(runner.work_dir) and os.path.exists(candidate):
                    ckpt = candidate

        if not os.path.exists(ckpt):
            runner.logger.warning(
                f'PredVideoHook: checkpoint not found for epoch {epoch}, skip.')
            return

        log_path = os.path.join(self.out_dir, f'video_epoch_{epoch:03d}.log')
        os.makedirs(self.out_dir, exist_ok=True)

        nodelist_line = f'#SBATCH --nodelist={self.nodelist}\n' if self.nodelist else ''
        sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=pred_vid_{epoch:03d}
#SBATCH --partition={self.partition}
{nodelist_line}#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output={log_path}
#SBATCH --error={log_path}
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd /cluster/datastore/aniketag/mmdetection3d
{self.python} {self.script} \\
  --config {self.config} \\
  --checkpoint {ckpt} \\
  --epoch {epoch} \\
  --data-root {self.data_root} \\
  --num-frames {self.num_frames} \\
  --out-dir {self.out_dir} \\
  --score-thr {self.score_thr}
echo "Video job done: $(date)"
"""
        sbatch_path = os.path.join(self.out_dir, f'sbatch_epoch_{epoch:03d}.sh')
        with open(sbatch_path, 'w') as f:
            f.write(sbatch_script)

        runner.logger.info(
            f'PredVideoHook: submitting SLURM video job for epoch {epoch}')
        try:
            result = subprocess.run(
                ['sbatch', sbatch_path],
                capture_output=True, text=True)
            runner.logger.info(f'PredVideoHook: {result.stdout.strip()}')
        except Exception as e:
            runner.logger.warning(f'PredVideoHook: sbatch failed: {e}')
