# A2D2 3D Object Detection — PV-RCNN & PointPillars

Training and inference for PV-RCNN and PointPillars on the Audi A2D2 dataset.
**5 classes**: Car, Truck, Bus, Pedestrian, Cyclist — trained for 1000 epochs.

---

## Downloads (GitHub Release v1.0-models)

| Asset | Size | Link |
|-------|------|------|
| PV-RCNN model (epoch 1000) | 226 MB | [pvrcnn_a2d2_epoch1000.pth](https://github.com/aniketntnu/mmdetection3d/releases/download/v1.0-models/pvrcnn_a2d2_epoch1000.pth) |
| PointPillars model (epoch 1000) | 73 MB | [pointpillars_a2d2_epoch1000.pth](https://github.com/aniketntnu/mmdetection3d/releases/download/v1.0-models/pointpillars_a2d2_epoch1000.pth) |
| PV-RCNN prediction video | 32 MB | [pvrcnn_pred_epoch1000.mp4](https://github.com/aniketntnu/mmdetection3d/releases/download/v1.0-models/pvrcnn_pred_epoch1000.mp4) |
| PointPillars prediction video | 32 MB | [pointpillars_pred_epoch1000.mp4](https://github.com/aniketntnu/mmdetection3d/releases/download/v1.0-models/pointpillars_pred_epoch1000.mp4) |

Training logs: [`training_logs/`](training_logs/)

---

## Table of Contents
1. [Dataset Overview](#1-dataset-overview)
2. [Environment Setup](#2-environment-setup)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Code Changes Made](#4-code-changes-made)
5. [Training PV-RCNN](#5-training-pv-rcnn)
6. [Training PointPillars](#6-training-pointpillars)
7. [Inference & Prediction Videos](#7-inference--prediction-videos)
8. [Results](#8-results)

---

## 1. Dataset Overview

**A2D2** (Audi Autonomous Driving Dataset) — real driving footage from German cities.

- **Download**: `wget https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic_bboxes.tar`
- **Size**: ~48GB tar (skip ~33GB semantic masks during extraction)
- **Frames**: 12,497 total across 18 sequences
- **Split**: last 10% of each sequence = val → **11,236 train / 1,261 val**
- **Tutorial**: https://www.a2d2.audi/en/tutorial/

### LiDAR Coordinate Frame
A2D2 LiDAR is **already in camera reference frame** — identical to KITTI convention:
- `x` = forward, `y` = left, `z` = up — **No coordinate transformation needed**

### 5-Class Mapping
| Label | Classes Merged | Approx. Count (full dataset) |
|-------|---------------|-------------------------------|
| 0 — Car | Car, VanSUV | ~25,000 |
| 1 — Truck | Truck, UtilityVehicle, Trailer, CaravanTransporter | ~7,700 |
| 2 — Bus | Bus | ~580 |
| 3 — Pedestrian | Pedestrian | ~3,500 |
| 4 — Cyclist | Cyclist, Bicycle, MotorBiker, Motorcycle | ~900 |

### Box Format Conversion
A2D2 raw JSON uses **geometric center + axis-angle rotation**.
mmdet3d requires **bottom-center + yaw** (KITTI format):
```python
cz_bottom = center_z - height / 2.0   # geometric center → bottom of box
yaw = rot_angle                        # rotation angle around z-axis
```

---

## 2. Environment Setup

### Cluster Requirements
| Node | GPU | Partition | Use |
|------|-----|-----------|-----|
| hpc10 | 2× A100 80GB | a100 | PV-RCNN training |
| hpc11 | 2× H100 80GB | h100 | Inference / video generation |
| hpc1-8 | 2× RTX 8000 | normal | PointPillars training |

> **Warning**: Do NOT run PV-RCNN inference on hpc1-8 (RTX 8000) — mmcv is compiled for
> sm_80+ (A100/H100), RTX 8000 is sm_75 → `CUDA error: no kernel image available`.
> Do NOT run on the login node — missing GLIBC_2.32.

### PV-RCNN Environment (`allEnv/mmdet3d`)
```bash
# Python 3.8, PyTorch 2.4+cu124
# mmcv 2.1.0 built from source on GPU node (CUDA 12.6)
# Pre-built cu118/cu121 wheels will NOT work on this cluster

export PYTHONNOUSERSITE=1  # REQUIRED — prevents stale ~/.local torch interference
unset PYTHONPATH           # REQUIRED
```

Build mmcv from source (if needed):
```bash
sbatch build_mmcv.sh   # runs on hpc10/hpc11, takes ~30 min
```

### PointPillars Environment (`allEnv/mmdet3d_normal`)
```bash
# Separate environment built on normal partition (RTX 8000, sm_75)
# PointPillars uses simpler CUDA ops compatible with sm_75
export PYTHONNOUSERSITE=1
unset PYTHONPATH
PYTHON=/cluster/datastore/aniketag/allEnv/mmdet3d_normal/bin/python
```

---

## 3. Data Preprocessing

Run **once** to prepare all data. Creates BIN files, annotation PKLs, and GT verification video:

```bash
sbatch prepare_a2d2.sh
```

This runs `tools/prepare_a2d2.py` which does:

### Step 1 — Extract tar (skip 33GB semantic masks)
```bash
tar -xf camera_lidar_semantic_bboxes.tar \
    --exclude=camera_lidar_semantic_bboxes/*/label/ \
    -C data/a2d2/raw/
```

### Step 2 — Convert LiDAR NPZ → BIN (16 workers in parallel)
```python
# data/a2d2/lidar/{sequence}_{timestamp}.bin
# Format: [x, y, z, reflectance] float32, N points
pts = d["points"].astype(np.float32)
refl = d["reflectance"].astype(np.float32) / 100.0
np.hstack([pts, refl[:, None]]).tofile(bin_out)
```

### Step 3 — Build annotation PKLs
- `data/a2d2/a2d2_infos_train.pkl` — 11,236 frames
- `data/a2d2/a2d2_infos_val.pkl` — 1,261 frames

Each entry stores `bbox_3d = [cx, cy, cz_bottom, l, w, h, yaw]` in mmdet3d bottom-center format.

> **Important**: Paths stored as `lidar/{stem}.bin` (relative to `data_root = data/a2d2/`).
> Do NOT include `data/a2d2/` prefix — mmengine prepends it automatically.

### Step 4 — GT Verification Video
Renders GT 3D boxes onto camera images to verify annotations before training.
Output: `outputs/gt_verification_h264.avi`

**Always check this video first** before training — confirms coordinate frames and box format.

### To rebuild PKL after class changes:
```bash
sbatch rebuild_pkl.sh
```

---

## 4. Code Changes Made

All changes adapt mmdet3d for A2D2. Do not revert.

### New Files
| File | Purpose |
|------|---------|
| `mmdet3d/datasets/a2d2_dataset.py` | Custom dataset class — 5-class, plain-list PKL, LiDARInstance3DBoxes wrapping |
| `mmdet3d/engine/hooks/pred_video_hook.py` | Auto-generates prediction videos via SLURM sbatch at epoch milestones |
| `mmdet3d/engine/hooks/nan_loss_skip_hook.py` | NaN loss detection utility hook |
| `configs/pv_rcnn/pv_rcnn_a2d2.py` | PV-RCNN 5-class config for A2D2 |
| `configs/pointpillars/pointpillars_a2d2.py` | PointPillars 5-class config for A2D2 |
| `tools/prepare_a2d2.py` | Full data preprocessing pipeline |
| `tools/make_pred_video.py` | Inference + 3D box projection + MP4 rendering |
| `train_a2d2_a100.sh` | PV-RCNN SLURM training script (a100 partition, 2 GPUs, `--resume`) |
| `gen_video.sh` | Manual PV-RCNN video generation (h100 partition) |
| `gen_video_pp.sh` | Manual PointPillars video generation (h100 partition) |
| `test_debug.sh` | 16-sample debug run — catches startup errors (~2 min) |
| `test_256.sh` | 256-sample debug run — catches NaN at step 100+ (~5 min) |
| `rebuild_pkl.sh` | Rebuilds annotation PKLs after class map changes |

### Modified Files

#### `mmdet3d/datasets/__init__.py`
Registered `A2D2Dataset`.

#### `mmdet3d/engine/hooks/__init__.py`
Registered `PredVideoHook`.

#### `mmdet3d/structures/ops/transforms.py` — `bbox3d2roi()`
**Bug**: Empty proposals produced `[0,7]` tensor (missing batch column) → `rois[:,1:]` had
only 6 columns → IndexError accessing yaw (column 6).
```python
# Fix:
rois = bboxes.new_zeros((0, bboxes.shape[-1] + 1))  # was: torch.zeros_like(bboxes)
```

#### `mmdet3d/models/detectors/pv_rcnn.py` — `loss()`
**Bug**: One bad batch with NaN loss → NaN gradients → NaN model weights → permanent NaN.
```python
# Fix: zero out NaN/Inf losses before backward pass
for k, v in list(losses.items()):
    if isinstance(v, torch.Tensor) and not v.isfinite().all():
        losses[k] = v.new_tensor(0.0)
```

#### `mmdet3d/models/roi_heads/pv_rcnn_roi_head.py`
**Bug 1**: `InstanceData[bool_mask]` crashes when fields have inconsistent shapes.
```python
# Fix: build subsets manually
pred_inds = pred_per_cls.nonzero(as_tuple=False).view(-1)
cls_proposals = InstanceData()
cls_proposals.bboxes_3d = cur_boxes[pred_inds]
```
**Bug 2**: CUDA kernel crash when `rois.size(0) == 0`.
```python
# Fix:
if rois.size(0) == 0:
    zero = rois.new_tensor(0.0)
    return dict(loss_bbox=dict(loss_cls=zero, loss_bbox=zero, loss_corner=zero))
```

#### `mmdet3d/models/roi_heads/bbox_heads/pv_rcnn_bbox_head.py`
**Bug**: Corner loss explodes to billions → gradient explosion → NaN weights.
```python
# Fix: clamp corner loss
pred_boxes3d = pred_boxes3d.clamp(-100.0, 100.0)
loss_corner = get_corner_loss_lidar(pred_boxes3d, pos_gt_bboxes)
loss_corner = loss_corner.clamp(max=20.0)
loss_corner = torch.nan_to_num(loss_corner, nan=0.0, posinf=0.0, neginf=0.0)
```

#### `mmdet3d/models/dense_heads/anchor3d_head.py`
**Bug**: NaN regression targets from degenerate anchor-GT pairs.
```python
# Fix:
pos_bbox_pred = torch.nan_to_num(pos_bbox_pred, nan=0.0, posinf=10.0, neginf=-10.0)
pos_bbox_targets = torch.nan_to_num(pos_bbox_targets, nan=0.0, posinf=10.0, neginf=-10.0)
```

#### `mmdet3d/models/roi_heads/mask_heads/foreground_segmentation_head.py`
**Bug**: `sigmoid(NaN)` → CUDA `binary_cross_entropy` assert → crash.
```python
# Fix:
seg_preds = torch.sigmoid(seg_preds)
seg_preds = torch.nan_to_num(seg_preds, nan=0.5, posinf=1.0-1e-6, neginf=1e-6)
```

#### `mmdet3d/evaluation/metrics/kitti_metric.py`
**Bug**: KittiMetric expects mmengine dict format; A2D2 PKL is a plain list.
Also missing fields: `alpha`, `score`, `height`, `width`, `lidar2cam`.
```python
# Fix: handle both formats with defaults for missing fields
if isinstance(data_infos, list):
    data_annos = data_infos
    label2cat = {0:'Car', 1:'Truck', 2:'Bus', 3:'Pedestrian', 4:'Cyclist'}
# Use .get() with defaults for all missing A2D2 fields
```

---

## 5. Training PV-RCNN

### Debug First (mandatory)
```bash
sbatch test_debug.sh   # 16 samples, ~2 min — catches import/shape errors
sbatch test_256.sh     # 256 samples, ~5 min — catches NaN at steps 50-100
```

### Full Training
```bash
sbatch train_a2d2_a100.sh
```
- Partition: `a100` (hpc10, 2× A100 80GB)
- Time limit: 10 days
- Automatically resumes from latest checkpoint with `--resume`

### Key Config (`configs/pv_rcnn/pv_rcnn_a2d2.py`)
```python
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
voxel_size = [0.05, 0.05, 0.1]
sparse_shape = [41, 1600, 1408]   # z=(4m/0.1)+1, y=80/0.05, x=70.4/0.05
batch_size = 16                    # per GPU → 32 total
lr = 0.001                        # AdamW — do NOT scale with batch size
max_epochs = 1000
grad_clip = dict(max_norm=5)
model_wrapper_cfg = dict(find_unused_parameters=True)
val_interval = 9999               # disabled — KittiMetric needs lidar2cam
```

### Anchor Sizes (measured from A2D2 data)
```python
# [L, W, H]           anchor_z    note
[3.95, 2.00, 1.71],   # z=-0.6   Car
[9.30, 3.09, 2.50],   # z=-1.5   Truck (H capped: full 3.83m > z_max=1m)
[8.55, 2.98, 2.30],   # z=-1.3   Bus   (H capped: full 3.10m > z_max=1m)
[0.97, 0.77, 1.82],   # z=-0.6   Pedestrian
[1.87, 0.81, 1.64],   # z=-1.78  Cyclist
```
> Rule: `anchor_z + anchor_H ≤ point_cloud_range z_max (1.0m)`. Violating this causes
> degenerate regression targets → NaN.

### Checkpoints
Saved every 20 epochs, 3 kept: `work_dirs/pv_rcnn_a2d2/epoch_*.pth`

---

## 6. Training PointPillars

### Full Training
```bash
sbatch train_pp_a2d2_normal.sh   # runs on normal partition (hpc1-8, RTX 8000)
```

### Key Config (`configs/pointpillars/pointpillars_a2d2.py`)
```python
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
voxel_size = [0.16, 0.2, 4]     # 4m z collapses entire height into one pillar
# Grid: y=(80/0.2)=400, x=(70.4/0.16)=440 — both divisible by 8 (SECONDFPN requirement)
batch_size = 16
lr = 0.001
max_epochs = 1000
```

### Checkpoints
Saved every 20 epochs: `work_dirs/pointpillars_a2d2/epoch_*.pth`

---

## 7. Inference & Prediction Videos

### Single Frame Inference (Python)
```python
from mmdet3d.apis import LiDAR3DInferencer

# PV-RCNN
inferencer = LiDAR3DInferencer(
    model='configs/pv_rcnn/pv_rcnn_a2d2.py',
    weights='work_dirs/pv_rcnn_a2d2/epoch_1000.pth'
)

# PointPillars
inferencer = LiDAR3DInferencer(
    model='configs/pointpillars/pointpillars_a2d2.py',
    weights='work_dirs/pointpillars_a2d2/epoch_1000.pth'
)

# Run inference on a BIN file
result = inferencer(inputs=dict(points='path/to/lidar.bin'), no_save_pred=False)
# result contains: bboxes_3d [N,7], labels_3d [N], scores_3d [N]
# bbox format: [cx, cy, cz_bottom, l, w, h, yaw]
```

> **Note**: Inference must run on hpc10 (A100) or hpc11 (H100) — not on login node
> or hpc1-8 (RTX 8000) — due to mmcv CUDA architecture requirements.

### Generate Prediction Video on Validation Set

**PV-RCNN:**
```bash
# Edit gen_video.sh to set checkpoint, then:
sbatch gen_video.sh
# Output: outputs/pred_videos/pred_epoch_1000.mp4
```

**PointPillars:**
```bash
sbatch gen_video_pp.sh
# Output: outputs/pred_videos_pp/pred_epoch_1000.mp4
```

Both scripts run on **h100 partition** (hpc11) with 1 GPU.
Videos show 60 validation frames with 3D boxes projected onto camera images.

### Manual Inference Script
`tools/make_pred_video.py` — runs inference on val frames and renders a video:
```bash
PYTHONNOUSERSITE=1 python tools/make_pred_video.py \
  --config configs/pv_rcnn/pv_rcnn_a2d2.py \
  --checkpoint work_dirs/pv_rcnn_a2d2/epoch_1000.pth \
  --epoch 1000 \
  --data-root data/a2d2/ \
  --num-frames 60 \
  --out-dir outputs/pred_videos \
  --score-thr 0.3
```

### 3D Box Projection to Camera
The video script uses a DLT-estimated projection matrix from LiDAR `col`/`row` correspondences:
```python
# Build A matrix from (X,Y,Z) → (u,v) correspondences
_, _, Vt = np.linalg.svd(A)
P = Vt[-1].reshape(3, 4)
P /= np.linalg.norm(P[2, :3])
if (P @ test_point)[2] < 0: P = -P   # ensure positive depth
```
Box corners use mmdet3d bottom-center convention:
```python
zs = [0, 1, 1, 0, 0, 1, 1, 0] * height   # 0=bottom, 1=top
```

### Score Threshold
Default `--score-thr 0.3`. Lower for more detections, higher for fewer false positives.

### Download Results
```bash
# PV-RCNN prediction video
scp user@server:mmdetection3d/outputs/pred_videos/pred_epoch_1000.mp4 .

# PointPillars prediction video
scp user@server:mmdetection3d/outputs/pred_videos_pp/pred_epoch_1000.mp4 .

# GT verification video
scp user@server:mmdetection3d/outputs/gt_verification_h264.avi .
```

---

## 8. Results

### Training Summary

| Model | Epochs | Final Loss | Hardware | Duration |
|-------|--------|-----------|----------|----------|
| PV-RCNN | 1000 | **0.644** | 2× A100 80GB | ~7.5 days |
| PointPillars | 1000 | — | 1× RTX 8000 | ~17 hrs |

### PV-RCNN Loss Curve
| Epoch | Loss |
|-------|------|
| 1 | 7.03 |
| 50 | ~2.5 |
| 100 | ~2.0 |
| 200 | ~1.7 |
| 500 | ~1.2 |
| 1000 | **0.644** |

### Key Findings
- A2D2 LiDAR is in camera frame — same as KITTI, no transformation needed
- Truck/Bus anchors must have heights capped to fit within `z_range = [-3, 1]`
- NaN loss from bad batches is inevitable — the zero-out fix in `pv_rcnn.loss()` is essential
- LR=0.001 with AdamW is stable; LR scaling with batch size causes NaN at epoch ~25
- Validation is disabled (`val_interval=9999`) because KittiMetric requires `lidar2cam`
  calibration matrices not present in the A2D2 PKL
