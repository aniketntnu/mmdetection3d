# A2D2 3D Object Detection — PV-RCNN & PointPillars

Training PV-RCNN and PointPillars on the Audi A2D2 dataset for 5-class 3D LiDAR object detection.

---

## Table of Contents
1. [Dataset Overview](#1-dataset-overview)
2. [Environment Setup](#2-environment-setup)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Code Changes Made](#4-code-changes-made)
5. [Training PV-RCNN](#5-training-pv-rcnn)
6. [Training PointPillars](#6-training-pointpillars)
7. [Generating Prediction Videos](#7-generating-prediction-videos)
8. [Results](#8-results)

---

## 1. Dataset Overview

**A2D2** (Audi Autonomous Driving Dataset) — real driving footage from German cities.

- **Download**: `wget https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic_bboxes.tar`
- **Size**: ~48GB tar (skip ~33GB semantic masks during extraction)
- **Frames**: 12,497 total across 18 sequences
- **Split**: last 10% of each sequence = val → **11,236 train / 1,261 val**

### LiDAR Coordinate Frame
A2D2 LiDAR is **already in camera reference frame** — identical to KITTI LiDAR convention:
- `x` = forward, `y` = left, `z` = up
- **No coordinate transformation needed**

### 5-Class Mapping
| Label | Classes Merged |
|-------|---------------|
| 0 — Car | Car, VanSUV |
| 1 — Truck | Truck, UtilityVehicle, Trailer, CaravanTransporter |
| 2 — Bus | Bus |
| 3 — Pedestrian | Pedestrian |
| 4 — Cyclist | Cyclist, Bicycle, MotorBiker, Motorcycle |

### Box Format
A2D2 annotations use **geometric center + axis-angle rotation**.
mmdet3d requires **bottom-center + yaw**:
```python
cz_bottom = center_z - height / 2.0   # geometric center → bottom
yaw = rot_angle                        # rotation angle around z-axis
```

---

## 2. Environment Setup

### Cluster Requirements
- **GPU node for training**: hpc10 (2× A100 80GB) or hpc11 (2× H100 80GB)
- **GPU node for inference/video**: hpc11 (H100) — mmcv compiled for sm_80+
- **Do NOT use hpc1-8 (RTX 8000)** for PV-RCNN inference — wrong CUDA arch (sm_75)
- **Do NOT use login node** — missing GLIBC_2.32

### PV-RCNN Environment (`allEnv/mmdet3d`)
```bash
# Python 3.8, PyTorch 2.4+cu124, mmcv 2.1.0 (built from source on GPU node)
# mmcv must be built from source because GPU nodes have CUDA 12.6
# Pre-built cu118/cu121 wheels will NOT work

export PYTHONNOUSERSITE=1
unset PYTHONPATH   # ALWAYS required — stale ~/.local torch will interfere
```

### PointPillars Environment (`allEnv/mmdet3d_normal`)
```bash
# Separate environment built on normal partition (RTX 8000, sm_75)
# PointPillars runs on hpc1-8 (RTX 8000) — simpler ops, compatible with sm_75
export PYTHONNOUSERSITE=1
unset PYTHONPATH
```

---

## 3. Data Preprocessing

Run once to prepare all data (extraction + conversion + PKL creation + GT video):

```bash
sbatch prepare_a2d2.sh
```

This script (`tools/prepare_a2d2.py`) does:

### Step 1 — Extract tar (skip 33GB semantic masks)
```bash
tar -xf camera_lidar_semantic_bboxes.tar \
    --exclude=camera_lidar_semantic_bboxes/*/label/ \
    -C data/a2d2/raw/
```

### Step 2 — Convert NPZ → BIN (parallel, 16 workers)
Each LiDAR `.npz` file → KITTI-style binary float32 `[x, y, z, reflectance]`:
```python
pts = d["points"].astype(np.float32)
refl = d["reflectance"].astype(np.float32) / 100.0
np.hstack([pts, refl[:, None]]).tofile(bin_out)
```

### Step 3 — Build annotation PKLs
Creates `data/a2d2/a2d2_infos_train.pkl` and `data/a2d2/a2d2_infos_val.pkl`.
Each entry format:
```python
{
    "lidar_points": {"lidar_path": "lidar/{stem}.bin", "num_pts_dim": 4},
    "images": {"CAM2": {"img_path": "images/{stem}.png"}},
    "instances": [{
        "bbox_3d": [cx, cy, cz_bottom, l, w, h, yaw],
        "bbox_label_3d": 0-4,
        "truncated": float, "occluded": int,
    }]
}
```
> **Note**: Paths are relative to `data_root = data/a2d2/` — do NOT include `data/a2d2/` prefix,
> mmengine prepends it automatically.

### Step 4 — GT Verification Video
Renders GT 3D boxes onto camera images (20 frames) to verify annotations before training.
Output: `outputs/gt_verification_h264.avi`
```bash
scp user@server:/path/mmdetection3d/outputs/gt_verification_h264.avi .
```
**Always check this video first** — it confirms coordinate frames and box format are correct.

---

## 4. Code Changes Made

All changes are required to adapt mmdet3d for A2D2. **Do not revert these.**

### New Files Added
| File | Purpose |
|------|---------|
| `mmdet3d/datasets/a2d2_dataset.py` | Custom dataset class for A2D2 |
| `mmdet3d/engine/hooks/pred_video_hook.py` | Auto-generate prediction videos via SLURM sbatch |
| `mmdet3d/engine/hooks/nan_loss_skip_hook.py` | NaN loss detection hook |
| `configs/pv_rcnn/pv_rcnn_a2d2.py` | PV-RCNN config for A2D2 |
| `configs/pointpillars/pointpillars_a2d2.py` | PointPillars config for A2D2 |
| `tools/prepare_a2d2.py` | Data preprocessing pipeline |
| `tools/make_pred_video.py` | Inference + video rendering script |
| `train_a2d2_a100.sh` | PV-RCNN SLURM training script |
| `test_debug.sh` | 16-sample debug run (catches startup errors) |
| `test_256.sh` | 256-sample debug run (catches NaN at step 100) |
| `rebuild_pkl.sh` | Rebuild PKL after class map changes |
| `gen_video.sh` | Manual video generation script |

### Modified Files

#### `mmdet3d/datasets/__init__.py`
Registered `A2D2Dataset` in the dataset registry.

#### `mmdet3d/engine/hooks/__init__.py`
Registered `PredVideoHook` in the hooks registry.

#### `mmdet3d/structures/ops/transforms.py` — `bbox3d2roi()`
**Bug**: When all proposals are empty, original code: `rois = torch.zeros_like(bboxes)` → produces
`[0, 7]` tensor (missing batch column). Then `rois[:, 1:]` has only 6 columns → IndexError on yaw.

**Fix**:
```python
# Old (wrong):
rois = torch.zeros_like(bboxes)
# New (correct):
rois = bboxes.new_zeros((0, bboxes.shape[-1] + 1))
```

#### `mmdet3d/models/detectors/pv_rcnn.py` — `loss()`
**Bug**: NaN loss from one bad batch cascades → NaN weights → permanent NaN.

**Fix**: Zero out any NaN/Inf losses before backward:
```python
for k, v in list(losses.items()):
    if isinstance(v, torch.Tensor) and not v.isfinite().all():
        losses[k] = v.new_tensor(0.0)
```

#### `mmdet3d/models/roi_heads/pv_rcnn_roi_head.py`
**Bug 1**: `_assign_and_sample` uses `InstanceData[bool_mask]` which crashes when
InstanceData has fields with inconsistent shapes.

**Fix**: Build InstanceData subsets manually:
```python
pred_inds = pred_per_cls.nonzero(as_tuple=False).view(-1)
cls_proposals = InstanceData()
cls_proposals.bboxes_3d = cur_boxes[pred_inds]
```

**Bug 2**: When all proposals are empty (`rois.size(0) == 0`), CUDA kernel receives
0-size grid → crash.

**Fix**:
```python
if rois.size(0) == 0:
    zero = rois.new_tensor(0.0)
    return dict(loss_bbox=dict(loss_cls=zero, loss_bbox=zero, loss_corner=zero))
```

#### `mmdet3d/models/roi_heads/bbox_heads/pv_rcnn_bbox_head.py`
**Bug**: Corner loss can explode to billions (e.g., 6.9B) when predicted boxes are
far from GT → gradient explosion → NaN weights.

**Fix**: Clamp corner loss per box:
```python
pred_boxes3d = pred_boxes3d.clamp(-100.0, 100.0)
loss_corner = get_corner_loss_lidar(pred_boxes3d, pos_gt_bboxes)
loss_corner = loss_corner.clamp(max=20.0)  # max 20m absolute error
loss_corner = torch.nan_to_num(loss_corner, nan=0.0, posinf=0.0, neginf=0.0)
```

#### `mmdet3d/models/dense_heads/anchor3d_head.py`
**Bug**: NaN regression targets from degenerate anchor-GT pairs.

**Fix**:
```python
pos_bbox_pred = torch.nan_to_num(pos_bbox_pred, nan=0.0, posinf=10.0, neginf=-10.0)
pos_bbox_targets = torch.nan_to_num(pos_bbox_targets, nan=0.0, posinf=10.0, neginf=-10.0)
```

#### `mmdet3d/models/roi_heads/mask_heads/foreground_segmentation_head.py`
**Bug**: NaN model weights cause `sigmoid(NaN) = NaN` → `binary_cross_entropy(NaN)`
triggers CUDA assert → crash.

**Fix**:
```python
seg_preds = torch.sigmoid(seg_preds)
seg_preds = torch.nan_to_num(seg_preds, nan=0.5, posinf=1.0-1e-6, neginf=1e-6)
```

#### `mmdet3d/evaluation/metrics/kitti_metric.py`
**Bug**: KittiMetric expects `{'data_list': [...], 'metainfo': {...}}` but A2D2 PKL
is a plain Python list.

**Fix**: Handle both formats:
```python
if isinstance(data_infos, list):
    data_annos = data_infos
    label2cat = {0: 'Car', 1: 'Truck', 2: 'Bus', 3: 'Pedestrian', 4: 'Cyclist'}
else:
    data_annos = data_infos['data_list']
    ...
```
Also added `.get()` defaults for missing A2D2 fields: `alpha`, `score`, `height`, `width`.

---

## 5. Training PV-RCNN

### Architecture
PV-RCNN (PointVoxel-RCNN) — two-stage 3D detector:
- **Stage 1 (RPN)**: VoxelNet + sparse 3D conv + Anchor3DHead → region proposals
- **Stage 2 (ROI)**: VoxelSetAbstraction → keypoint features → PVRCNNBBoxHead → refined boxes

### Key Config Settings
```python
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
voxel_size = [0.05, 0.05, 0.1]
sparse_shape = [41, 1600, 1408]   # MUST match: z=(4m/0.1)+1=41, y=80/0.05=1600, x=70.4/0.05=1408
batch_size = 16  # per GPU, 2 GPUs = 32 total
lr = 0.001       # AdamW. Do NOT scale with batch size — tested 0.004 caused NaN
max_epochs = 1000
grad_clip = dict(max_norm=5)
model_wrapper_cfg = dict(find_unused_parameters=True)
val_interval = 9999  # disabled — KittiMetric needs lidar2cam calibration not in A2D2 PKL
```

### Anchor Sizes (measured from A2D2 data, capped to fit z_range)
```python
# [L, W, H]          anchor_z (cz_bottom)   note
[3.95, 2.00, 1.71],  # Car        z=-0.6    top=1.11m ✓
[9.30, 3.09, 2.50],  # Truck      z=-1.5    H capped: full 3.83m exceeds z_max=1
[8.55, 2.98, 2.30],  # Bus        z=-1.3    H capped: full 3.10m exceeds z_max=1
[0.97, 0.77, 1.82],  # Pedestrian z=-0.6
[1.87, 0.81, 1.64],  # Cyclist    z=-1.78
```
> **Important**: Anchor top = anchor_z + anchor_h must be ≤ point_cloud_range z_max (1.0m).
> If anchors extend above the voxel grid, regression targets become degenerate → NaN.

### LR Schedule
- Epochs 0→100: cosine warmup from `lr` to `lr×10`
- Epochs 100→1000: cosine decay from `lr×10` to `lr×1e-4`

### Debug Before Full Training (mandatory)
```bash
sbatch test_debug.sh    # 16 samples, ~2 min — catches import/shape errors
sbatch test_256.sh      # 256 samples, ~5 min — catches NaN at step 100+
```

### Full Training
```bash
# First time:
sbatch train_a2d2_a100.sh

# Resume after interruption (auto-detects latest checkpoint):
# --resume flag is already in train_a2d2_a100.sh
sbatch train_a2d2_a100.sh
```

Training script runs on **a100 partition** (hpc10, 2× A100 80GB), 10-day time limit.

### Checkpoints
Saved every 20 epochs, 3 kept: `work_dirs/pv_rcnn_a2d2/epoch_*.pth`
Final model: `work_dirs/pv_rcnn_a2d2/epoch_1000.pth` (226MB)

---

## 6. Training PointPillars

### Architecture
PointPillars — single-stage 3D detector:
- Pillars (voxels with z collapsed) → PillarFeatureNet → 2D pseudo-image → SECOND backbone → SSD head

### Key Config Settings
```python
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
voxel_size = [0.16, 0.2, 4]    # 4m z = full height in one pillar
# Output grid: y=(80/0.2)=400, x=(70.4/0.16)=440  — both divisible by 8 (required by SECONDFPN)
batch_size = 16   # per GPU
lr = 0.001
max_epochs = 1000
```

### Environment — Use `mmdet3d_normal`
PointPillars runs on the **normal partition** (hpc1-8, RTX 8000):
```bash
PYTHON=/cluster/datastore/aniketag/allEnv/mmdet3d_normal/bin/python
```
RTX 8000 (sm_75) supports PointPillars' simpler CUDA ops but NOT PV-RCNN's sparse conv ops.

### Training
```bash
# PointPillars uses its own SLURM script targeting normal partition
sbatch train_a2d2_pp.sh    # runs on hpc1-8 with mmdet3d_normal env
```

### Checkpoints
Final model: `work_dirs/pointpillars_a2d2/epoch_1000.pth`

---

## 7. Generating Prediction Videos

Videos show model predictions on validation frames (60 frames, 1fps MP4).
Colors: Green=Car, Orange/yellow=Truck, Blue=Bus, Blue=Pedestrian, Magenta=Cyclist.
Confidence scores shown on each box.

### Auto-generation (built into training)
`PredVideoHook` fires automatically at epochs 50, 100, 200, 300, ... 1000.
Submits a dedicated SLURM job to **h100 partition** (hpc11) for each video.
Output: `outputs/pred_videos/pred_epoch_XXX.mp4`

> **Note**: Video jobs use hpc11 (H100) to avoid CUDA conflict with training on hpc10 (A100).
> Do not change partition to a100 — both GPUs are occupied by training.

### Manual generation
```bash
# Edit gen_video.sh to set desired checkpoint epoch, then:
sbatch gen_video.sh
```

### Download
```bash
scp user@server:/path/mmdetection3d/outputs/pred_videos/pred_epoch_1000.mp4 .
scp user@server:/path/mmdetection3d/outputs/gt_verification_h264.avi .
```

---

## 8. Results

### Model Downloads (GitHub Release v1.0-models)

| Model | Size | Download |
|-------|------|----------|
| PV-RCNN (epoch 1000) | **226 MB** | [pvrcnn_a2d2_epoch1000.pth](https://github.com/aniketntnu/mmdetection3d/releases/download/v1.0-models/pvrcnn_a2d2_epoch1000.pth) |
| PointPillars (epoch 1000) | **73 MB** | [pointpillars_a2d2_epoch1000.pth](https://github.com/aniketntnu/mmdetection3d/releases/download/v1.0-models/pointpillars_a2d2_epoch1000.pth) |

Release page: https://github.com/aniketntnu/mmdetection3d/releases/tag/v1.0-models

### Training Logs
Full training logs in `training_logs/`:
- `training_logs/pvrcnn_a2d2_1000epochs.log` — PV-RCNN (2.4MB)
- `training_logs/pointpillars_a2d2_1000epochs.log` — PointPillars (765KB)

### PV-RCNN on A2D2 (1000 epochs)
| Epoch | Loss |
|-------|------|
| 1 | 7.03 |
| 50 | ~2.5 |
| 100 | ~2.0 |
| 500 | ~1.2 |
| 1000 | **0.644** |

### Videos Available
- `outputs/pred_videos/pred_epoch_040.mp4` — early model (epoch 40)
- `outputs/pred_videos/pred_epoch_680.mp4` — mid-training (epoch 680)
- `outputs/pred_videos/pred_epoch_1000.mp4` — final model (epoch 1000)
- `outputs/gt_verification_h264.avi` — GT annotations (ground truth reference)
