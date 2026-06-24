# A2D2 3D Object Detection — PV-RCNN & PointPillars

Training and inference for **PV-RCNN** and **PointPillars** on the Audi A2D2 dataset.
**5 classes**: Car, Truck, Bus, Pedestrian, Cyclist — trained for 1000 epochs each.

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
1. [Dataset Overview & Train/Val Split](#1-dataset-overview--trainval-split)
2. [Installation](#2-installation)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Code Changes Made](#4-code-changes-made)
5. [Training PV-RCNN](#5-training-pv-rcnn)
6. [Training PointPillars](#6-training-pointpillars)
7. [Inference & Prediction Videos](#7-inference--prediction-videos)
8. [Evaluating on Test Data](#8-evaluating-on-test-data)
9. [Results](#9-results)

---

## 1. Dataset Overview & Train/Val Split

**A2D2** (Audi Autonomous Driving Dataset) — real driving footage from German cities.

- **Download**: `wget https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic_bboxes.tar`
- **Total**: 12,497 frames across 18 sequences
- **Tutorial**: https://www.a2d2.audi/en/tutorial/

### Train/Val Split
Split strategy: **last 10% of each sequence → val**, rest → train.
This preserves temporal coherence within sequences.

| Sequence | Train frames | Val frames |
|----------|-------------|-----------|
| 20180807_145028 | 847 | 95 |
| 20180810_142822 | 506 | 57 |
| 20180925_101535 | 954 | 106 |
| 20180925_112730 | 856 | 96 |
| 20180925_124435 | 878 | 98 |
| 20180925_135056 | 622 | 70 |
| 20181008_095521 | 654 | 73 |
| 20181016_125231 | 803 | 90 |
| 20181107_132300 | 513 | 58 |
| 20181107_132730 | 730 | 82 |
| 20181107_133258 | 105 | 12 |
| 20181108_084007 | 578 | 65 |
| 20181108_091945 | 775 | 87 |
| 20181108_103155 | 465 | 52 |
| 20181108_123750 | 1154 | 129 |
| 20181204_135952 | 561 | 63 |
| 20181204_154421 | 135 | 16 |
| 20181204_170238 | 100 | 12 |
| **TOTAL** | **11,236** | **1,261** |

### LiDAR Coordinate Frame
A2D2 LiDAR is already in camera reference frame — same as KITTI:
- `x` = forward, `y` = left, `z` = up — **no transformation needed**

### 5-Class Mapping
| Label | Raw Classes Merged | Approx. Count |
|-------|-------------------|--------------|
| 0 — Car | Car, VanSUV | ~25,000 |
| 1 — Truck | Truck, UtilityVehicle, Trailer, CaravanTransporter | ~7,700 |
| 2 — Bus | Bus | ~580 |
| 3 — Pedestrian | Pedestrian | ~3,500 |
| 4 — Cyclist | Cyclist, Bicycle, MotorBiker, Motorcycle | ~900 |

### Box Format
A2D2 JSON uses geometric center + axis-angle. mmdet3d requires bottom-center + yaw:
```python
cz_bottom = center_z - height / 2.0
yaw = rot_angle
```

---

## 2. Installation

### Step 1 — Clone the Repository
```bash
git clone https://github.com/aniketntnu/mmdetection3d.git
cd mmdetection3d
```

### Step 2 — Update Hardcoded Paths
Two files have paths that must match your installation location:

**`tools/prepare_a2d2.py`** (lines 20-21):
```python
TAR_PATH = "/your/path/to/camera_lidar_semantic_bboxes.tar"
REPO     = Path("/your/path/to/mmdetection3d")
```

**`configs/pv_rcnn/pv_rcnn_a2d2.py`** and **`configs/pointpillars/pointpillars_a2d2.py`**:
```python
REPO = '/your/path/to/mmdetection3d'
```
Also update the Python interpreter path inside the PredVideoHook config block.

### Step 3 — Create PV-RCNN Environment (requires GPU node, CUDA 12.x)

```bash
# On a GPU node (A100 or H100):
python3.8 -m venv /path/to/allEnv/mmdet3d
source /path/to/allEnv/mmdet3d/bin/activate

# Install PyTorch 2.4+cu124
pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Install mmlab stack (pinned versions — do not change)
pip install mmengine==0.10.7
pip install mmdet==3.3.0
# mmcv MUST be built from source (prebuilt wheels may not match CUDA version)
MMCV_WITH_OPS=1 pip install "mmcv==2.1.0" --no-binary mmcv \
    -f https://download.openmmlab.com/mmcv/dist/cu124/torch2.4/index.html

# Install mmdet3d in dev mode (picks up all code changes)
pip install -e . --no-build-isolation

# Verify installation
python -c "import mmdet3d; import mmcv; print('OK')"
```

> **Note**: If `MMCV_WITH_OPS=1 pip install mmcv` fails, run `sbatch build_mmcv.sh`
> which compiles mmcv from source on a GPU node. This takes ~30 min.

### Step 4 — Create PointPillars Environment (login node, for RTX 8000)

PointPillars uses a separate environment because it runs on RTX 8000 (sm_75),
while the PV-RCNN environment is compiled for A100/H100 (sm_80+).

```bash
# From login node (no GPU needed for env creation):
bash setup_env_normal.sh   # creates allEnv/mmdet3d_normal

# Then build mmcv for RTX 8000 on a GPU node:
sbatch build_mmcv_normal.sh
```

### Step 5 — Verify Environment
```bash
bash test_install.sh   # runs GPU import + CUDA kernel check on a GPU node
```

### Cluster-specific Notes
| Component | Requirement |
|-----------|------------|
| PV-RCNN training | 2× A100 80GB (hpc10) or 2× H100 80GB (hpc11) |
| PointPillars training | RTX 8000 (hpc1-8, normal partition) |
| Inference / video | H100 (hpc11) — mmcv compiled for sm_80+ |
| Login node | Cannot run inference — missing GLIBC_2.32 |
| RTX 8000 | Cannot run PV-RCNN inference — wrong CUDA arch (sm_75 vs sm_80) |

Always set before any SLURM job:
```bash
export PYTHONNOUSERSITE=1
unset PYTHONPATH
```

---

## 3. Data Preprocessing

Run **once** to extract, convert, and create annotation files.

```bash
# Edit TAR_PATH and REPO in tools/prepare_a2d2.py first, then:
sbatch prepare_a2d2.sh
```

### What it does

**Step 1 — Extract (skip 33GB semantic masks):**
```bash
tar -xf camera_lidar_semantic_bboxes.tar \
    --exclude=camera_lidar_semantic_bboxes/*/label/ \
    -C data/a2d2/raw/
```

**Step 2 — NPZ → BIN conversion (16 workers):**
```python
# data/a2d2/lidar/{sequence}_{timestamp}.bin
# Format: [x, y, z, reflectance] float32, N points
pts  = d["points"].astype(np.float32)
refl = d["reflectance"].astype(np.float32) / 100.0
np.hstack([pts, refl[:, None]]).tofile(bin_out)
```

**Step 3 — Build annotation PKLs:**
Creates `data/a2d2/a2d2_infos_train.pkl` (11,236 frames) and
`data/a2d2/a2d2_infos_val.pkl` (1,261 frames).

Each entry: `bbox_3d = [cx, cy, cz_bottom, l, w, h, yaw]`.
Paths stored as `lidar/{stem}.bin` — relative to `data_root = data/a2d2/`
(mmengine prepends data_root automatically — do NOT add `data/a2d2/` prefix).

**Step 4 — GT Verification Video:**
Output: `outputs/gt_verification_h264.avi` — renders GT boxes on camera images.
**Always inspect this before training** to confirm coordinate frame and box format are correct.

### Create debug PKLs (needed for test_debug.sh and test_256.sh):
```bash
python - << 'EOF'
import pickle
with open('data/a2d2/a2d2_infos_train.pkl','rb') as f: data = pickle.load(f)
subset16  = [x for x in data if x['instances']][:16]
subset256 = [x for x in data if x['instances']][:256]
with open('data/a2d2/a2d2_infos_debug.pkl','wb') as f: pickle.dump(subset16, f)
with open('data/a2d2/a2d2_infos_256.pkl','wb') as f:   pickle.dump(subset256, f)
print('Created debug PKLs: 16 and 256 samples')
EOF
```

### Rebuild PKL after class changes:
```bash
sbatch rebuild_pkl.sh
```

---

## 4. Code Changes Made

All changes adapt mmdet3d for A2D2. Do not revert.

### New Files
| File | Purpose |
|------|---------|
| `mmdet3d/datasets/a2d2_dataset.py` | Custom dataset class — 5-class, plain-list PKL |
| `mmdet3d/engine/hooks/pred_video_hook.py` | Auto video generation via SLURM sbatch |
| `mmdet3d/engine/hooks/nan_loss_skip_hook.py` | NaN loss detection hook |
| `configs/pv_rcnn/pv_rcnn_a2d2.py` | PV-RCNN 5-class config |
| `configs/pointpillars/pointpillars_a2d2.py` | PointPillars 5-class config |
| `tools/prepare_a2d2.py` | Full data preprocessing |
| `tools/make_pred_video.py` | Inference + video rendering |
| `train_a2d2_a100.sh` | PV-RCNN SLURM training (2× A100, `--resume`) |
| `gen_video.sh` | PV-RCNN video generation (H100) |
| `gen_video_pp.sh` | PointPillars video generation (H100) |
| `test_debug.sh` | 16-sample debug (PV-RCNN) |
| `test_256.sh` | 256-sample debug (PV-RCNN) |
| `build_mmcv.sh` | Build mmcv from source on A100 |
| `setup_env_normal.sh` | Create PointPillars venv |
| `rebuild_pkl.sh` | Rebuild annotation PKLs |

### Modified Files — Bugs Fixed

#### `mmdet3d/structures/ops/transforms.py`
**Bug**: `bbox3d2roi()` produced `[0,7]` tensor for empty proposals (missing batch column).
`rois[:,1:]` had 6 columns → IndexError on yaw column access.
```python
# Fix:
rois = bboxes.new_zeros((0, bboxes.shape[-1] + 1))  # was: torch.zeros_like(bboxes)
```

#### `mmdet3d/models/detectors/pv_rcnn.py`
**Bug**: NaN from one bad batch cascaded to permanent NaN weights.
```python
# Fix: zero out NaN/Inf losses before backward
for k, v in list(losses.items()):
    if isinstance(v, torch.Tensor) and not v.isfinite().all():
        losses[k] = v.new_tensor(0.0)
```

#### `mmdet3d/models/roi_heads/pv_rcnn_roi_head.py`
**Bug 1**: `InstanceData[bool_mask]` crashed with inconsistent field shapes.
```python
# Fix: build subsets manually
pred_inds = pred_per_cls.nonzero(as_tuple=False).view(-1)
cls_proposals = InstanceData(); cls_proposals.bboxes_3d = cur_boxes[pred_inds]
```
**Bug 2**: CUDA kernel crash when `rois.size(0) == 0`.
```python
# Fix:
if rois.size(0) == 0:
    zero = rois.new_tensor(0.0)
    return dict(loss_bbox=dict(loss_cls=zero, loss_bbox=zero, loss_corner=zero))
```

#### `mmdet3d/models/roi_heads/bbox_heads/pv_rcnn_bbox_head.py`
**Bug**: Corner loss exploded to billions → gradient explosion → NaN weights.
```python
# Fix:
pred_boxes3d = pred_boxes3d.clamp(-100.0, 100.0)
loss_corner = get_corner_loss_lidar(pred_boxes3d, pos_gt_bboxes)
loss_corner = loss_corner.clamp(max=20.0)
loss_corner = torch.nan_to_num(loss_corner, nan=0.0, posinf=0.0, neginf=0.0)
```

#### `mmdet3d/models/dense_heads/anchor3d_head.py`
**Bug**: NaN regression targets from degenerate anchor-GT pairs.
```python
# Fix:
pos_bbox_pred    = torch.nan_to_num(pos_bbox_pred,    nan=0.0, posinf=10.0, neginf=-10.0)
pos_bbox_targets = torch.nan_to_num(pos_bbox_targets, nan=0.0, posinf=10.0, neginf=-10.0)
```

#### `mmdet3d/models/roi_heads/mask_heads/foreground_segmentation_head.py`
**Bug**: `sigmoid(NaN)` → CUDA binary_cross_entropy assert → crash.
```python
# Fix:
seg_preds = torch.sigmoid(seg_preds)
seg_preds = torch.nan_to_num(seg_preds, nan=0.5, posinf=1.0-1e-6, neginf=1e-6)
```

#### `mmdet3d/evaluation/metrics/kitti_metric.py`
**Bug**: KittiMetric expects `{'data_list':[...], 'metainfo':{...}}`; A2D2 PKL is a plain list.
Also missing fields: `alpha`, `score`, `height`, `width`.
```python
# Fix:
if isinstance(data_infos, list):
    data_annos = data_infos
    label2cat = {0:'Car', 1:'Truck', 2:'Bus', 3:'Pedestrian', 4:'Cyclist'}
# Use .get() with defaults for all missing A2D2 fields
```

---

## 5. Training PV-RCNN

### Debug First (mandatory — do not skip)
```bash
# Create debug PKLs first (see Section 3 above)
sbatch test_debug.sh   # 16 samples, ~2 min — catches import/shape errors
sbatch test_256.sh     # 256 samples, ~5 min — catches NaN at step 50-100
```

### Full Training
```bash
sbatch train_a2d2_a100.sh
```
- Partition: `a100` (2× A100 80GB), 10-day time limit
- `--resume` flag: auto-resumes from latest checkpoint if job is restarted
- Logs: `logs/train_a2d2_*.log`

### Key Config (`configs/pv_rcnn/pv_rcnn_a2d2.py`)
```python
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
voxel_size        = [0.05, 0.05, 0.1]
sparse_shape      = [41, 1600, 1408]   # MUST match: z=(4m/0.1)+1, y=80/0.05, x=70.4/0.05
batch_size        = 16                  # per GPU → 32 total with 2 GPUs
lr                = 0.001              # AdamW — do NOT scale with batch size
max_epochs        = 1000
grad_clip         = dict(max_norm=5)
model_wrapper_cfg = dict(find_unused_parameters=True)
val_interval      = 9999               # disabled — KittiMetric requires lidar2cam
```

### Anchor Sizes
```
Class        L      W      H      anchor_z   Note
Car         3.95   2.00   1.71    -0.6
Truck       9.30   3.09   2.50    -1.5       H capped (actual 3.83m > z_max=1m)
Bus         8.55   2.98   2.30    -1.3       H capped (actual 3.10m > z_max=1m)
Pedestrian  0.97   0.77   1.82    -0.6
Cyclist     1.87   0.81   1.64    -1.78
```
> Rule: `anchor_z + anchor_H ≤ z_max (1.0m)`. Anchors extending above the voxel grid
> produce degenerate regression targets → NaN.

### Checkpoints
Every 20 epochs, 3 kept: `work_dirs/pv_rcnn_a2d2/epoch_*.pth`

---

## 6. Training PointPillars

### Debug First
```bash
sbatch test_debug_pp.sh   # 16 samples
sbatch test_256_pp.sh     # 256 samples
```

### Full Training
```bash
sbatch train_pp_a2d2_normal.sh   # normal partition, RTX 8000
```

### Key Config (`configs/pointpillars/pointpillars_a2d2.py`)
```python
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
voxel_size        = [0.16, 0.2, 4]    # 4m z collapses full height into one pillar
# Grid: y=(80/0.2)=400, x=(70.4/0.16)=440 — both divisible by 8 (SECONDFPN requirement)
batch_size        = 16
lr                = 0.001
max_epochs        = 1000
```

### Checkpoints
Every 20 epochs: `work_dirs/pointpillars_a2d2/epoch_*.pth`

---

## 7. Inference & Prediction Videos

> **Important**: Inference must run on A100 (hpc10) or H100 (hpc11).
> NOT on login node (wrong GLIBC) or RTX 8000 (wrong CUDA arch).

### Python API
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

# Run on a BIN file [x, y, z, reflectance] float32
result = inferencer(inputs=dict(points='path/to/frame.bin'), no_save_pred=False)
# result: bboxes_3d [N,7], labels_3d [N], scores_3d [N]
# bbox format: [cx, cy, cz_bottom, l, w, h, yaw]
```

### CLI
```bash
export PYTHONNOUSERSITE=1; unset PYTHONPATH
PYTHON=/path/to/allEnv/mmdet3d/bin/python

# PV-RCNN
$PYTHON tools/make_pred_video.py \
  --config  configs/pv_rcnn/pv_rcnn_a2d2.py \
  --checkpoint work_dirs/pv_rcnn_a2d2/epoch_1000.pth \
  --epoch 1000 \
  --data-root data/a2d2/ \
  --num-frames 60 \
  --out-dir outputs/pred_videos \
  --score-thr 0.3

# PointPillars
$PYTHON tools/make_pred_video.py \
  --config  configs/pointpillars/pointpillars_a2d2.py \
  --checkpoint work_dirs/pointpillars_a2d2/epoch_1000.pth \
  --epoch 1000 \
  --data-root data/a2d2/ \
  --num-frames 60 \
  --out-dir outputs/pred_videos_pp \
  --score-thr 0.3
```

### Via SLURM (recommended)
```bash
sbatch gen_video.sh      # PV-RCNN — edit checkpoint path inside first
sbatch gen_video_pp.sh   # PointPillars
```

### 3D Box Projection to Camera
The video script estimates a DLT projection matrix from LiDAR `col`/`row` correspondences:
```python
_, _, Vt = np.linalg.svd(A)
P = Vt[-1].reshape(3, 4)
P /= np.linalg.norm(P[2, :3])
if (P @ test_point)[2] < 0: P = -P   # ensure positive depth
```
Box corners use mmdet3d bottom-center convention:
```python
zs = [0,1,1,0,0,1,1,0] * height   # 0=bottom face, 1=top face
```

### Download Videos
```bash
scp user@server:mmdetection3d/outputs/pred_videos/pred_epoch_1000.mp4 .      # PV-RCNN
scp user@server:mmdetection3d/outputs/pred_videos_pp/pred_epoch_1000.mp4 .   # PointPillars
scp user@server:mmdetection3d/outputs/gt_verification_h264.avi .              # GT reference
```

---

## 8. Evaluating on Test Data

If you have the full A2D2 dataset and want to evaluate on the val set:

### Option A — Use existing val PKL
The val PKL (`data/a2d2/a2d2_infos_val.pkl`) already contains the 1,261 val frames.
Use `tools/make_pred_video.py` to run inference (see Section 7).

### Option B — Evaluate on a custom subset
```python
import pickle

# Load val PKL
with open('data/a2d2/a2d2_infos_val.pkl', 'rb') as f:
    val_data = pickle.load(f)

# Each entry has: sample_idx, lidar_points.lidar_path, images.CAM2.img_path, instances
# Val frames are the LAST 10% of each sequence (see split table in Section 1)

# To evaluate specific sequences, filter by lidar_path prefix:
seq_data = [x for x in val_data
            if '20181108_123750' in x['lidar_points']['lidar_path']]
```

### Option C — Rebuild with different split
Edit `tools/prepare_a2d2.py` function `split_triplets()`:
```python
def split_triplets(seq_triplets):
    train, val = [], []
    for seq, frames in seq_triplets.items():
        n_val = max(1, math.ceil(len(frames) * 0.10))  # change 0.10 for different ratio
        train.extend(...)
        val.extend(...)
```
Then `sbatch rebuild_pkl.sh`.

### Run Full Inference on Val Set
```python
from mmdet3d.apis import LiDAR3DInferencer
import pickle, os

inferencer = LiDAR3DInferencer(
    model='configs/pv_rcnn/pv_rcnn_a2d2.py',
    weights='work_dirs/pv_rcnn_a2d2/epoch_1000.pth'
)

with open('data/a2d2/a2d2_infos_val.pkl', 'rb') as f:
    val_data = pickle.load(f)

results = []
for info in val_data:
    lidar_path = os.path.join('data/a2d2', info['lidar_points']['lidar_path'])
    if not os.path.exists(lidar_path):
        continue
    pred = inferencer(inputs=dict(points=lidar_path), no_save_pred=False)
    results.append(pred)
# results[i]: bboxes_3d [N,7], labels_3d [N], scores_3d [N]
```

---

## 9. Results

| Model | Epochs | Final Loss | Hardware | Duration | Download |
|-------|--------|-----------|----------|----------|----------|
| PV-RCNN | 1000 | **0.644** | 2× A100 80GB | ~7.5 days | [model](https://github.com/aniketntnu/mmdetection3d/releases/download/v1.0-models/pvrcnn_a2d2_epoch1000.pth) · [video](https://github.com/aniketntnu/mmdetection3d/releases/download/v1.0-models/pvrcnn_pred_epoch1000.mp4) |
| PointPillars | 1000 | — | 1× RTX 8000 | ~17 hrs | [model](https://github.com/aniketntnu/mmdetection3d/releases/download/v1.0-models/pointpillars_a2d2_epoch1000.pth) · [video](https://github.com/aniketntnu/mmdetection3d/releases/download/v1.0-models/pointpillars_pred_epoch1000.mp4) |

### PV-RCNN Loss Curve
| Epoch | Loss |
|-------|------|
| 1 | 7.03 |
| 50 | ~2.5 |
| 100 | ~2.0 |
| 200 | ~1.7 |
| 500 | ~1.2 |
| 1000 | **0.644** |

### Key Lessons Learned
- A2D2 LiDAR is in camera frame — no transformation needed (verified via `col`/`row` correlation)
- Truck/Bus anchor heights must be capped: `anchor_z + H ≤ z_max=1.0m`
- NaN loss from bad batches is inevitable in early training — the zero-out fix in `pv_rcnn.loss()` is essential
- `lr=0.001` with AdamW is stable; scaling LR with batch size causes NaN (~epoch 25)
- Validation is disabled (`val_interval=9999`) because KittiMetric needs `lidar2cam` calibration matrices not in A2D2 PKL
- Video generation must use a separate GPU (H100/hpc11) from training — CUDA context conflict otherwise
