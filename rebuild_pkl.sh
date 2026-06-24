#!/bin/bash
#SBATCH --job-name=a2d2_pkl
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/datastore/aniketag/mmdetection3d/logs/rebuild_pkl_%j.log
#SBATCH --error=/cluster/datastore/aniketag/mmdetection3d/logs/rebuild_pkl_%j.log

export PYTHONNOUSERSITE=1
unset PYTHONPATH
PYTHON=/cluster/datastore/aniketag/allEnv/mmdet3d/bin/python
REPO=/cluster/datastore/aniketag/mmdetection3d

cd $REPO
echo "=== Rebuilding PKL with 5 classes ==="

# Skip extraction (already done), just rebuild annotations + GT video
$PYTHON - << 'EOF'
import os, sys, re, json, math, pickle
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm

REPO       = Path("/cluster/datastore/aniketag/mmdetection3d")
DATA_ROOT  = REPO / "data" / "a2d2"
RAW_DIR    = DATA_ROOT / "raw" / "camera_lidar_semantic_bboxes"

CLASS_MAP = {
    "Car": 0, "VanSUV": 0,
    "Truck": 1, "UtilityVehicle": 1, "Trailer": 1, "CaravanTransporter": 1,
    "Bus": 2,
    "Pedestrian": 3,
    "Cyclist": 4, "Bicycle": 4, "MotorBiker": 4, "Motorcycle": 4,
}

def _ts(fname):
    m = re.search(r'_(\d{9,})\.', fname)
    return m.group(1) if m else None

def find_triplets():
    seq_triplets = defaultdict(list)
    for seq_dir in sorted(RAW_DIR.iterdir()):
        if not seq_dir.is_dir(): continue
        cam_d = seq_dir/"camera"/"cam_front_center"
        lid_d = seq_dir/"lidar"/"cam_front_center"
        lab_d = seq_dir/"label3D"/"cam_front_center"
        if not (cam_d.exists() and lid_d.exists() and lab_d.exists()): continue
        cams={_ts(f.name):f for f in cam_d.glob("*.png") if _ts(f.name)}
        lids={_ts(f.name):f for f in lid_d.glob("*.npz") if _ts(f.name)}
        labs={_ts(f.name):f for f in lab_d.glob("*.json") if _ts(f.name)}
        for ts in sorted(set(cams)&set(lids)&set(labs)):
            seq_triplets[seq_dir.name].append((ts,cams[ts],lids[ts],labs[ts]))
    total = sum(len(v) for v in seq_triplets.values())
    print(f"Found {total} frames across {len(seq_triplets)} sequences")
    return seq_triplets

def split_triplets(seq_triplets):
    train, val = [], []
    for seq, frames in seq_triplets.items():
        n_val = max(1, math.ceil(len(frames)*0.10))
        train.extend([(seq,*f) for f in frames[:-n_val]])
        val.extend(  [(seq,*f) for f in frames[-n_val:]])
    print(f"Train: {len(train)}  Val: {len(val)}")
    return train, val

def _parse_one(args):
    idx, seq, ts, cam_path, npz_path, lab_path = args
    stem = f"{seq}_{ts}"
    instances = []
    with open(lab_path) as f: lab=json.load(f)
    for obj in lab.values():
        cls=obj.get("class","")
        if cls not in CLASS_MAP: continue
        c=obj["center"]; s=obj["size"]; yaw=obj["rot_angle"]
        instances.append({
            "bbox_3d":       [c[0],c[1],c[2]-s[2]/2.0, s[0],s[1],s[2], yaw],
            "bbox_label_3d": CLASS_MAP[cls],
            "bbox":          obj.get("2d_bbox",[0,0,1,1]),
            "bbox_label":    CLASS_MAP[cls],
            "num_lidar_pts": -1, "difficulty":0,
            "truncated":     float(obj.get("truncation",0.0)),
            "occluded":      int(obj.get("occlusion",0)),
        })
    return {
        "sample_idx":   str(idx),
        "lidar_points": {"lidar_path":f"lidar/{stem}.bin","num_pts_dim":4},
        "images":       {"CAM2":{"img_path":f"images/{stem}.png"}},
        "instances":    instances,
        "_npz_path": str(npz_path), "_cam_path": str(cam_path),
        "_label_path": str(lab_path),
    }

def build_pkl(frames, name):
    NUM_WORKERS = min(16, cpu_count())
    args=[(i,seq,ts,str(c),str(l),str(lb)) for i,(seq,ts,c,l,lb) in enumerate(frames)]
    print(f"Building {name} pkl ({len(args)} frames) with {NUM_WORKERS} workers...")
    with Pool(NUM_WORKERS) as p:
        infos=list(tqdm(p.imap(_parse_one,args,chunksize=64),total=len(args)))
    out=DATA_ROOT/f"a2d2_infos_{name}.pkl"
    with open(out,"wb") as f: pickle.dump(infos,f)
    # Count per class
    from collections import Counter
    labels = Counter(inst['bbox_label_3d'] for item in infos for inst in item['instances'])
    names = {0:'Car',1:'Truck',2:'Bus',3:'Pedestrian',4:'Cyclist'}
    print(f"  → {out}")
    for k,v in sorted(labels.items()):
        print(f"     {names[k]:12}: {v:6d}")
    return infos

seq_triplets = find_triplets()
train_frames, val_frames = split_triplets(seq_triplets)
build_pkl(train_frames, "train")
build_pkl(val_frames, "val")
print("\n=== PKL rebuild complete ===")
EOF

if [ $? -eq 0 ]; then
    echo "=== PKL rebuild SUCCESS — submitting training ==="
    sbatch /cluster/datastore/aniketag/mmdetection3d/train_a2d2_a100.sh
else
    echo "=== PKL rebuild FAILED ==="
fi
