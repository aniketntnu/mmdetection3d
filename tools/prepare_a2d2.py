#!/cluster/datastore/aniketag/allEnv/mmdet3d/bin/python
"""
prepare_a2d2.py — Fast A2D2 data prep (subprocess tar + multiprocessing).

Key speedups:
  - subprocess tar --wildcards: skips ~33GB semantic-label masks
  - multiprocessing Pool (16 workers): NPZ->BIN + annotation parsing in parallel
  - GT video: only 20 frames
"""

import os, sys, re, json, math, pickle, subprocess
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import cv2
from tqdm import tqdm

TAR_PATH   = "/cluster/datastore/aniketag/allData/camera_lidar_semantic_bboxes.tar"
REPO       = Path("/cluster/datastore/aniketag/mmdetection3d")
DATA_ROOT  = REPO / "data" / "a2d2"
RAW_DIR    = DATA_ROOT / "raw" / "camera_lidar_semantic_bboxes"
LIDAR_DIR  = DATA_ROOT / "lidar"
IMAGE_DIR  = DATA_ROOT / "images"
GT_VID_DIR = REPO / "outputs" / "gt_video"
NUM_WORKERS = min(16, cpu_count())

CLASS_MAP = {
    "Car": 0, "VanSUV": 0,                               # 0 Car
    "Truck": 1, "UtilityVehicle": 1,                      # 1 Truck
    "Trailer": 1, "CaravanTransporter": 1,                # 1 Truck
    "Bus": 2,                                             # 2 Bus
    "Pedestrian": 3,                                      # 3 Pedestrian
    "Cyclist": 4, "Bicycle": 4, "MotorBiker": 4,          # 4 Cyclist
    "Motorcycle": 4,                                      # 4 Cyclist
}


# ── 1. Extract (skip ~33GB semantic masks) ────────────────────────────────────
def extract_tar():
    flag = DATA_ROOT / ".extracted"
    if flag.exists():
        print("Already extracted, skipping."); return
    for d in [DATA_ROOT, LIDAR_DIR, IMAGE_DIR, RAW_DIR.parent, GT_VID_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print("Extracting (camera+lidar+label3D only, skipping semantic masks)...")
    # --exclude skips the large semantic-mask label/ dir (~33GB) while extracting everything else
    cmd = ["tar","-xf", TAR_PATH,
           "--exclude=camera_lidar_semantic_bboxes/*/label/",
           "-C", str(RAW_DIR.parent)]
    r = subprocess.run(cmd)
    if r.returncode != 0: sys.exit("tar failed")
    flag.touch()
    print("Extraction done.")


# ── 2. Find matching triplets ─────────────────────────────────────────────────
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


# ── 3. 90/10 split ────────────────────────────────────────────────────────────
def split_triplets(seq_triplets):
    train, val = [], []
    for seq, frames in seq_triplets.items():
        n_val = max(1, math.ceil(len(frames)*0.10))
        train.extend([(seq,*f) for f in frames[:-n_val]])
        val.extend(  [(seq,*f) for f in frames[-n_val:]])
    print(f"Train: {len(train)}  Val: {len(val)}")
    return train, val


# ── 4. NPZ→BIN + image copy (parallel) ───────────────────────────────────────
def _convert_one(args):
    seq, ts, cam_path, npz_path, lab_path = args
    stem = f"{seq}_{ts}"
    bin_out = LIDAR_DIR/f"{stem}.bin"
    img_out = IMAGE_DIR/f"{stem}.png"
    if not bin_out.exists():
        d = np.load(npz_path)
        pts  = d["points"].astype(np.float32)
        refl = d["reflectance"].astype(np.float32)/100.0
        np.hstack([pts, refl[:,None]]).tofile(bin_out)
    if not img_out.exists():
        import shutil; shutil.copy2(cam_path, img_out)
    return stem

def convert_all(frames):
    LIDAR_DIR.mkdir(exist_ok=True); IMAGE_DIR.mkdir(exist_ok=True)
    args=[(seq,ts,str(c),str(l),str(lb)) for seq,ts,c,l,lb in frames]
    print(f"NPZ→BIN with {NUM_WORKERS} workers ({len(args)} frames)...")
    with Pool(NUM_WORKERS) as p:
        list(tqdm(p.imap(_convert_one, args, chunksize=64), total=len(args)))


# ── 5. Build annotation pkl (parallel) ───────────────────────────────────────
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
    args=[(i,seq,ts,str(c),str(l),str(lb)) for i,(seq,ts,c,l,lb) in enumerate(frames)]
    print(f"Building {name} pkl ({len(args)} frames)...")
    with Pool(NUM_WORKERS) as p:
        infos=list(tqdm(p.imap(_parse_one,args,chunksize=64),total=len(args)))
    out=DATA_ROOT/f"a2d2_infos_{name}.pkl"
    with open(out,"wb") as f: pickle.dump(infos,f)
    print(f"  → {out}"); return infos


# ── 6. GT video (20 frames) ───────────────────────────────────────────────────
def _skew(u): return np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
def _aarot(ax,ang):
    return np.cos(ang)*np.eye(3)+np.sin(ang)*_skew(ax)+(1-np.cos(ang))*np.outer(ax,ax)

GT_EDGES=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
COLORS={0:(0,255,0),1:(0,128,255),2:(255,128,0),3:(255,0,255),4:(0,255,255)}

def _render(info):
    img=cv2.imread(info["_cam_path"])
    if img is None: return None
    d=np.load(info["_npz_path"])
    pts,col,row=d["points"],d["col"],d["row"]
    valid=(pts[:,0]>2)&(col>50)&(col<1870)&(row>50)&(row<1158)
    idx=np.where(valid)[0]; np.random.seed(42)
    idx=np.random.choice(idx,min(2000,len(idx)),replace=False)
    A=[]
    for (X,Y,Z),(u,v) in zip(pts[idx],np.stack([col[idx],row[idx]],1)):
        A+=[[X,Y,Z,1,0,0,0,0,-u*X,-u*Y,-u*Z,-u],[0,0,0,0,X,Y,Z,1,-v*X,-v*Y,-v*Z,-v]]
    _,_,Vt=np.linalg.svd(np.array(A))
    P=Vt[-1].reshape(3,4); P/=np.linalg.norm(P[2,:3])
    h=np.array([pts[idx[0],0],pts[idx[0],1],pts[idx[0],2],1.0])
    if (P@h)[2]<0: P=-P
    with open(info["_label_path"]) as f: lab=json.load(f)
    for obj in lab.values():
        cls=obj.get("class","")
        if cls not in CLASS_MAP: continue
        c=np.array(obj["center"]); s=np.array(obj["size"])/2
        rot=_aarot(np.array(obj["axis"]),obj["rot_angle"])
        local=np.array([[-s[0],+s[1],-s[2]],[+s[0],+s[1],-s[2]],
                         [+s[0],-s[1],-s[2]],[-s[0],-s[1],-s[2]],
                         [-s[0],+s[1],+s[2]],[+s[0],+s[1],+s[2]],
                         [+s[0],-s[1],+s[2]],[-s[0],-s[1],+s[2]]])
        corners=(rot@local.T).T+c
        ph=np.hstack([corners,np.ones((8,1))]); q=(P@ph.T).T
        u2=q[:,0]/q[:,2]; v2=q[:,1]/q[:,2]
        col2=COLORS[CLASS_MAP[cls]]
        for i,j in GT_EDGES:
            cv2.line(img,(int(np.clip(u2[i],-9999,9999)),int(np.clip(v2[i],-9999,9999))),
                         (int(np.clip(u2[j],-9999,9999)),int(np.clip(v2[j],-9999,9999))),
                     col2,2,cv2.LINE_AA)
        cv2.putText(img,cls,(int(np.clip(u2.mean(),0,1919)),int(np.clip(v2.min()-5,14,1207))),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,col2,2,cv2.LINE_AA)
    return img

def make_gt_video(train_infos, n=20, out_path=None):
    print(f"GT video ({n} frames)...")
    step=max(1,len(train_infos)//n)
    chosen=train_infos[::step][:n]
    GT_VID_DIR.mkdir(parents=True,exist_ok=True)
    if out_path is None:
        out_path=str(REPO/"outputs"/"gt_verification.mp4")
    writer=cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*"mp4v"),1,(1920,1208))
    for i,info in enumerate(tqdm(chosen)):
        frame=_render(info)
        if frame is None: continue
        cv2.imwrite(str(GT_VID_DIR/f"gt_{i:03d}.jpg"),frame)
        writer.write(frame)
    writer.release()
    print(f"GT video → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Workers: {NUM_WORKERS}")
    extract_tar()
    seq_triplets = find_triplets()
    train_frames, val_frames = split_triplets(seq_triplets)
    convert_all(train_frames + val_frames)
    train_infos = build_pkl(train_frames, "train")
    build_pkl(val_frames, "val")
    make_gt_video(train_infos)
    print(f"\n=== Done ===")
    print(f"  GT video  : {REPO}/outputs/gt_verification.mp4")
    print(f"  Train pkl : {DATA_ROOT}/a2d2_infos_train.pkl")
    print(f"  Val pkl   : {DATA_ROOT}/a2d2_infos_val.pkl")

if __name__=="__main__": main()
