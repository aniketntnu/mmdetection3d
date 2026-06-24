#!/usr/bin/env python3
"""
make_pred_video.py — Run PVRCNN inference on A2D2 validation frames and render
a prediction video with 3-D bounding boxes projected onto the camera image.

IMPORTANT: This script requires a GPU (CUDA).  Launch it on a GPU node with:
    PYTHONNOUSERSITE=1 python tools/make_pred_video.py \\
        --config  configs/pv_rcnn/pv_rcnn_a2d2.py \\
        --checkpoint work_dirs/pv_rcnn_a2d2/epoch_80.pth \\
        --epoch 80

Corner convention (mmdet3d bottom-center LiDAR box):
    Given box = [cx, cy, cz_bottom, dx, dy, dz, yaw]:
        xs = [-1,-1,-1,-1,+1,+1,+1,+1] * (dx/2)
        ys = [-1,-1,+1,+1,-1,-1,+1,+1] * (dy/2)
        zs = [ 0, 1, 1, 0, 0, 1, 1, 0] *  dz
    After rotation by yaw around z and translation by (cx, cy, cz_bottom).

Edge list (12 edges, same as KITTI):
    Bottom ring : (0,3),(3,7),(7,4),(4,0)
    Top ring    : (1,2),(2,6),(6,5),(5,1)
    Verticals   : (0,1),(3,2),(7,6),(4,5)
"""

import argparse
import os
import os.path as osp
import pickle
import subprocess
import sys
import tempfile

import cv2
import numpy as np

# ── mmdet3d imports (requires PYTHONNOUSERSITE=1 on cluster) ──────────────────
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))
from mmdet3d.apis import LidarDet3DInferencer  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']
CLASS_COLORS = {
    0: (0, 255, 0),       # Car      — green
    1: (0, 128, 255),     # Pedestrian — blue-orange
    2: (255, 128, 0),     # Cyclist  — orange
}

# mmdet3d bottom-center corner template (unit box, centred at origin / bottom z=0)
_XS = np.array([-1, -1, -1, -1, +1, +1, +1, +1], dtype=np.float64)
_YS = np.array([-1, -1, +1, +1, -1, -1, +1, +1], dtype=np.float64)
_ZS = np.array([0,  1,  1,  0,  0,  1,  1,  0], dtype=np.float64)

EDGES = [
    (0, 3), (3, 7), (7, 4), (4, 0),   # bottom face
    (1, 2), (2, 6), (6, 5), (5, 1),   # top face
    (0, 1), (3, 2), (7, 6), (4, 5),   # verticals
]


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def box_corners(cx, cy, cz_bottom, dx, dy, dz, yaw):
    """Return (8, 3) corner array for a bottom-center LiDAR box.

    Args:
        cx, cy, cz_bottom : box centre-bottom in LiDAR frame
        dx, dy, dz        : dimensions (length, width, height)
        yaw               : rotation around z axis (radians)

    Returns:
        np.ndarray: shape (8, 3)
    """
    xs = _XS * (dx / 2.0)
    ys = _YS * (dy / 2.0)
    zs = _ZS * dz

    # Rotation around z
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0],
                    [s,  c, 0],
                    [0,  0, 1]], dtype=np.float64)

    local = np.stack([xs, ys, zs], axis=1)          # (8, 3)
    world = (rot @ local.T).T + np.array([cx, cy, cz_bottom])
    return world


def compute_proj(pts_3d, col, row):
    """DLT projection matrix from LiDAR col/row pixel correspondences.

    Args:
        pts_3d : (N, 3) float  — LiDAR points
        col    : (N,) float    — pixel column of each point
        row    : (N,) float    — pixel row of each point

    Returns:
        np.ndarray or None: (3, 4) projection matrix, or None on failure.
    """
    valid = (
        (pts_3d[:, 0] > 2.0) &
        (col > 50) & (col < 1870) &
        (row > 50) & (row < 1158)
    )
    idx = np.where(valid)[0]
    if len(idx) < 100:
        return None

    rng = np.random.default_rng(42)
    idx = rng.choice(idx, min(3000, len(idx)), replace=False)

    p2d = np.stack([col[idx], row[idx]], axis=1)
    p3d = pts_3d[idx]

    A = []
    for (X, Y, Z), (u, v) in zip(p3d, p2d):
        A.append([X, Y, Z, 1,  0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0,  X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])

    _, _, Vt = np.linalg.svd(np.array(A, dtype=np.float64))
    P = Vt[-1].reshape(3, 4)
    P /= np.linalg.norm(P[2, :3])

    # Ensure frontal points have positive depth
    test_h = np.array([p3d[0, 0], p3d[0, 1], p3d[0, 2], 1.0])
    if (P @ test_h)[2] < 0:
        P = -P

    return P


def project_pts(pts_3d, P):
    """Project (N, 3) 3-D points through (3, 4) matrix P.

    Returns:
        pix   : (N, 2) float  — [u, v] pixel coordinates
        depth : (N,)   float  — projective depth
    """
    pts_h = np.hstack([pts_3d, np.ones((len(pts_3d), 1))])
    q = (P @ pts_h.T).T                              # (N, 3)
    depth = q[:, 2]
    denom = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    u = q[:, 0] / denom
    v = q[:, 1] / denom
    return np.stack([u, v], axis=1), depth


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_box(img, corners_3d, P, label, score):
    """Project a single 3-D box and draw it on img (in-place).

    Args:
        img        : BGR numpy image (modified in place)
        corners_3d : (8, 3) LiDAR-frame corners
        P          : (3, 4) projection matrix
        label      : int class index
        score      : float confidence score
    """
    h, w = img.shape[:2]
    color = CLASS_COLORS.get(int(label), (255, 255, 255))

    # Skip boxes mostly behind camera (forward axis = x)
    if corners_3d[:, 0].mean() < 0.5:
        return

    pix, depth = project_pts(corners_3d, P)
    if (depth < 0).all():
        return

    u, v = pix[:, 0], pix[:, 1]
    if not ((u > -w) & (u < 2 * w) & (v > -h) & (v < 2 * h)).any():
        return

    def pt(i):
        return (int(np.clip(pix[i, 0], -10000, 10000)),
                int(np.clip(pix[i, 1], -10000, 10000)))

    for i, j in EDGES:
        cv2.line(img, pt(i), pt(j), color, 2, cv2.LINE_AA)

    # Score label above front-top corners (indices 1 and 5)
    tx = int(np.clip((pix[1, 0] + pix[5, 0]) / 2, 0, w - 1))
    ty = int(np.clip(min(pix[1, 1], pix[5, 1]) - 5, 14, h - 1))
    name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else str(label)
    cv2.putText(img, f'{name} {score:.2f}', (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# NPZ loading (to obtain col/row for DLT)
# ─────────────────────────────────────────────────────────────────────────────

def load_npz_for_frame(info, data_root):
    """Try to locate the original NPZ for a val frame and load it.

    The lidar_path in the pkl points to a .bin file inside data/a2d2/lidar/.
    The original NPZ lives at data/a2d2/raw/camera_lidar_semantic_bboxes/
    <seq>/lidar/cam_front_center/<seq_ts>.npz.

    Strategy:
        1. Derive seq + ts from the bin filename: <seq_compact>_<ts>.bin
        2. Search raw/ recursively for a matching NPZ stem.

    Returns:
        dict with 'points', 'col', 'row' keys, or None on failure.
    """
    lidar_rel = info['lidar_points']['lidar_path']
    bin_name = osp.basename(lidar_rel)          # e.g. 20181107132300_000012345.bin
    stem = osp.splitext(bin_name)[0]             # 20181107132300_000012345

    # Reconstruct seq_compact and ts
    parts = stem.rsplit('_', 1)
    if len(parts) != 2:
        return None
    seq_compact, ts = parts[0], parts[1]

    # Find raw dir
    raw_dir = osp.join(data_root, 'raw', 'camera_lidar_semantic_bboxes')
    if not osp.isdir(raw_dir):
        return None

    # Reconstruct sequence dir name: insert underscore after date part (8 chars)
    # seq_compact = "20181107132300" → seq = "20181107_132300"
    if len(seq_compact) == 14:
        seq_dir = seq_compact[:8] + '_' + seq_compact[8:]
    else:
        seq_dir = seq_compact

    npz_path = osp.join(raw_dir, seq_dir, 'lidar', 'cam_front_center',
                        f'{seq_dir}_{ts}.npz')
    if not osp.isfile(npz_path):
        # Fallback: glob search
        import glob
        pattern = osp.join(raw_dir, '**', f'*{ts}.npz')
        hits = glob.glob(pattern, recursive=True)
        if not hits:
            return None
        npz_path = hits[0]

    try:
        return dict(np.load(npz_path))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(inferencer, lidar_path, score_thr):
    """Run LidarDet3DInferencer on one .bin file.

    Returns:
        list of dicts with keys: bbox_3d, score_3d, label_3d
        (empty list on failure / no detections above threshold)
    """
    results = inferencer(
        inputs=dict(points=lidar_path),
        no_save_vis=True,
        pred_score_thr=score_thr)

    predictions = results.get('predictions', [])
    if not predictions:
        return []

    pred = predictions[0]
    bboxes = np.array(pred.get('bboxes_3d', []), dtype=np.float32)
    scores = np.array(pred.get('scores_3d', []), dtype=np.float32)
    labels = np.array(pred.get('labels_3d', []), dtype=np.int32)

    detections = []
    for bbox, score, label in zip(bboxes, scores, labels):
        detections.append(dict(bbox_3d=bbox, score_3d=float(score),
                               label_3d=int(label)))
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run PVRCNN inference on A2D2 val frames and make a video')
    parser.add_argument('--config', required=True,
                        help='Path to mmdet3d config (.py)')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--epoch', type=int, default=0,
                        help='Epoch number used in output filename')
    parser.add_argument('--data-root', default='data/a2d2/',
                        help='A2D2 data root (default: data/a2d2/)')
    parser.add_argument('--num-frames', type=int, default=60,
                        help='Number of val frames to process (default: 60)')
    parser.add_argument('--out-dir', default='outputs/pred_videos/',
                        help='Output directory (default: outputs/pred_videos/)')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='Score threshold for predictions (default: 0.3)')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    data_root = args.data_root
    if not osp.isabs(data_root):
        data_root = osp.join(repo_root, data_root)

    out_dir = args.out_dir
    if not osp.isabs(out_dir):
        out_dir = osp.join(repo_root, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    frames_dir = osp.join(out_dir, f'frames_epoch_{args.epoch:03d}')
    os.makedirs(frames_dir, exist_ok=True)

    video_path = osp.join(out_dir, f'pred_epoch_{args.epoch:03d}.mp4')

    # ── Load val pkl ──────────────────────────────────────────────────────────
    pkl_path = osp.join(data_root, 'a2d2_infos_val.pkl')
    print(f'[info] Loading val pkl: {pkl_path}')
    with open(pkl_path, 'rb') as f:
        val_infos = pickle.load(f)

    n_total = len(val_infos)
    n_pick = min(args.num_frames, n_total)
    indices = np.linspace(0, n_total - 1, n_pick, dtype=int)
    picked = [val_infos[i] for i in indices]
    print(f'[info] Using {n_pick} / {n_total} val frames')

    # ── Build inferencer ──────────────────────────────────────────────────────
    print(f'[info] Loading model: {args.config}  /  {args.checkpoint}')
    inferencer = LidarDet3DInferencer(
        model=args.config,
        weights=args.checkpoint,
        device='cuda')

    # ── Process frames ────────────────────────────────────────────────────────
    frame_paths = []
    img_shape = None

    for frame_idx, info in enumerate(picked):
        print(f'[frame {frame_idx + 1:3d}/{n_pick}] ', end='', flush=True)

        # ── LiDAR path ───────────────────────────────────────────────────────
        lidar_rel = info['lidar_points']['lidar_path']
        lidar_abs = lidar_rel if osp.isabs(lidar_rel) else osp.join(data_root, lidar_rel)

        if not osp.isfile(lidar_abs):
            print(f'WARN: lidar not found: {lidar_abs}')
            continue

        # ── Camera image ──────────────────────────────────────────────────────
        img_path = None
        if 'images' in info:
            cam_info = info['images']
            # prefer CAM2 / first available
            for cam_key in ('CAM2', 'cam_front_center', sorted(cam_info.keys())[0]):
                if cam_key in cam_info and 'img_path' in cam_info[cam_key]:
                    rel = cam_info[cam_key]['img_path']
                    img_path = rel if osp.isabs(rel) else osp.join(data_root, rel)
                    break

        if img_path is None or not osp.isfile(img_path):
            print(f'WARN: image not found, skipping.')
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f'WARN: cv2.imread failed for {img_path}')
            continue

        if img_shape is None:
            img_shape = img.shape[:2]   # (H, W)

        # ── Projection matrix (DLT from NPZ col/row) ─────────────────────────
        P = None
        npz_data = load_npz_for_frame(info, data_root)
        if npz_data is not None:
            pts_3d = npz_data.get('points', None)
            col = npz_data.get('col', None)
            row = npz_data.get('row', None)
            if pts_3d is not None and col is not None and row is not None:
                P = compute_proj(
                    pts_3d.astype(np.float64),
                    col.astype(np.float64).ravel(),
                    row.astype(np.float64).ravel())

        if P is None:
            print('WARN: DLT failed (no valid projection), skipping boxes.')

        # ── Run inference ─────────────────────────────────────────────────────
        detections = run_inference(inferencer, lidar_abs, args.score_thr)
        print(f'{len(detections)} dets', end='')

        # ── Draw predicted boxes ──────────────────────────────────────────────
        if P is not None:
            for det in detections:
                bbox = det['bbox_3d']           # [cx,cy,cz_bottom,dx,dy,dz,yaw]
                cx, cy, cz_b = float(bbox[0]), float(bbox[1]), float(bbox[2])
                dx, dy, dz   = float(bbox[3]), float(bbox[4]), float(bbox[5])
                yaw           = float(bbox[6])
                corners = box_corners(cx, cy, cz_b, dx, dy, dz, yaw)
                draw_box(img, corners, P, det['label_3d'], det['score_3d'])

        # ── Text overlay ──────────────────────────────────────────────────────
        sample_id = info.get('sample_idx', str(frame_idx))
        cv2.putText(img,
                    f'Frame {frame_idx + 1}/{n_pick}  idx={sample_id}  '
                    f'epoch={args.epoch}  dets={len(detections)}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # ── Save frame ────────────────────────────────────────────────────────
        frame_file = osp.join(frames_dir, f'frame_{frame_idx:04d}.jpg')
        cv2.imwrite(frame_file, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_paths.append(frame_file)
        print(f'  → {osp.basename(frame_file)}')

    if not frame_paths:
        print('[error] No frames were successfully rendered.  Exiting.')
        sys.exit(1)

    # ── Assemble MP4 with ffmpeg (1 fps) ─────────────────────────────────────
    print(f'\n[video] Encoding {len(frame_paths)} frames → {video_path}')

    # Write a temp file-list for ffmpeg concat demuxer (handles any filename)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                     delete=False) as flist:
        for fp in frame_paths:
            flist.write(f"file '{fp}'\n")
            flist.write("duration 1\n")
        flist_path = flist.name

    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0',
            '-i', flist_path,
            '-vf', f'scale={img_shape[1] if img_shape else 1920}:'
                   f'{img_shape[0] if img_shape else 1200}',
            '-c:v', 'libx264',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            video_path
        ], check=True)
        print(f'[done] Video saved → {video_path}')
    except FileNotFoundError:
        # ffmpeg not available — fall back to OpenCV VideoWriter
        print('[warn] ffmpeg not found, falling back to OpenCV VideoWriter.')
        h = img_shape[0] if img_shape else 1200
        w = img_shape[1] if img_shape else 1920
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, 1.0, (w, h))
        for fp in frame_paths:
            frame = cv2.imread(fp)
            if frame is not None:
                writer.write(frame)
        writer.release()
        print(f'[done] Video saved → {video_path}')
    finally:
        os.unlink(flist_path)

    print(video_path)


if __name__ == '__main__':
    main()
