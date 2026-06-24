# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
from typing import Callable, List, Union

from typing import Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class A2D2Dataset(Det3DDataset):
    """A2D2 (Audi Autonomous Driving Dataset) for 3-class LiDAR detection.

    Classes:
        0 - Car
        1 - Pedestrian
        2 - Cyclist

    The coordinate frame matches KITTI LiDAR:  x=forward, y=left, z=up.
    All 3-D boxes are stored in bottom-center format:
        [cx, cy, cz_bottom, length, width, height, yaw].

    Args:
        data_root (str): Path of dataset root (e.g. 'data/a2d2/').
        ann_file (str): Annotation pkl file name (relative to data_root),
            e.g. 'a2d2_infos_train.pkl'.
        pipeline (list[dict]): Processing pipeline.
        modality (dict): Sensor modalities. Defaults to use_lidar=True.
        box_type_3d (str): Type of 3-D box. Defaults to 'LiDAR'.
        filter_empty_gt (bool): Drop samples with no annotations during
            training. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
    """

    METAINFO = {
        'classes': ('Car', 'Truck', 'Bus', 'Pedestrian', 'Cyclist'),
        'palette': [
            (0, 255, 0),       # Car        - green
            (255, 0, 0),       # Truck      - red
            (255, 165, 0),     # Bus        - orange
            (0, 128, 255),     # Pedestrian - blue
            (255, 0, 255),     # Cyclist    - magenta
        ]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True),
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 **kwargs) -> None:
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load the A2D2 annotation pkl file.

        The pkl is a plain Python list of per-frame dicts (not the mmengine
        dict-with-metainfo format), so we override load_data_list to read it
        directly and call parse_data_info on each entry.

        Returns:
            list[dict]: List of processed data-info dicts.
        """
        ann_file_path = self.ann_file
        with open(ann_file_path, 'rb') as f:
            raw_list = pickle.load(f)

        if not isinstance(raw_list, list):
            raise TypeError(
                f'A2D2 annotation pkl must contain a list, '
                f'got {type(raw_list)}')

        data_list = []
        for raw_info in raw_list:
            data_info = self.parse_data_info(raw_info)
            if isinstance(data_info, dict):
                data_list.append(data_info)
            elif isinstance(data_info, list):
                data_list.extend(data_info)

        return data_list

    def parse_data_info(self, info: dict) -> dict:
        """Convert a raw pkl entry to the mmdet3d standard format.

        The base-class ``Det3DDataset.parse_data_info`` expects:
          - ``info['lidar_points']['lidar_path']``: (possibly relative) path
          - ``info['lidar_points']['num_pts_feats']``: number of point dims
          - ``info['instances']``: list of dicts with
              ``bbox_3d`` (list[7]) and ``bbox_label_3d`` (int)

        A2D2 pkl stores ``num_pts_dim`` instead of ``num_pts_feats``, so we
        normalise that here before delegating to the parent class.

        Args:
            info (dict): Raw frame dict from the pkl.

        Returns:
            dict: Standardised data-info dict ready for the pipeline.
        """
        data_info = dict()

        # ── LiDAR points ─────────────────────────────────────────────────────
        lidar_points = dict(info['lidar_points'])  # shallow copy

        # Normalise key name: num_pts_dim → num_pts_feats
        if 'num_pts_feats' not in lidar_points and 'num_pts_dim' in lidar_points:
            lidar_points['num_pts_feats'] = lidar_points.pop('num_pts_dim')
        elif 'num_pts_feats' not in lidar_points:
            lidar_points['num_pts_feats'] = 4  # default: x,y,z,intensity

        # Strip leading 'data/a2d2/' prefix if present — mmengine prepends
        # data_root to data_prefix, so the path must be relative to data_root.
        lp = lidar_points.get('lidar_path', '')
        if lp.startswith('data/a2d2/'):
            lidar_points['lidar_path'] = lp[len('data/a2d2/'):]

        data_info['lidar_points'] = lidar_points

        # ── Sample identity ───────────────────────────────────────────────────
        data_info['sample_idx'] = info.get('sample_idx', '')
        if 'sequence' in info:
            data_info['sequence'] = info['sequence']
        if 'timestamp' in info:
            data_info['timestamp'] = info['timestamp']

        # ── Images (carry through, pipeline ignores them if use_camera=False) ─
        if 'images' in info:
            images = {}
            for cam_id, cam_info in info['images'].items():
                cam_info = dict(cam_info)
                ip = cam_info.get('img_path', '')
                if ip.startswith('data/a2d2/'):
                    cam_info['img_path'] = ip[len('data/a2d2/'):]
                images[cam_id] = cam_info
            data_info['images'] = images

        # ── Instances ─────────────────────────────────────────────────────────
        #  Each instance dict must have:
        #    bbox_3d       : list[7]  [cx, cy, cz_bottom, l, w, h, yaw]
        #    bbox_label_3d : int      0=Car, 1=Pedestrian, 2=Cyclist
        instances = []
        for inst in info.get('instances', []):
            instance = dict(
                bbox_3d=list(inst['bbox_3d']),
                bbox_label_3d=int(inst['bbox_label_3d']),
            )
            instances.append(instance)
        data_info['instances'] = instances

        # ── Delegate path resolution + ann_info construction to parent ────────
        data_info = super().parse_data_info(data_info)

        return data_info

    def parse_ann_info(self, info: dict) -> dict:
        """Wrap gt_bboxes_3d numpy array into LiDARInstance3DBoxes.

        Frames with zero annotations return an empty ann_info (not None) so
        that Det3DDataset.prepare_data can apply filter_empty_gt properly.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            # No instances: return empty ann_info so prepare_data handles it
            return dict(
                gt_bboxes_3d=LiDARInstance3DBoxes(
                    np.zeros((0, 7), dtype=np.float32)),
                gt_labels_3d=np.array([], dtype=np.int64),
            )
        ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(
            ann_info['gt_bboxes_3d'], box_dim=7)
        return ann_info
