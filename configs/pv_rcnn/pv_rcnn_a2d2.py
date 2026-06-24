# PVRCNN on A2D2 (5-class: Car, Truck, Bus, Pedestrian, Cyclist)
# Adapted from pv_rcnn_8xb2-80e_kitti-3d-3class.py.
# Key differences:
#   - A2D2Dataset (plain pkl, LiDAR in KITTI frame: x=fwd, y=left, z=up)
#   - No GT-database / ObjectSample augmentation
#   - batch_size=16 per GPU, 2 GPUs → 32 total
#   - 80-epoch cyclic schedule (doubled from KITTI 40e baseline)
#   - Checkpoint saved every 5 epochs, keep only 1 copy
#   - lr=0.001 (kept at KITTI default for training stability)

_base_ = [
    '../_base_/default_runtime.py',
]

# ── Data ──────────────────────────────────────────────────────────────────────
data_root = 'data/a2d2/'
dataset_type = 'A2D2Dataset'
class_names = ['Car', 'Truck', 'Bus', 'Pedestrian', 'Cyclist']
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

point_cloud_range = [0, -40, -3, 70.4, 40, 1]

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # No ObjectSample: A2D2 has no ground-truth database
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=point_cloud_range),
        ]),
    dict(type='Pack3DDetInputs', keys=['points']),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='a2d2_infos_train.pkl',
        data_prefix=dict(pts='', img=''),
        pipeline=train_pipeline,
        modality=input_modality,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        filter_empty_gt=True,
        test_mode=False,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='a2d2_infos_val.pkl',
        data_prefix=dict(pts='', img=''),
        pipeline=test_pipeline,
        modality=input_modality,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        test_mode=True,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'a2d2_infos_val.pkl',
    metric='bbox',
    format_only=True,
    submission_prefix='/cluster/datastore/aniketag/mmdetection3d/work_dirs/pv_rcnn_a2d2/val_results',
    backend_args=backend_args)
test_evaluator = val_evaluator

# ── Model (PVRCNN, 3 classes) ─────────────────────────────────────────────────
voxel_size = [0.05, 0.05, 0.1]

model = dict(
    type='PointVoxelRCNN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        encoder_paddings=((0, 0, 0), ((1, 1, 1), 0, 0), ((1, 1, 1), 0, 0),
                          ((0, 1, 1), 0, 0)),
        return_middle_feats=True),
    points_encoder=dict(
        type='VoxelSetAbstraction',
        num_keypoints=2048,
        fused_out_channel=128,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        voxel_sa_cfgs_list=[
            dict(
                type='StackedSAModuleMSG',
                in_channels=16,
                scale_factor=1,
                radius=(0.4, 0.8),
                sample_nums=(16, 16),
                mlp_channels=((16, 16), (16, 16)),
                use_xyz=True),
            dict(
                type='StackedSAModuleMSG',
                in_channels=32,
                scale_factor=2,
                radius=(0.8, 1.2),
                sample_nums=(16, 32),
                mlp_channels=((32, 32), (32, 32)),
                use_xyz=True),
            dict(
                type='StackedSAModuleMSG',
                in_channels=64,
                scale_factor=4,
                radius=(1.2, 2.4),
                sample_nums=(16, 32),
                mlp_channels=((64, 64), (64, 64)),
                use_xyz=True),
            dict(
                type='StackedSAModuleMSG',
                in_channels=64,
                scale_factor=8,
                radius=(2.4, 4.8),
                sample_nums=(16, 32),
                mlp_channels=((64, 64), (64, 64)),
                use_xyz=True),
        ],
        rawpoints_sa_cfgs=dict(
            type='StackedSAModuleMSG',
            in_channels=1,
            radius=(0.4, 0.8),
            sample_nums=(16, 16),
            mlp_channels=((16, 16), (16, 16)),
            use_xyz=True),
        bev_feat_channel=256,
        bev_scale_factor=8),
    backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    rpn_head=dict(
        type='PartA2RPNHead',
        num_classes=5,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        dir_offset=0.78539,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [0, -40.0, -0.6,  70.4, 40.0, -0.6],   # Car
                [0, -40.0, -1.5,  70.4, 40.0, -1.5],   # Truck cz_bottom=-1.5
                [0, -40.0, -1.3,  70.4, 40.0, -1.3],   # Bus   cz_bottom=-1.3
                [0, -40.0, -0.6,  70.4, 40.0, -0.6],   # Pedestrian
                [0, -40.0, -1.78, 70.4, 40.0, -1.78],  # Cyclist
            ],
            sizes=[
                [3.95, 2.00, 1.71],   # Car  (top=-0.6+1.71=1.11 ≈ z_max ✓)
                [9.30, 3.09, 2.50],   # Truck capped h=2.5 (top=-1.5+2.5=1.0 ✓)
                [8.55, 2.98, 2.30],   # Bus   capped h=2.3 (top=-1.3+2.3=1.0 ✓)
                [0.97, 0.77, 1.82],   # Pedestrian
                [1.87, 0.81, 1.64],   # Cyclist
            ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        assigner_per_size=True,
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.2)),
    roi_head=dict(
        type='PVRCNNRoiHead',
        num_classes=5,
        semantic_head=dict(
            type='ForegroundSegmentationHead',
            in_channels=640,
            extra_width=0.1,
            loss_seg=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                reduction='sum',
                gamma=2.0,
                alpha=0.25,
                activated=True,
                loss_weight=1.0)),
        bbox_roi_extractor=dict(
            type='Batch3DRoIGridExtractor',
            grid_size=6,
            roi_layer=dict(
                type='StackedSAModuleMSG',
                in_channels=128,
                radius=(0.8, 1.6),
                sample_nums=(16, 16),
                mlp_channels=((64, 64), (64, 64)),
                use_xyz=True,
                pool_mod='max')),
        bbox_head=dict(
            type='PVRCNNBBoxHead',
            in_channels=128,
            grid_size=6,
            num_classes=5,
            class_agnostic=True,
            shared_fc_channels=(256, 256),
            reg_channels=(256, 256),
            cls_channels=(256, 256),
            dropout_ratio=0.3,
            with_corner_loss=True,
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss',
                beta=1.0 / 9.0,
                reduction='sum',
                loss_weight=1.0),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=[
                dict(  # Car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6, neg_iou_thr=0.45,
                    min_pos_iou=0.45, ignore_iof_thr=-1),
                dict(  # Truck
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55, neg_iou_thr=0.40,
                    min_pos_iou=0.40, ignore_iof_thr=-1),
                dict(  # Bus
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55, neg_iou_thr=0.40,
                    min_pos_iou=0.40, ignore_iof_thr=-1),
                dict(  # Pedestrian
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5, neg_iou_thr=0.35,
                    min_pos_iou=0.35, ignore_iof_thr=-1),
                dict(  # Cyclist
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5, neg_iou_thr=0.35,
                    min_pos_iou=0.35, ignore_iof_thr=-1),
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=9000,
            nms_post=512,
            max_num=512,
            nms_thr=0.8,
            score_thr=0,
            use_rotate_nms=True),
        rcnn=dict(
            assigner=[
                dict(  # Car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55, neg_iou_thr=0.55,
                    min_pos_iou=0.55, ignore_iof_thr=-1),
                dict(  # Truck
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55, neg_iou_thr=0.55,
                    min_pos_iou=0.55, ignore_iof_thr=-1),
                dict(  # Bus
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55, neg_iou_thr=0.55,
                    min_pos_iou=0.55, ignore_iof_thr=-1),
                dict(  # Pedestrian
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55, neg_iou_thr=0.55,
                    min_pos_iou=0.55, ignore_iof_thr=-1),
                dict(  # Cyclist
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55, neg_iou_thr=0.55,
                    min_pos_iou=0.55, ignore_iof_thr=-1),
            ],
            sampler=dict(
                type='IoUNegPiecewiseSampler',
                num=128,
                pos_fraction=0.5,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True),
            cls_pos_thr=0.75,
            cls_neg_thr=0.25)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1024,
            nms_post=100,
            max_num=100,
            nms_thr=0.7,
            score_thr=0,
            use_rotate_nms=True),
        rcnn=dict(
            use_rotate_nms=True,
            use_raw_score=True,
            nms_thr=0.1,
            score_thr=0.1)))

# ── Optimiser & LR schedule (80-epoch cyclic) ─────────────────────────────────
# Phase 1: epochs 0→30  — cosine warm-up from lr to lr*10
# Phase 2: epochs 30→80 — cosine decay from lr*10 to lr*1e-4
# Momentum mirrors in reverse.
lr = 0.001  # AdamW handles large batch well — no LR scaling needed
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=5, norm_type=2))

param_scheduler = [
    # Warm-up: epochs 0→100, lr rises from lr to lr*10
    dict(
        type='CosineAnnealingLR',
        T_max=100,
        eta_min=lr * 10,
        begin=0,
        end=100,
        by_epoch=True,
        convert_to_iter_based=True),
    # Decay: epochs 100→1000, lr falls from lr*10 to lr*1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=900,
        eta_min=lr * 1e-4,
        begin=100,
        end=1000,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=100,
        eta_min=0.85 / 0.95,
        begin=0,
        end=100,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=900,
        eta_min=1,
        begin=100,
        end=1000,
        by_epoch=True,
        convert_to_iter_based=True),
]

# ── Training / validation / test config ───────────────────────────────────────
train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=9999)
val_cfg = dict()
test_cfg = dict()

# ── Hooks ─────────────────────────────────────────────────────────────────────
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=20,
        save_best=None,
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

# ── Prediction video hook (fires at epochs 50,100,200,400,600,800,1000) ──────
REPO = '/cluster/datastore/aniketag/mmdetection3d'
custom_hooks = [
    dict(
        type='PredVideoHook',
        script=REPO + '/tools/make_pred_video.py',
        config=REPO + '/configs/pv_rcnn/pv_rcnn_a2d2.py',
        data_root=REPO + '/data/a2d2/',
        num_frames=60,
        score_thr=0.3,
        out_dir=REPO + '/outputs/pred_videos',
    )
]

# ── Misc ──────────────────────────────────────────────────────────────────────
auto_scale_lr = dict(enable=False, base_batch_size=64)  # 16 GPUs × batch 4

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Allow some params to skip gradient when all ROIs are empty in a batch
model_wrapper_cfg = dict(find_unused_parameters=True)
