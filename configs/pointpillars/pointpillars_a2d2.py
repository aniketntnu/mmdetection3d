# PointPillars on A2D2 (5-class: Car, Truck, Bus, Pedestrian, Cyclist)
# Reuses same PKL files, A2D2Dataset, point_cloud_range, and anchor sizes as pv_rcnn_a2d2.py.
# Runs on hpc1-8 (RTX 8000, sm_75) using allEnv/mmdet3d_normal.
#
# Pillar grid: voxel_size=[0.16,0.2,4], output_shape=[400,440]
#   y: (-40 to 40) / 0.2 = 400  (divisible by 8 — required by SECONDFPN)
#   x: (0 to 70.4) / 0.16 = 440 (divisible by 8)

_base_ = ['../_base_/default_runtime.py']

# ── Data (identical to pv_rcnn_a2d2.py) ─────────────────────────────────────
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
    batch_size=48,
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
    submission_prefix='/cluster/datastore/aniketag/mmdetection3d/work_dirs/pointpillars_a2d2/val_results',
    backend_args=backend_args)
test_evaluator = val_evaluator

# ── Model ────────────────────────────────────────────────────────────────────
# voxel_size y=0.2 so grid y=80/0.2=400 (divisible by 8 — required by SECONDFPN)
# voxel_size x=0.16 so grid x=70.4/0.16=440 (divisible by 8)
voxel_size = [0.16, 0.2, 4]

model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[400, 440]),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=5,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        dir_offset=0.78539,
        assign_per_class=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [0, -40.0, -0.6,  70.4, 40.0, -0.6],   # Car
                [0, -40.0, -1.5,  70.4, 40.0, -1.5],   # Truck  cz_bottom=-1.5
                [0, -40.0, -1.3,  70.4, 40.0, -1.3],   # Bus    cz_bottom=-1.3
                [0, -40.0, -0.6,  70.4, 40.0, -0.6],   # Pedestrian
                [0, -40.0, -1.78, 70.4, 40.0, -1.78],  # Cyclist
            ],
            sizes=[
                [3.95, 2.00, 1.71],   # Car         (top=-0.6+1.71=1.11 ✓)
                [9.30, 3.09, 2.50],   # Truck capped (top=-1.5+2.5=1.0 ✓)
                [8.55, 2.98, 2.30],   # Bus   capped (top=-1.3+2.3=1.0 ✓)
                [0.97, 0.77, 1.82],   # Pedestrian
                [1.87, 0.81, 1.64],   # Cyclist
            ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
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
    train_cfg=dict(
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
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))

# ── Optimiser & LR schedule (same 1000-epoch cyclic as pv_rcnn_a2d2.py) ─────
lr = 0.001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=5, norm_type=2))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=100,
        eta_min=lr * 10,
        begin=0,
        end=100,
        by_epoch=True,
        convert_to_iter_based=True),
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

# ── Training config ──────────────────────────────────────────────────────────
train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=9999)
val_cfg = dict()
test_cfg = dict()

# ── Hooks ────────────────────────────────────────────────────────────────────
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best=None,
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

# ── Misc ─────────────────────────────────────────────────────────────────────
auto_scale_lr = dict(enable=False, base_batch_size=48)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

model_wrapper_cfg = dict(find_unused_parameters=True)

# ── Prediction video hook ─────────────────────────────────────────────────────
REPO = '/cluster/datastore/aniketag/mmdetection3d'
custom_hooks = [
    dict(
        type='PredVideoHook',
        script=REPO + '/tools/make_pred_video.py',
        config=REPO + '/configs/pointpillars/pointpillars_a2d2.py',
        data_root=REPO + '/data/a2d2/',
        num_frames=60,
        score_thr=0.3,
        out_dir=REPO + '/outputs/pred_videos_pp',
        python='/cluster/datastore/aniketag/allEnv/mmdet3d_normal/bin/python',
        partition='normal',
        nodelist='hpc2',
        epochs=[5, 25, 50, 75, 100, 200, 400, 600, 800, 1000],
    )
]
