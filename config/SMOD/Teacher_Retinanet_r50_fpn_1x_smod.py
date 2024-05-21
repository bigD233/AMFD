plugin = True
plugin_dir='./projects/Distillation/distillation/'

__code_version__='double_retinanet_thermal_rpn_woExtraconv'
work_dir = '/home/featurize/work/mmdetection/work_dirs_smct/SMCT_' + __code_version__

custom_imports = dict(
    imports=['projects.Distillation.distillation'], allow_failed_imports=False)

model = dict(
    type='ThermalFirstRetinaNet',
    # fusion_module = [dict(
    #     type='TransformerFusionModule',
    #     feat_size = feat_size[:2],
    #     d_model = 256,
    #     num_layers = 4
    #     ) for feat_size in [(100, 128, 16), (50, 64, 3), (25, 32, 3), (13, 16, 3),(7, 8, 3)]],
    data_preprocessor=dict(
        type='BGR3TDataPreprocessor',
        mean=[123.675, 116.28, 103.53, 135.438, 135.438, 135.438],
        std=[58.395, 57.12, 57.375, 57.12, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        # add_extra_convs='on_input',
        start_level=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg = None,
        ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=5,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.41, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(
            type='PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
)
dataset_type = 'KAISTDataset'
data_root = ''
backend_args = None

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='SMCTDataset',
        data_prefix=dict(
            img_path=
            '/home/featurize/data/SMCT'
        ),
        ann_file=
        '/home/featurize/data/SMCT/new_train_annotations_rgb.json',
        pipeline=[
            dict(type='LoadBGR3TFromSMCT', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            # dict(
            #     type='RandomMask',
            #     prob=0.5,
            #     mask_type='black',
            #     mask_modality='RGB'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            # dict(
            #     type='RandomChoiceResize',
            #     scales=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
            #             (1333, 608), (1333, 640), (1333, 672), (1333, 704),
            #             (1333, 736), (1333, 768), (1333, 800)],
            #     keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'flip', 'flip_direction',
                        #    'RGB_mask'
                           ))
        ],
        backend_args=None))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='SMCTDataset',
        data_prefix=dict(
            img_path=
            '/home/featurize/data/SMCT'
        ),
        ann_file=
        '/home/featurize/data/SMCT/new_test_annotations_rgb.json',
        test_mode=True,
        pipeline=[
            dict(type='LoadBGR3TFromSMCT', backend_args=None),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=
        '/home/featurize/data/SMCT/new_test_annotations_rgb.json',
        metric='bbox',
        format_only=False,
        backend_args=None,
    )
]
test_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=
        '/home/featurize/data/SMCT/new_test_annotations_rgb.json',
        metric='bbox',
        format_only=False,
        backend_args=None,
    )
]

default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_best=['coco/bbox_mAP'],
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
log_level = 'INFO'
resume = False
train_cfg = dict(type='IterBasedTrainLoop', max_iters=30000, val_interval=600)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        T_max=30000,
        eta_min_ratio=0.1,
        begin=0,
        end=30000,
        by_epoch=False)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
launcher = 'none'
seed = 0
