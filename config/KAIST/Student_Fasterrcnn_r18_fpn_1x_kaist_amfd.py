
_base_ = [
    # '../_base_/schedules/schedule_1x.py', 
    # '../_base_/default_runtime.py'
]
__code_version__='single_faster_rcnn'
plugin=True
plugin_dir='./projects/Distillation/distillation/'


temp=1
# alpha_fgd=0.00005
beta_fgd=0
# beta_fgd=0.000025
# gamma_fgd=0.00005
alpha_fgd=0
gamma_fgd=0
lambda_fgd=0.0000005
# lambda_fgd=0

custom_imports = dict(
    imports=['projects.Distillation.distillation'], allow_failed_imports=False)

work_dir='/home/featurize/work/mmdetection/work_dirs_retinanet/KAIST_' + __code_version__
# model settings
model = dict(
    type='MultiSpecFGDIrRgbFasterRCNN',
    data_preprocessor=dict(
        type='BGR3TDataPreprocessor',
        mean=[123.675, 116.28, 103.53, 135.438, 135.438, 135.438],
        std=[58.395, 57.12, 57.375, 20.7315, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='FusionResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        weight_path='/home/featurize/work/mmdetection/ckpts/resnet18-5c106cde.pth',
        ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=5,
        ),
    # teacher_cfg = '/home/featurize/work/mmdetection/projects/BAANet/configs/NEW_BAANet_r50_fpn_1x_kaist_thermal_first.py',
    # teacher_pretrained = '/home/featurize/work/mmdetection/projects/BAANet/checkpoints/best_Glare_normal_all_iter_2500.pth',
    distill_cfg = [ 
                    dict(methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd ,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd ,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd ,
                                       gamma_fgd=gamma_fgd ,
                                       lambda_fgd=lambda_fgd ,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd ,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd ,
                                       lambda_fgd=lambda_fgd ,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd ,
                                       beta_fgd=beta_fgd ,
                                       gamma_fgd=gamma_fgd ,
                                       lambda_fgd=lambda_fgd ,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_ir_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd ,
                                       beta_fgd=beta_fgd ,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_ir_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd ,
                                       beta_fgd=beta_fgd ,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_ir_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd ,
                                       beta_fgd=beta_fgd ,
                                       gamma_fgd=gamma_fgd ,
                                       lambda_fgd=lambda_fgd ,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_ir_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd ,
                                       beta_fgd=beta_fgd ,
                                       gamma_fgd=gamma_fgd ,
                                       lambda_fgd=lambda_fgd ,
                                       )
                                ]
                        ),
                   ],
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))
)

# dataset settings
dataset_type = 'KAISTDataset'
data_root = ''

backend_args = None



image_size = [(1333, 480), (1333, 512), (1333, 544), (1333, 576), (1333, 608),
              (1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768),
              (1333, 800)]
train_pipeline = [
    dict(type='LoadBGR3TFromKAIST', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                (1333, 736), (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadBGR3TFromKAIST', backend_args=None),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='KAISTDataset',
        ann_file=
        '/home/featurize/data/KAIST/kaist_test_anno/anno/train_anno/KAIST_train_RGB_annotation.json',
        pipeline=[
            dict(type='LoadBGR3TFromKAIST', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            # dict(
            #     type='RandomMask',
            #     prob=0.5,
            #     mask_type='black',
            #     mask_modality='RGB'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
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
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KAISTDataset',
        data_prefix=dict(
            img_path=
            '/home/featurize/data/KAIST/kaist_test_anno/kaist_test/kaist_test_lwir'
        ),
        ann_file=
        '/home/featurize/data/KAIST/KAIST_annotation_with_complex_light.json',
        test_mode=True,
        pipeline=[
            dict(type='LoadBGR3TFromKAIST', backend_args=None),
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
        type='GlareKAISTMissrateMetric',
        ann_file=
        '/home/featurize/data/KAIST/KAIST_annotation_with_complex_light.json',
        metric='bbox',
        format_only=False,
        backend_args=None,
    )
]
test_evaluator = [
    dict(
        type='GlareKAISTMissrateMetric',
        ann_file=
        '/home/featurize/data/KAIST/KAIST_annotation_with_complex_light.json',
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
        save_best=['Glare/normal_all'],
        rule='less'),
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
# load_from = '/home/featurize/work/mmdetection/projects/Distillation/checkpoints/best_coco_all_iter_15000.pth'
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
