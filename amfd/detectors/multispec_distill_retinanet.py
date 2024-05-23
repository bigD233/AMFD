# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage_distill_align_necks import SingleStageDistillAllDetector
from .single_stage_distill import SingleStageDistillDetector

@MODELS.register_module()
class MultiSpecRetinaNet(SingleStageDistillDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 distill_cfg: OptConfigType = None,
                 teacher_cfg: OptConfigType = None,
                 teacher_pretrained: OptConfigType = None,
                 multi_teacher: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            distill_cfg=distill_cfg,
            teacher_cfg = teacher_cfg,
            teacher_pretrained = teacher_pretrained,
            multi_teacher = multi_teacher,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)


@MODELS.register_module()
class MultiSpecAMFDRetinaNet(SingleStageDistillAllDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 distill_cfg: OptConfigType = None,
                 teacher_cfg: OptConfigType = None,
                 teacher_pretrained: OptConfigType = None,
                 multi_teacher: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            distill_cfg=distill_cfg,
            teacher_cfg = teacher_cfg,
            teacher_pretrained = teacher_pretrained,
            multi_teacher = multi_teacher,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)


@MODELS.register_module()
class MultiSpecDistillAllATSS(SingleStageDistillAllDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 distill_cfg: OptConfigType = None,
                 teacher_cfg: OptConfigType = None,
                 teacher_pretrained: OptConfigType = None,
                 multi_teacher: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            distill_cfg=distill_cfg,
            teacher_cfg = teacher_cfg,
            teacher_pretrained = teacher_pretrained,
            multi_teacher = multi_teacher,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)