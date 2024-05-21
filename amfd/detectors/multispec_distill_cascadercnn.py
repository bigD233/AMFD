# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .multispec_three_stage_detector import MultiSpecDistillDetector
# from .two_stage_fgd import TwoStageFGDDetector
from .two_stage_distill_thermal_first_rpn import TwoStageFGDThermalRPNFstDetector
from .two_stage_distill_thermal_first import TwoStageFGDThermalFstDetector


@MODELS.register_module()
class MultiSpecFGDCascadeRCNN(TwoStageFGDThermalFstDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 distill_cfg: OptConfigType = None,
                 teacher_cfg: OptConfigType = None,
                 teacher_pretrained: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            distill_cfg=distill_cfg,
            teacher_cfg = teacher_cfg,
            teacher_pretrained = teacher_pretrained,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

@MODELS.register_module()
class MultiSpecFGDFasterRCNN(TwoStageFGDThermalFstDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 distill_cfg: OptConfigType = None,
                 teacher_cfg: OptConfigType = None,
                 teacher_pretrained: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            distill_cfg=distill_cfg,
            teacher_cfg = teacher_cfg,
            teacher_pretrained = teacher_pretrained,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)