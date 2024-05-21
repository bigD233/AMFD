# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .thermal_first_two_stage_detector_ori import ThermalFirstTwoStageDetector


@MODELS.register_module()
class ThermalFirstCascadeRCNN(ThermalFirstTwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 fusion_module = None,
                 if_multi_gts = False,
                 if_fusion_roi = False,
                 if_patch_emb = False,
                 if_double_loss = False,
                 if_add_RGB_ROI = False) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            fusion_module = fusion_module,
            if_multi_gts = if_multi_gts,
            if_fusion_roi = if_fusion_roi,
            if_patch_emb = if_patch_emb,
            if_double_loss = if_double_loss,
            if_add_RGB_ROI = if_add_RGB_ROI)


@MODELS.register_module()
class ThermalFirstFasterRCNN(ThermalFirstTwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 fusion_module = None,
                 if_multi_gts = False,
                 if_fusion_roi = False,
                 if_patch_emb = False,
                 if_double_loss = False,
                 if_add_RGB_ROI = False) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor,
            fusion_module = fusion_module,
            if_multi_gts = if_multi_gts,
            if_fusion_roi = if_fusion_roi,
            if_patch_emb = if_patch_emb,
            if_double_loss = if_double_loss,
            if_add_RGB_ROI = if_add_RGB_ROI)