# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union
import time
from torch import Tensor
import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.base import BaseDetector
from projects.AMFD.amfd.modules.baa_gate import DWConv
from mmengine.runner import load_checkpoint
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.config import Config

@MODELS.register_module()
class ThermalFirstSingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 fusion_module = None, 
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.backbone_ir = MODELS.build(backbone)
        self.fusion_conv = DWConv(in_channels=256 * 2, out_channels=256, ksize=3).cuda()
        if neck is not None:
            self.neck = MODELS.build(neck)
            self.neck_ir = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        if fusion_module != None:
            self.fusion_module = MODELS.build(fusion_module)

        

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    


    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x,x_ir = self.extract_feat(batch_inputs)
        losses = dict()
        
        fused_feature_maps = []
        for i in [0, 1,2, 3, 4]:
            fused_feature_maps.append(self.fusion_conv(torch.cat([x[i], x_ir[i]], dim=1)))
        # for i in [2, 3, 4]:
        #     fused_feature_maps.append(self.fusion_module[i](x[i], x_ir[i]))
        # for item in x_t:
        #     print(item.shape)
        # for item in x_t_ir:
        #     print(item.shape)
        # for item in x:
        #     print(item.shape)

        losses.update(self.bbox_head.loss(fused_feature_maps, batch_data_samples))
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x, x_ir = self.extract_feat(batch_inputs)
        fused_feature_maps = []
        for i in [0, 1,2, 3, 4]:
            fused_feature_maps.append(self.fusion_conv(torch.cat([x[i], x_ir[i]], dim=1)))

        # for i in [2, 3, 4]:
        #     fused_feature_maps.append(self.fusion_module[i](x[i], x_ir[i]))
        results_list = self.bbox_head.predict(
            fused_feature_maps, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
        # s = time.time()
        # x = self.extract_feat(batch_inputs)
        # results = self.bbox_head.forward(x)
        # e = time.time()
        # print(f"forward time:{e-s}s")
        # return results

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x, x_ir = self.extract_feat(batch_inputs)
        fused_feature_maps = []
        for i in [0, 1,2, 3, 4]:
            fused_feature_maps.append(self.fusion_conv(torch.cat([x[i], x_ir[i]], dim=1)))

        # for i in [2, 3, 4]:
        #     fused_feature_maps.append(self.fusion_module[i](x[i], x_ir[i]))
        results = self.bbox_head.forward(fused_feature_maps)
        
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs[:, :3, :, :])
        x_ir = self.backbone_ir(batch_inputs[:, 3:, :, :])
        if self.with_neck:
            x = self.neck(x)
            x_ir = self.neck_ir(x_ir)
        # NOTE 为了单模态试验
        # return x_ir, x_ir
        return x, x_ir
