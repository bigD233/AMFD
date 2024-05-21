# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Tuple, Union
from mmengine.structures import InstanceData
import torch
from torch import Tensor
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.base import BaseDetector
from projects.BAANet.baanet.modules.baa_gate import DWConv
from mmdet.models.layers import multiclass_nms
from mmcv.ops import batched_nms
from mmdet.structures.bbox import (cat_boxes, empty_box_as, get_box_tensor,
                                   get_box_wh, scale_boxes)
@MODELS.register_module()
class ThermalFirstTwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

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
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.backbone_ir = MODELS.build(backbone)
        self.fusion_conv = DWConv(in_channels=256 * 2, out_channels=256, ksize=3).cuda()
        self.if_multi_gts = if_multi_gts
        self.if_fusion_roi = if_fusion_roi
        self.if_patch_emb = if_patch_emb
        self.if_double_loss = if_double_loss
        self.if_add_RGB_ROI = if_add_RGB_ROI

        if neck is not None:
            self.neck = MODELS.build(neck)
            self.neck_ir = MODELS.build(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)
            self.rpn_head_rgb = MODELS.build(rpn_head_)
            self.rpn_head_fusion = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)
            self.roi_head_rgb = MODELS.build(roi_head)
            self.roi_head_ir = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.fusion_module = MODELS.build(fusion_module) 
        # 256, 200, 256; 256, 100, 128; 256, 50, 64; 256, 25, 32; 256, 13, 16

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading single-stage
        weights into two-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) != 0 and len(rpn_head_keys) == 0:
            for bbox_head_key in bbox_head_keys:
                rpn_head_key = rpn_head_prefix + \
                               bbox_head_key[len(bbox_head_prefix):]
                state_dict[rpn_head_key] = state_dict.pop(bbox_head_key)
        def is_substring_of_key(dict_obj, sub_str):
            return any(sub_str in key for key in dict_obj.keys())
        if not is_substring_of_key(state_dict, 'backbone_ir'):
            ori_dict = copy.deepcopy(state_dict)
            for k, v in ori_dict.items():
                if k.split('.')[0] == 'backbone':
                    state_dict[k.replace('backbone', 'backbone_ir')] = v
                if k.split('.')[0] == 'neck':
                    state_dict[k.replace('neck', 'neck_ir')] = v
                if k.split('.')[0] == 'rpn_head':
                    state_dict[k.replace('rpn_head', 'rpn_head_fusion')] = v
                if k.split('.')[0] == 'roi_head':
                    state_dict[k.replace('roi_head', 'roi_head_rgb')] = v
                    state_dict[k.replace('roi_head', 'roi_head_ir')] = v
            del ori_dict
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

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

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        x, x_ir = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x_ir, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        # 将特征图送入融合模块
        fused_feature_maps = []
        for i in [0,1,2]:
            fused_feature_maps.append(self.fusion_conv(torch.cat([x[i], x_ir[i]], dim=1)))

        for i in [3,4]:
            fused_feature_maps.append(self.fusion_module[i-3](x[i], x_ir[i]))
        roi_outs = self.roi_head.forward(fused_feature_maps, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs, )
        return results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x, x_ir = self.extract_feat(batch_inputs)

        losses = dict()

        fused_feature_maps = []
        # 将特征图送入融合模块
        if self.if_patch_emb:
            for i in [0, 1]:
                fused_feature_maps.append(self.fusion_conv(torch.cat([x[i], x_ir[i]], dim=1)))
            for i in [2, 3, 4]:
                fused_feature_maps.append(self.fusion_module[i](x[i], x_ir[i]))
        else:
            for i in [0,1,2]:
                fused_feature_maps.append(self.fusion_conv(torch.cat([x[i], x_ir[i]], dim=1)))

            for i in [3, 4]:
                fused_feature_maps.append(self.fusion_module[i-3](x[i], x_ir[i]))

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            if self.if_multi_gts:
                # use gt_instances_ir as RPN gts
                # set cat_id of gt_labels to 0 in RPN
                for data_sample in rpn_data_samples:
                    data_sample.gt_instances_ir.labels = \
                        torch.zeros_like(data_sample.gt_instances_ir.labels)
                    data_sample.gt_instances = data_sample.gt_instances_ir
                    data_sample.ignored_instances = data_sample.ignored_instances_ir
            else:
                # set cat_id of gt_labels to 0 in RPN
                for data_sample in rpn_data_samples:
                    data_sample.gt_instances.labels = \
                        torch.zeros_like(data_sample.gt_instances.labels)

            # load ir rois and loss
            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x_ir, rpn_data_samples, proposal_cfg=proposal_cfg)

            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                rpn_losses[f'ir_rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)

            if self.if_add_RGB_ROI:
                # load RGB rois and loss
                rpn_data_samples_for_RGB = copy.deepcopy(rpn_data_samples)
                for sample in rpn_data_samples_for_RGB:
                    if sample.RGB_mask == True:
                        sample.gt_instances.bboxes = HorizontalBoxes([])
                        sample.gt_instances.labels = torch.tensor([], device='cuda:0', dtype=torch.int64)
                rpn_losses_rgb, rpn_results_list_rgb = self.rpn_head_rgb.loss_and_predict(
                    x, rpn_data_samples_for_RGB, proposal_cfg=proposal_cfg)
                # for i in range(len(rpn_results_list)):
                #     rpn_results_list[i].bboxes = torch.cat([rpn_results_list[i].bboxes[:1000], 
                #                                             rpn_results_list_rgb[i].bboxes[:1000]])# 2000, 4
                #     rpn_results_list[i].labels = torch.cat([rpn_results_list[i].labels[:1000], 
                #                                             rpn_results_list_rgb[i].labels[:1000]])# 2000
                #     rpn_results_list[i].scores = torch.cat([rpn_results_list[i].scores[:1000], 
                #                                             rpn_results_list_rgb[i].scores[:1000]])# 2000
                #     # 使用argsort对score进行排序
                #     sorted_indices = torch.argsort(rpn_results_list[i].scores, descending=True)
                #     # 根据排序后的索引对score、bboxes和labels重新排序
                #     rpn_results_list[i].scores = rpn_results_list[i].scores[sorted_indices]
                #     rpn_results_list[i].bboxes = rpn_results_list[i].bboxes[sorted_indices]
                #     rpn_results_list[i].labels = rpn_results_list[i].labels[sorted_indices]

                # avoid get same name with roi_head loss
                keys = rpn_losses_rgb.keys()
                for key in list(keys):
                    rpn_losses_rgb[f'rgb_rpn_{key}'] = rpn_losses_rgb.pop(key)
                losses.update(rpn_losses_rgb)              

            # if introduce fusion roi into rois.
            rpn_results_list_ir = None
            if self.if_fusion_roi:
                rpn_losses_fusion, rpn_results_list_fusion = self.rpn_head_fusion.loss_and_predict(
                    fused_feature_maps, rpn_data_samples, proposal_cfg=proposal_cfg)
                keys = rpn_losses_fusion.keys()
                rpn_results_list_ir = copy.deepcopy(rpn_results_list)
                for i in range(len(rpn_results_list)):
                    rpn_results_list[i].bboxes = torch.cat([rpn_results_list[i].bboxes, 
                                                            rpn_results_list_fusion[i].bboxes])# 2000, 4
                    rpn_results_list[i].level_ids = torch.cat([rpn_results_list[i].level_ids, 
                                                            rpn_results_list_fusion[i].level_ids])# 2000
                    rpn_results_list[i].scores = torch.cat([rpn_results_list[i].scores, 
                                                            rpn_results_list_fusion[i].scores])# 2000
                    if type(self.rpn_head_fusion).__name__ == "RPNHeadWoNMS":
                        if rpn_results_list[i].bboxes.numel() > 0:
                            bboxes = get_box_tensor(rpn_results_list[i].bboxes)
                            det_bboxes, keep_idxs = batched_nms(bboxes, rpn_results_list[i].scores,
                                                                rpn_results_list[i].level_ids, proposal_cfg.nms)
                            rpn_results_list[i] = rpn_results_list[i][keep_idxs]
                            # some nms would reweight the score, such as softnms
                            rpn_results_list[i].scores = det_bboxes[:, -1]
                            rpn_results_list[i] = rpn_results_list[i][:proposal_cfg.max_per_img]
                            # TODO: This would unreasonably show the 0th class label
                            #  in visualization
                            rpn_results_list[i].labels = rpn_results_list[i].scores.new_zeros(
                                len(rpn_results_list[i]), dtype=torch.long)
                            del rpn_results_list[i].level_ids
                        else:
                            # To avoid some potential error
                            results_ = InstanceData()
                            results_.bboxes = empty_box_as(rpn_results_list[i].bboxes)
                            results_.scores = rpn_results_list[i].scores.new_zeros(0)
                            results_.labels = rpn_results_list[i].scores.new_zeros(0)
                            rpn_results_list[i] = results_

                for key in list(keys):
                    rpn_losses_fusion[f'fusion_rpn_{key}'] = rpn_losses_fusion.pop(key)
                losses.update(rpn_losses_fusion)        

        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        # NOTE: need to cancle comments here
        
        # rgb
        if self.if_add_RGB_ROI:
            roi_losses_rgb = self.roi_head_rgb.loss(x, rpn_results_list_rgb,
                                rpn_data_samples_for_RGB)
        else:
            rpn_results_list_rgb = copy.deepcopy(rpn_results_list)
            batch_data_samples_rgb = copy.deepcopy(batch_data_samples)
            roi_losses_rgb = self.roi_head_rgb.loss(x, rpn_results_list_rgb,
                                            batch_data_samples_rgb)
        keys = roi_losses_rgb.keys()
        for key in list(keys):
            if 'loss' in key:
                roi_losses_rgb[f'rgb_roi_{key}'] = roi_losses_rgb.pop(key)
        # if self.if_double_loss:
        #     for k, v in roi_losses_rgb.items():
        #         roi_losses_rgb[k] = v * 1.5
        losses.update(roi_losses_rgb)
        

        # ir
        batch_data_samples_ir = copy.deepcopy(batch_data_samples)
        
        # rpn_results_list_rgb = copy.deepcopy(rpn_results_list)
        

        if self.if_multi_gts:
            for data_sample in batch_data_samples_ir:
                data_sample.gt_instances = data_sample.gt_instances_ir
                data_sample.ignored_instances = data_sample.ignored_instances_ir
        if rpn_results_list_ir is not None:
            roi_losses_ir = self.roi_head_ir.loss(x_ir, rpn_results_list_ir,
                                        batch_data_samples_ir)
        else:
            rpn_results_list_ir = copy.deepcopy(rpn_results_list)
            roi_losses_ir = self.roi_head_ir.loss(x_ir, rpn_results_list_ir,
                            batch_data_samples_ir)
        keys = roi_losses_ir.keys()
        for key in list(keys):
            if 'loss' in key:
                roi_losses_ir[f'ir_roi_{key}'] = roi_losses_ir.pop(key)
        losses.update(roi_losses_ir)
        
        
        roi_losses = self.roi_head.loss(fused_feature_maps, rpn_results_list,
                                        batch_data_samples_ir)
        if self.if_double_loss:
            for k, v in roi_losses.items():
                roi_losses[k] = v * 2
        losses.update(roi_losses)

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
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x, x_ir = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x_ir, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        # 将特征图送入融合模块
        fused_feature_maps = []
        # 将特征图送入融合模块
        if self.if_patch_emb:
            for i in [0, 1]:
                fused_feature_maps.append(self.fusion_conv(torch.cat([x[i], x_ir[i]], dim=1)))
            for i in [2, 3, 4]:
                fused_feature_maps.append(self.fusion_module[i](x[i], x_ir[i]))
        else:
            for i in [0,1,2]:
                fused_feature_maps.append(self.fusion_conv(torch.cat([x[i], x_ir[i]], dim=1)))

            for i in [3, 4]:
                fused_feature_maps.append(self.fusion_module[i-3](x[i], x_ir[i]))

        if self.if_fusion_roi:
            # load RGB rois and loss
            rpn_results_list_fusion = self.rpn_head_fusion.predict(
                fused_feature_maps, batch_data_samples, rescale=False)
            for i in range(len(rpn_results_list)):
                rpn_results_list[i].bboxes = torch.cat([rpn_results_list[i].bboxes, 
                                                        rpn_results_list_fusion[i].bboxes])# 2000, 4
                rpn_results_list[i].level_ids = torch.cat([rpn_results_list[i].level_ids, 
                                                        rpn_results_list_fusion[i].level_ids])# 2000
                rpn_results_list[i].scores = torch.cat([rpn_results_list[i].scores, 
                                                        rpn_results_list_fusion[i].scores])# 2000
                if type(self.rpn_head_fusion).__name__ == "RPNHeadWoNMS":
                    proposal_cfg=dict(
                        nms_pre=1000,
                        max_per_img=1000,
                        nms=dict(type='nms', iou_threshold=0.6),
                        min_bbox_size=0)
                    if rpn_results_list[i].bboxes.numel() > 0:
                        bboxes = get_box_tensor(rpn_results_list[i].bboxes)
                        det_bboxes, keep_idxs = batched_nms(bboxes, rpn_results_list[i].scores,
                                                            rpn_results_list[i].level_ids, proposal_cfg['nms'])
                        rpn_results_list[i] = rpn_results_list[i][keep_idxs]
                        # some nms would reweight the score, such as softnms
                        rpn_results_list[i].scores = det_bboxes[:, -1]
                        rpn_results_list[i] = rpn_results_list[i][:proposal_cfg['max_per_img']]
                        # TODO: This would unreasonably show the 0th class label
                        #  in visualization
                        rpn_results_list[i].labels = rpn_results_list[i].scores.new_zeros(
                            len(rpn_results_list[i].scores), dtype=torch.long)
                        del rpn_results_list[i].level_ids
                    else:
                        # To avoid some potential error
                        results_ = InstanceData()
                        results_.bboxes = empty_box_as(rpn_results_list[i].scores.bboxes)
                        results_.scores = rpn_results_list[i].scores.scores.new_zeros(0)
                        results_.labels = rpn_results_list[i].scores.scores.new_zeros(0)
                        rpn_results_list[i].scores = results_

        results_list = self.roi_head.predict(
            fused_feature_maps, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
