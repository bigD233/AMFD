import copy
import warnings
from typing import List, Tuple, Union, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from mmengine.runner import load_checkpoint
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.config import Config

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.base import BaseDetector

from mmdet.apis import init_backbone_neck
from pathlib import Path


def FMSE_loss(current_bacth, true_batch):
    batch_size,C,H,W = current_bacth.shape
    device = 'cuda:0'


    d = torch.sum((current_bacth-true_batch)**2, dim=1, keepdim=True)

    soft_d = F.softmax(d.view(batch_size, 1, -1), dim=2)
    
    soft_d = soft_d.view(d.size())
    # print(soft_d.max())
    # print(soft_d.min())
    loss_matrix = soft_d * d

    loss = torch.sum(loss_matrix)
    loss = loss/C

    return loss



@MODELS.register_module()
class MultiSpecDistillDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: ConfigType,
                 enable_distilled = False,
                 distilled_file_config:  Union[str, Path, Config] = None,
                 distilled_checkpoint:  Optional[str] = None,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.backbone = MODELS.build(backbone)
        # import pdb; pdb.set_trace()
        self.enable_distilled = enable_distilled

        self.distilled_file_config = distilled_file_config
        self.distilled_checkpoint = distilled_checkpoint

        # if self.enable_distilled:
        #     self.distilled_backbone , self.distilled_neck = init_backbone_neck(distilled_file_config,distilled_checkpoint)
        
        if neck is not None:
            self.neck = MODELS.build(neck)

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

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if self.enable_distilled:
            self.distilled_backbone, self.distilled_neck = self._load_distilled_weights()

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
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)


    def _load_distilled_weights(self, device: str = 'cuda:0'):
        config = Config.fromfile(self.distilled_file_config)

        model_backbone = MODELS.build(config.model.backbone)
        model_neck = MODELS.build(config.model.neck)
        model_neck = revert_sync_batchnorm(model_neck)
        model_backbone = revert_sync_batchnorm(model_backbone)

        load_checkpoint(model_backbone, self.distilled_checkpoint, map_location='cpu',revise_keys=[(r'^backbone\.', '')])
        load_checkpoint(model_neck, self.distilled_checkpoint, map_location='cpu',revise_keys=[(r'^neck\.', '')])

        return model_backbone,model_neck
    
    def _mask_loss(self,mask,batch_data_samples):
        current_shape =mask.shape
        scale = 1
        mask_gt = torch.zeros(current_shape,device='cuda:0')
        # print(current_shape)

        for i in range(len(batch_data_samples)):

            bboxes = batch_data_samples[i].gt_instances.bboxes
            # bboxes = torch.floor(bboxes/scale)            

            for bbox in bboxes:
                xmin, ymin, xmax, ymax = bbox
                mask_gt[i, :, int(ymin):int(ymax), int(xmin):int(xmax)] = 1

        dice_loss =1-  (2*torch.sum(mask*mask_gt)+0.0001)/(torch.sum(mask)+torch.sum(mask_gt)+0.0001)
        loss = dict()
        loss['mask_loss'] = dice_loss

        return loss



    def _distilled_loss(self,batch_inputs: Tensor,x)-> Tuple[Tensor]:

        x_truth = self.distilled_backbone(batch_inputs)
        x_truth = self.distilled_neck(x_truth)
        
        masks = []
        
        

        distilled_loss = 0
        for i in range(len(x)):

            distilled_loss += FMSE_loss(x[i],x_truth[i])
        distilled_loss = distilled_loss/len(x)
        loss = dict()
        loss['distilled_loss'] = distilled_loss
        return loss



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
        x = self.backbone(batch_inputs)
        # for i in x :
        #     print(i.shape)
        # when batch size is 4 , x is a tuple with tensors below.
        # torch.Size([4, 256, 128, 160])
        # torch.Size([4, 512, 64, 80])
        # torch.Size([4, 1024, 32, 40])
        # torch.Size([4, 2048, 16, 20])
        

        if self.with_neck:
            x = self.neck(x)


        # for i in x :
        #     print(i.shape)
        # print(mask.shape)
        # torch.Size([1, 256, 152, 192])
        # torch.Size([1, 256, 76, 96])
        # torch.Size([1, 256, 38, 48])
        # torch.Size([1, 256, 19, 24])
        # torch.Size([1, 256, 10, 12])

        return x

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
        x, mask = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_outs = self.roi_head.forward(x, rpn_results_list,
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
        x = self.extract_feat(batch_inputs)
        # print(x[0].shape)

        # print(batch_data_samples[0])
        # bboxes = batch_data_samples[0].gt_instances.bboxes
        # print(torch.floor(bboxes/8))
        
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        # mask_loss = self._mask_loss(mask,batch_data_samples)
        # losses.update(mask_loss)
        if self.enable_distilled:
            dis_loss = self._distilled_loss(batch_inputs,x)
            losses.update(dis_loss)

        # print(losses)
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
        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
