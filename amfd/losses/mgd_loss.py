# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS

    
@MODELS.register_module()
class MGDLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation.

    <https://arxiv.org/abs/2205.01529>`

    Args:
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
    """

    def __init__(self, name,
                 student_channels,
                 teacher_channels,
                 lambda_mgd: float = 0.65,
                 alpha_mgd: float = 0.00002,
                 mask_on_channel: bool = False) -> None:
        super(MGDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.loss_mse = nn.MSELoss(reduction='sum')

        self.lambda_mgd = lambda_mgd
        self.mask_on_channel = mask_on_channel
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(
                student_channels,
                teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(
                teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                teacher_channels, teacher_channels, kernel_size=3, padding=1))

    def mask_perd_S(self,preds_S: torch.Tensor) -> torch.Tensor:
        
        if self.align is not None:
            preds_S = self.align(preds_S)

        N, C, H, W = preds_S.shape

        device = preds_S.device
        if not self.mask_on_channel:
            mat = torch.rand((N, 1, H, W)).to(device)
        else:
            mat = torch.rand((N, C, 1, 1)).to(device)

        mat = torch.where(mat > 1 - self.lambda_mgd,
                          torch.zeros(1).to(device),
                          torch.ones(1).to(device)).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)
        return new_fea

    def forward(self, preds_S: torch.Tensor,
                preds_T: torch.Tensor,
                batch_data_samples) -> torch.Tensor:
        """Forward function.

        Args:
            preds_S(torch.Tensor): Bs*C*H*W, student's feature map
            preds_T(torch.Tensor): Bs*C*H*W, teacher's feature map

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape == preds_T.shape
        mask_S = self.mask_perd_S(preds_S)
        loss = self.get_dis_loss(mask_S, preds_T) * self.alpha_mgd

        return loss

    def get_dis_loss(self, preds_S: torch.Tensor,
                     preds_T: torch.Tensor) -> torch.Tensor:
        """Get MSE distance of preds_S and preds_T.

        Args:
            preds_S(torch.Tensor): Bs*C*H*W, student's feature map
            preds_T(torch.Tensor): Bs*C*H*W, teacher's feature map

        Return:
            torch.Tensor: The calculated mse distance value.
        """
        N, C, H, W = preds_T.shape
        dis_loss = self.loss_mse(preds_S, preds_T) / N

        return dis_loss
