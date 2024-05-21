#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from mmdet.registry import MODELS
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.ops import DeformConv2d
import math
from mmengine.utils import to_2tuple
import torch.nn.functional as F

class DConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=2, padding=1, bias=False):
        super(DConv, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.conv2 = DeformConv2d(inplanes, planes, kernel_size=kernel_size,
                                  stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x, out)
        return out

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

@MODELS.register_module()
class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class ChannelDistilling(nn.Module):
    def __init__(
        self,
        channel_num,
        reduction=8
    ):
        super().__init__()
        self.channel_num = channel_num
        self.reduction = reduction

        self.avg_gap = nn.AdaptiveAvgPool2d(1)
        self.max_gap = nn.AdaptiveMaxPool2d(1)

        self.dense1 = nn.Linear(int(4 * channel_num), int(channel_num // reduction))
        self.bn1 = nn.BatchNorm1d(int(channel_num // reduction))
        self.act1 = nn.ReLU(inplace=True)
        self.dense2 = nn.Linear(int(channel_num // reduction), int(channel_num))
        self.bn2 = nn.BatchNorm1d(int(channel_num))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.bn3 = nn.BatchNorm1d(int(channel_num))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x_ref, w_feat):
        x_cat = torch.cat([x, x_ref], 1)
        x_cat_avg = self.avg_gap(x_cat)
        x_cat_max = self.max_gap(x_cat)
        x_cat = torch.cat([x_cat_avg, x_cat_max], dim=1)
        x_cat = torch.flatten(x_cat, start_dim=1)
        x_cat = self.dense1(x_cat)
        x_cat = self.bn1(x_cat)
        x_cat = self.act1(x_cat)
        ch_w = self.dense2(x_cat)
        ch_w = self.bn2(ch_w)
        ch_w = torch.sigmoid(ch_w)
        ch_w = torch.unsqueeze(ch_w, 2)
        ch_w = torch.unsqueeze(ch_w, 3)
        com_ch = torch.mul(x_ref, ch_w)
        com_ch_w = torch.mul(com_ch, w_feat)
        channel_feat = self.bn3(x + com_ch_w)
        return channel_feat

class SpatialAggregation(nn.Module):
    def __init__(
        self,
        channel_num,
    ):
        super().__init__()
        self.channel_num = channel_num

        self.conv_rgb = BaseConv(2, out_channels=1, ksize=1, stride=1)
        self.conv_ir = BaseConv(2, out_channels=1, ksize=1, stride=1)
        self.gamma_1 = nn.Parameter(torch.zeros(1))
        self.gamma_2 = nn.Parameter(torch.zeros(1))

        self.bn_rgb = nn.BatchNorm2d(1)
        self.bn_ir = nn.BatchNorm2d(1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_rgb, x_ir, w_rgb, w_ir):
        x_cat = torch.cat([x_rgb, x_ir], 1)
        w_rs = self.conv_rgb(torch.cat([
            x_cat.mean(dim=1, keepdims=True),
            x_cat.max(dim=1, keepdims=True)[0]],
            dim=1)
            ) # [4, 1, 200, 256]
        w_ts = self.conv_ir(torch.cat([
            x_cat.mean(dim=1, keepdims=True),
            x_cat.max(dim=1, keepdims=True)[0]],
            dim=1)
            ) # [4, 1, 200, 256]
        
        w_rs = torch.sigmoid(w_rs)
        w_ts = torch.sigmoid(w_ts)

        x_rgb_sa = torch.mul(w_rs, x_rgb) +  x_rgb# [4, 256, 200, 256]
        x_ir_sa = torch.mul(w_ts, x_ir) + x_ir # [4, 256, 200, 256]

        x_rgb_sa_illu = self.bn_rgb(torch.mul(w_rgb, x_rgb_sa))
        x_ir_sa_illiu = self.bn_ir(torch.mul(w_ir, x_ir_sa))
        return x_rgb_sa_illu, x_ir_sa_illiu

@MODELS.register_module()
class BAAGate(nn.Module):
    def __init__(
        self,
        channel_num,
        reduction=8,
        act="silu",
    ):
        super().__init__()
        self.channel_num = channel_num
        self.reduction = reduction
        self.act = act

        self.cd1 = ChannelDistilling(self.channel_num, self.reduction)
        self.cd2 = ChannelDistilling(self.channel_num, self.reduction)
        self.sa = SpatialAggregation(self.channel_num)
        self.gamma_1 = nn.Parameter(torch.zeros(1))
        self.gamma_2 = nn.Parameter(torch.zeros(1))

        self.rgb_dcn = DConv(channel_num, channel_num, 3, 1, 1)
        self.ir_dcn = DConv(channel_num, channel_num, 3, 1, 1)

    def forward(self, input_features_rgb, input_features_ir, w_rgb, w_ir):
        R_rec = self.cd1(input_features_rgb, input_features_ir, w_ir)
        T_rec = self.cd2(input_features_ir, input_features_rgb, w_rgb)

        [input_features_rgb, input_features_ir] = self.sa(R_rec, T_rec, w_rgb, w_ir)

        # input_features_rgb = self.rgb_dcn(input_features_rgb)
        # input_features_ir = self.ir_dcn(input_features_ir)

        return [input_features_rgb, input_features_ir]

@MODELS.register_module()
class IlluminationNetwork(nn.Module):
    def __init__(
        self,
        k1=0.5,
        k2=3,
        act="silu",
        depthwise=False,
    ):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        Conv = DWConv if depthwise else BaseConv

        # layers for illumination network
        self.resized_rgb = transforms.Resize([60, 80])
        self.resized_ir = transforms.Resize([60, 80])

        self.rgb_conv1 = Conv(3, 128, 3, 2, act=act)
        self.ir_conv1 = Conv(3, 128, 3, 2, act=act)

        self.rgb_conv2 = Conv(128, 64, 3, 2, act=act)
        self.ir_conv2 = Conv(128, 64, 3, 2, act=act)

        self.rgb_conv3 = Conv(64, 32, 3, 2, act=act)
        self.ir_conv3 = Conv(64, 32, 3, 2, act=act)

        self.dense1 = nn.Linear(64 * 8 * 10, 128)
        self.dense2 = nn.Linear(128, 1)
        self.softmax = nn.Softmax(dim=1)
        self.k = {"False": 3, "True": 0.5}

    def forward(self, x, x_ir):
        resized_rgb = self.resized_rgb(x)
        resized_ir = self.resized_ir(x_ir)

        resized_rgb = self.rgb_conv1(resized_rgb)
        resized_ir = self.ir_conv1(resized_ir)

        resized_rgb = self.rgb_conv2(resized_rgb)
        resized_ir = self.ir_conv2(resized_ir)

        resized_rgb = self.rgb_conv3(resized_rgb)
        resized_ir = self.ir_conv3(resized_ir)

        rgb_ir = torch.cat([resized_rgb, resized_ir], 1)
        rgb_ir = torch.flatten(rgb_ir, start_dim=1)
        rgb_ir = self.dense1(rgb_ir)
        rgb_ir = torch.sigmoid(rgb_ir)
        w_d = self.dense2(rgb_ir)
        w_d = torch.sigmoid(w_d)
        w_n = 1 - w_d

        # rgb_ir = torch.sigmoid(rgb_ir)
        deltas = 2 * (w_d - w_n)

        k_list = [self.k[str((i[0]).cpu().numpy())] for i in deltas > 0]
        k_list = torch.from_numpy(np.array(k_list)).unsqueeze(1)
        w_r = 1 / (1 + torch.exp(-k_list.cuda() * deltas))
        w_t = 1 - w_r
        illu_output = torch.cat([w_d, w_n, w_r, w_t], 1)
        return illu_output

@MODELS.register_module()
class TransformerFusionModule(nn.Module):
    def __init__(
        self,
        feat_size,
        num_layers = 4,
        d_model = 256,
        num_heads = 8
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model * 2,
            nhead = num_heads,
            dim_feedforward = d_model * 2,
        )
        self.fusion_conv = DWConv(in_channels=256 * 2, out_channels=256, ksize=3)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        max_len = feat_size[0] * feat_size[1]
        pe = torch.zeros(max_len, d_model * 2)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        self.pe = pe.unsqueeze(0).cuda()

    def forward(self, x, x_ir):
        # 维度变为 B C S
        B, C, H, W = x.size() #BCHW

        x_hw = x.view(B, C, H * W) # B C HW
        x_ir_hw = x_ir.view(B, C, H * W) # B C HW
        fusion_feat = torch.cat([x_hw, x_ir_hw], dim = 1) # B 2C HW
        # import pdb;pdb.set_trace()
        # print(fusion_feat.shape)
        # print(self.pe.permute(0, 2, 1).shape)
        
        fusion_feat = fusion_feat + self.pe.permute(0, 2, 1) # [1, 512, 800]
        fusion_feat = self.encoder(fusion_feat.permute(2, 0, 1))
        # import pdb; pdb.set_trace()
        fusion_feat = fusion_feat.view(B, 2 * C, H, W) # [4, 512, 25, 32]
        fusion_feat = self.fusion_conv(fusion_feat) # [4, 256, 25, 32]

        return fusion_feat

@MODELS.register_module()
class MultiLevelTransformerFusionModule(nn.Module):
    def __init__(
        self,
        max_len,
        d_model = 256,
        num_heads = 8
    ):
        super().__init__()
        num_layers = 3
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model * 2,
            nhead = num_heads,
            dim_feedforward = d_model * 2,
        )
        self.fusion_conv = DWConv(in_channels=256 * 2, out_channels=256, ksize=3)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        pe = torch.zeros(max_len, d_model * 2)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        self.pe = pe.unsqueeze(0).cuda()

    def forward(self, x, x_ir):
        # 维度变为 B C S
        B, C, H, W = x.size() #BCHW
        x_hw = x.view(B, C, H * W) # B C HW
        x_ir_hw = x_ir.view(B, C, H * W) # B C HW
        fusion_feat = torch.cat([x_hw, x_ir_hw], dim = 1) # B 2C HW

        fusion_feat = fusion_feat + self.pe.permute(0, 2, 1) # [1, 512, 51200]
        fusion_feat = self.encoder(fusion_feat.permute(2, 0, 1))
        # import pdb; pdb.set_trace()
        fusion_feat = fusion_feat.view(B, 2 * C, H, W)
        fusion_feat = self.fusion_conv(fusion_feat)

        return fusion_feat

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # 14, 14
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # 14 * 14 = 196
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        # 256, 32, 25 -> 256, 10, 8
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x) # B, 256, 200, 256 -> B, 1024, 12, 16
        if self.flatten:
            x = x.flatten(2)  # B, 1024, 12, 16 -> B, 1024(D), 12*16(N)
        x = self.norm(x)
        return x

@MODELS.register_module()
class VisionTransformerFusionModule(nn.Module):
    def __init__(
        self,
        feat_size,
        patch_size,
        num_layers = 4,
        d_model = 256,
        num_heads = 8
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = num_heads,
            dim_feedforward = d_model,
        )
        self.d_model = d_model
        self.fusion_conv = DWConv(in_channels=256 * 2, out_channels=256, ksize=3)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        max_len = (feat_size[0] // patch_size) * (feat_size[1] // patch_size) * 2
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        self.pe = pe.unsqueeze(0).cuda()

        self.patch_emb_1 = PatchEmbed(feat_size, patch_size=patch_size, in_chans=256, embed_dim=d_model, norm_layer=None, flatten=True)
        self.patch_emb_2 = PatchEmbed(feat_size, patch_size=patch_size, in_chans=256, embed_dim=d_model, norm_layer=None, flatten=True)

        self.conv_transpose_rgb = nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2, padding=0)
        self.conv_transpose_ir = nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2, padding=0)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)


    def forward(self, x, x_ir):
        # 维度变为 B C S
        B, C, H, W = x.size() #BCHW
        x_patch_emb = self.patch_emb_1(x) # B, C, H, W -> B, D, N(H / patch_size * W / patch_size) [4, 256, 80]
        x_ir_patch_emb = self.patch_emb_2(x_ir)

        fusion_feat = torch.cat([x_patch_emb, x_ir_patch_emb], dim = 2) # B D 2*N [4, 256, 80*2]

        fusion_feat = fusion_feat + self.pe.permute(0, 2, 1) # [160, 4, 256]
        fusion_feat = self.encoder(fusion_feat.permute(2, 0, 1)) # [160(N), 4, 256(D)]
        fusion_feat_rgb = fusion_feat.permute(1, 2, 0)[:, :, :self.patch_emb_1.num_patches]\
                                    .view(B, self.d_model, self.patch_emb_1.grid_size[0], self.patch_emb_1.grid_size[1]) # [4, 256, 80] -> [4, 256, 8, 10]
        fusion_feat_ir = fusion_feat.permute(1, 2, 0)[:, :, self.patch_emb_1.num_patches:]\
                                    .view(B, self.d_model, self.patch_emb_1.grid_size[0], self.patch_emb_1.grid_size[1]) # [4, 256, 80] -> [4, 256, 8, 10]

        fusion_feat_rgb = self.conv_transpose_rgb(fusion_feat_rgb)
        fusion_feat_ir = self.conv_transpose_ir(fusion_feat_ir)

        # fusion_feat_rgb = F.interpolate(fusion_feat_rgb, scale_factor=2, mode='bilinear', align_corners=False)
        # fusion_feat_ir = F.interpolate(fusion_feat_ir, scale_factor=2, mode='bilinear', align_corners=False)

        # fusion_feat = torch.cat([fusion_feat_rgb, fusion_feat_ir], dim = 1)
        fusion_feat = fusion_feat_rgb + fusion_feat_ir

        return fusion_feat
    

@MODELS.register_module()
class ConfidenceFusionModule(nn.Module):
    def __init__(
        self,
        in_channels = 256,
        mid_channels = 256
    ):
        super().__init__()
        self.in_channels = in_channels
        self.internal_conf_rgb = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )
        self.internal_conf_ir = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )
        self.interaction_conf = nn.Sequential(
            nn.Conv2d(2 * in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )

    def forward(self, x, x_ir):
        conf_mask_rgb = torch.sigmoid(self.internal_conf_rgb(x))
        conf_mask_ir_in = torch.sigmoid(self.internal_conf_rgb(x_ir))

        cat_feat = torch.cat([x, x_ir], dim = 1)
        conf_mask_interaction = torch.sigmoid(self.interaction_conf(cat_feat))

        x_w = conf_mask_rgb * x

        conf_mask_ir = (conf_mask_rgb * conf_mask_interaction + conf_mask_ir_in) * 0.5
        x_ir_w = conf_mask_ir * x_ir

        fusion_feat = x_w + x_ir_w

        return fusion_feat
