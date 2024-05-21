# Copyright (c) OpenMMLab. All rights reserved.
from .Fusionresnets import FusionResNet
from .Alignresnet import AlignResNet
from .doubleresnets import MultiSpecResNets
from .six_channelresnet import SixResNet
__all__ = [
    'FusionResNet','MultiSpecResNets',"SixResNet"
]
