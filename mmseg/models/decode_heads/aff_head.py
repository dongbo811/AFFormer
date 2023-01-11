'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class CLS(BaseDecodeHead):
    def __init__(self,
                 aff_channels=512,
                 aff_kwargs=dict(),
                 **kwargs):
        super(CLS, self).__init__(
                input_transform='multiple_select', **kwargs)
        self.aff_channels = aff_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        

        self.align = ConvModule(
            self.aff_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)[0]

        x = self.squeeze(inputs)
            
        output = self.cls_seg(x)
        return output
