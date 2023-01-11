'''
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/model/segmentors/__init__.py
'''
from .base import BaseSegmentor
from .cascade_encoder_decoder import CascadeEncoderDecoder
from .encoder_decoder import EncoderDecoder

__all__ = ['BaseSegmentor', 'EncoderDecoder', 'CascadeEncoderDecoder']
