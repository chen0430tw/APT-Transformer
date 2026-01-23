#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预训练模块

提供各种自监督学习方法用于模型预训练

作者: chen0430tw
"""

from .contrastive_pretrain import (
    ContrastivePretrainer,
    ContrastiveConfig,
    create_contrastive_pretrainer
)

from .mlm_pretrain import (
    MLMPretrainer,
    MLMConfig,
    create_mlm_pretrainer
)

__all__ = [
    'ContrastivePretrainer',
    'ContrastiveConfig',
    'create_contrastive_pretrainer',
    'MLMPretrainer',
    'MLMConfig',
    'create_mlm_pretrainer',
]
