#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements the MetaBlock

If you find any bug or have some suggestion, please, email me.
"""

import torch
import torch.nn as nn


class MetaBlock(nn.Module):
    """
    Implementing the Metadata Processing Block (MetaBlock)
    """

    def __init__(self, V, U):
        super(MetaBlock, self).__init__()
        self.fb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))
        self.gb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))

    def forward(self, V, U):
        t1 = self.fb(U)
        t2 = self.gb(U)
        V = torch.sigmoid(torch.tanh(V * t1.unsqueeze(-1)) + t2.unsqueeze(-1))
        return V


class MetaBlockAdapter(nn.Module):
    def __init__(self, vision_backbone_ouput_size, n_metadata):
        super().__init__()
        self.feature_fusion = MetaBlock(vision_backbone_ouput_size // 32, n_metadata)

    def forward(self, cnn_features, metadata):
        x = cnn_features.view(
            cnn_features.size(0), cnn_features.size(1) // 32, 32, -1
        ).squeeze(
            -1
        )  # getting the feature maps
        x = self.feature_fusion(x, metadata.float())  # applying metablock
        return x.view(x.size(0), -1)  # flatting
