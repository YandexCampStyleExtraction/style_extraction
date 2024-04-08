# https://colab.research.google.com/drive/1FObv_7kot3X-F3Cf7RTGWAd28SEbzAM7#scrollTo=7z-hUC-OGRac

from typing import Union

import torch
import torch.nn as nn
from torch.nn import TripletMarginWithDistanceLoss


class ContrastiveLoss(nn.Module):

    def __init__(self, smooth=0.0):
        super().__init__()
        self.H = torch.nn.CrossEntropyLoss(label_smoothing=smooth, reduction='mean')

    def forward(self, predicted_sim, gt_sim, T: Union[float, nn.Parameter] = 0.08):
        return self.H(predicted_sim / T, gt_sim)


class ArcFace(nn.Module):
    # https://arxiv.org/pdf/1801.07698.pdf
    # https://github.com/deepinsight/insightface
    pass


# https://arxiv.org/pdf/1901.05903.pdf
