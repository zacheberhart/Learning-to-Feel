import numpy as np
import torch
from torch import *
from torch.autograd import *
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms

class BCELoss(nn.Module):
    def __init__(self, reduce=False):
        super().__init__()
        self.reduce = reduce

    def forward(self, logit, target):
        target = target.float()
        loss = nn.BCEWithLogitsLoss()(logit, target)
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        if not self.reduce:
            return loss
        else:
            return loss.mean()


# Adapted from https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduce=False):
        super().__init__()
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        if not self.reduce:
            return loss
        else:
            return loss.mean()

class MixupBCELoss(BCELoss):
    def forward(self, x, y):
        if isinstance(y, dict):
            y0, y1, a = y['y0'], y['y1'], y['a']
            loss = a*super().forward(x, y0) + (1-a)*super().forward(x, y1)
        else:
            loss = super().forward(x, y)
        return 100*loss.mean()


class MixupFocalLoss(FocalLoss):
    def forward(self, x, y):
        if isinstance(y, dict):
            y0, y1, a = y['y0'], y['y1'], y['a']
            loss = a*super().forward(x, y0) + (1-a)*super().forward(x, y1)
        else:
            loss = super().forward(x, y)
        return loss.mean()