import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
cwd = os.getcwd()
path = cwd + '/LibMTL'
sys.path.insert(1, path)
from metrics import AbsMetric
from loss import AbsLoss


class AttributeMetric(AbsMetric):
    def __init__(self):
        super(AttributeMetric, self).__init__()
        self.accuracy = 0.0
        self.num_updates = 0.0

    def update_fun(self, pred, gt):
        predictions = pred.data.max(1, keepdim=True)[1]
        self.accuracy += (predictions.eq(gt.data.view_as(predictions)).cpu().sum())  # count number of correct predictions
        self.num_updates += predictions.shape[0]

    def score_fun(self):
        acc_err = (1 - (self.accuracy / self.num_updates)) * 100
        return [acc_err.item()]

    def reinit(self):
        """Reinitialize accuracy and update counter after every epoch"""
        self.accuracy = 0.0
        self.num_updates = 0.0


class AttributeLoss(AbsLoss):
    def __init__(self):
        super(AttributeLoss, self).__init__()
        self.loss_fn = F.nll_loss

    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt.long())
