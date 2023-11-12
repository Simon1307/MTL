import sys
import os
cwd = os.getcwd()
path = cwd + '/LibMTL/weighting'
sys.path.insert(1, path)
from abstract_weighting import AbsWeighting


class UW_mlp(AbsWeighting):
    def __init__(self, trainer, **kwargs):
        AbsWeighting.__init__(self, trainer, kwargs["train_batch"], kwargs["writer"])

    def update_step(self, losses, **kwargs):
        loss_scale = kwargs["loss_scale"]
        loss = (losses / (2 * loss_scale.exp()) + loss_scale / 2).sum()
        loss.backward()
        return loss_scale.detach()
