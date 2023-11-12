import torch
import sys
import os
cwd = os.getcwd()
path = cwd + '/LibMTL/weighting'
sys.path.insert(1, path)
from abstract_weighting import AbsWeighting


class STL(AbsWeighting):
    def __init__(self, trainer, **kwargs):
        super(STL, self).__init__(trainer, kwargs["train_batch"], kwargs["writer"])
        self.task = kwargs["weight_args"]["task"]

    def update_step(self, losses, **kwargs):
        losses[self.task - 1].backward()

