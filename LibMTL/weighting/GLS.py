import torch
import sys
import os
cwd = os.getcwd()
path = cwd + '/LibMTL/weighting'
sys.path.insert(1, path)
from abstract_weighting import AbsWeighting


class GLS(AbsWeighting):
    def __init__(self, trainer, **kwargs):
        super(GLS, self).__init__(trainer, kwargs["train_batch"], kwargs["writer"])

    def update_step(self, losses, **kwargs):
        if self.task_num <= 3:
            loss = torch.pow(losses.prod(), 1. / self.task_num)
            loss.backward()
        else:
            # (1 / 2) * [log(l1 + 10**-5) + log(l2 + 10**-5)]
            constant = torch.tensor(10**-5, requires_grad=False)
            loss = (1. / self.task_num) * losses.add(constant).log().sum()
            loss.backward()
        return None
