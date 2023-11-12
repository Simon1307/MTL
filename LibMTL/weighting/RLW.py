import torch
import torch.nn.functional as F
import sys
import os
cwd = os.getcwd()
path = cwd + '/LibMTL/weighting'
sys.path.insert(1, path)
from abstract_weighting import AbsWeighting


class RLW(AbsWeighting):
    def __init__(self, trainer, **kwargs):
        super(RLW, self).__init__(trainer, kwargs["train_batch"], kwargs["writer"])
        self.distribution = kwargs["weight_args"]["distribution"]

    def update_step(self, losses, **kwargs):
        batch_weight = self.get_random_weights()
        loss = torch.mul(losses, batch_weight).sum()
        loss.backward()
        return batch_weight.detach()

    def get_random_weights(self):
        if self.distribution == "normal":
            batch_weight = F.softmax(torch.randn(self.task_num), dim=-1).to(self.device)
        elif self.distribution == "uniform":
            batch_weight = F.softmax(torch.rand(self.task_num), dim=-1).to(self.device)
        return batch_weight
