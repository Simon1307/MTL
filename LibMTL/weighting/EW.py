import torch
import sys
import os
cwd = os.getcwd()
path = cwd + '/LibMTL/weighting'
sys.path.insert(1, path)
from abstract_weighting import AbsWeighting


class EW(AbsWeighting):
    def __init__(self, trainer, **kwargs):
        super(EW, self).__init__(trainer, kwargs["train_batch"], kwargs["writer"])
        
    def update_step(self, losses, **kwargs):
        batch_weight = torch.ones_like(losses).to(self.device)
        loss = torch.mul(losses, batch_weight).sum()
        loss.backward()
        return batch_weight.detach()
