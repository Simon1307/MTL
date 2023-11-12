import torch.nn.functional as F
import torch
import sys
import os
cwd = os.getcwd()
path = cwd + '/LibMTL/weighting'
sys.path.insert(1, path)
from abstract_weighting import AbsWeighting


class DWA(AbsWeighting):
    def __init__(self, trainer, **kwargs):
        super(DWA, self).__init__(trainer, kwargs["train_batch"], kwargs["writer"])
        self.train_loss_buffer = torch.zeros(size=(self.epochs, self.train_batch, self.task_num),
                                             requires_grad=False).to(self.device)
        self.T = torch.tensor(kwargs["weight_args"]["T"], requires_grad=False).to(self.device)

    def update_step(self, losses, **kwargs):
        epoch = kwargs["epoch"]
        batch_index = kwargs["batch_index"]

        self.train_loss_buffer[epoch][batch_index] = losses.clone().detach()
        if epoch > 1:
            r_k = (self.train_loss_buffer[epoch - 1].sum(dim=0) / self.train_batch) / (
                        self.train_loss_buffer[epoch - 2].sum(dim=0) / self.train_batch)
            w_k = self.task_num * F.softmax(r_k / self.T, dim=-1)
        else:
            w_k = torch.ones_like(losses).to(self.device)
        loss = torch.mul(losses, w_k).sum()
        loss.backward()
        return w_k
