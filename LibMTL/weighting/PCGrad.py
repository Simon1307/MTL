import torch, random
import sys
import os
cwd = os.getcwd()
path = cwd + '/LibMTL/weighting'
sys.path.insert(1, path)
from abstract_weighting import AbsWeighting


class PCGrad(AbsWeighting):
    def __init__(self, trainer, **kwargs):
        AbsWeighting.__init__(self, trainer, kwargs["train_batch"], kwargs["writer"])
        self.grad_dims = []
        # Determine gradient dimensions for shared modules
        for mm in trainer.model.shared_modules():
            for param in mm.parameters():
                self.grad_dims.append(param.data.numel())
        self.grads = torch.zeros((sum(self.grad_dims), self.task_num), requires_grad=True).to(self.device)

    def update_step(self, losses, **kwargs):
        model = kwargs["model"]
        batch_weight = torch.ones_like(losses).to(self.device)
        # compute gradients with individual task losses
        for i in range(self.task_num):
            losses[i].backward(retain_graph=True)

            # modify grads tensor in place
            self.grad2vec(
                model,
                self.grads,
                self.grad_dims,
                i,
                detach=False
            )
            model.zero_grad_shared_modules()  # set gradients of shared modules to 0
        pc_grads = self.grads.clone()
        for tn_i in range(self.task_num):
            task_index = list(range(self.task_num))
            random.shuffle(task_index)
            for tn_j in task_index:
                g_ij = torch.dot(pc_grads[tn_i], self.grads[tn_j])
                if g_ij < 0:
                    pc_grads[tn_i] -= g_ij * self.grads[tn_j] / (self.grads[tn_j].norm().pow(2))
                    batch_weight[tn_j] -= (g_ij/(self.grads[tn_j].norm().pow(2))).item()
        new_grads = pc_grads.sum(dim=1)
        self.overwrite_grad(model, new_grads, self.grad_dims)
        return batch_weight

    def grad2vec(self, m, grads, grad_dims, task, detach=True):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        for mm in m.shared_modules():
            for p in mm.parameters():
                grad = p.grad
                if grad is not None:
                    if detach:
                        grad_cur = grad.data.detach().clone()
                    else:
                        grad_cur = grad
                    beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                    en = sum(grad_dims[: cnt + 1])
                    if detach:  # in place modification
                        grads[beg:en, task].copy_(grad_cur.data.view(-1))
                    else:  # not in place but with gradient tape
                        grads[beg:en, task] = grad_cur.view(-1)

                cnt += 1

    def overwrite_grad(self, m, newgrad, grad_dims):
        #newgrad = newgrad * self.task_num  # to match the sum loss
        cnt = 0
        for mm in m.shared_modules():
            for param in mm.parameters():
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                this_grad = newgrad[beg:en].contiguous().view(param.data.size())
                param.grad = this_grad.data.clone()
                cnt += 1


class UW_PCGrad(PCGrad):
    def __init__(self, trainer, **kwargs):
        PCGrad.__init__(self, trainer, **kwargs)

    def update_step(self, losses, **kwargs):
        loss_scale = kwargs["loss_scale"]
        losses = (losses / (2 * loss_scale.exp()) + loss_scale / 2)
        _ = super().update_step(losses, **kwargs)
        return loss_scale.detach()


class DCUW_PCGrad(PCGrad):
    def __init__(self, trainer, **kwargs):
        PCGrad.__init__(self, trainer, **kwargs)

    def update_step(self, losses, **kwargs):
        loss_scale = kwargs["loss_scale"]
        iter_ctr = kwargs["iter_ctr"]
        model = kwargs["model"]
        epoch = kwargs["epoch"]
        losses = (losses / (2 * loss_scale.exp()) + loss_scale / 2)
        _ = super().update_step(losses, **kwargs)
        return loss_scale.detach()


class CUW_PCGrad(PCGrad):
    def __init__(self, trainer, **kwargs):
        PCGrad.__init__(self, trainer, **kwargs)

    def update_step(self, losses, **kwargs):
        loss_scale = kwargs["loss_scale"]
        iter_ctr = kwargs["iter_ctr"]
        model = kwargs["model"]
        epoch = kwargs["epoch"]
        losses = (losses / (2 * loss_scale.exp()) + loss_scale / 2)
        _ = super().update_step(losses, **kwargs)
        return loss_scale.detach()
