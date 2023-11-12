import numpy as np
import torch
import sys
from scipy.optimize import minimize
import os
cwd = os.getcwd()
path = cwd + '/LibMTL/weighting'
sys.path.insert(1, path)
from abstract_weighting import AbsWeighting


class CAGrad(AbsWeighting):
    def __init__(self, trainer, **kwargs):
        AbsWeighting.__init__(self, trainer, kwargs["train_batch"], kwargs["writer"])
        self.rescale = kwargs["weight_args"]["rescale"]
        self.calpha = kwargs["weight_args"]["calpha"]
        self.grad_dims = []
        # Determine gradient dimensions for shared modules
        for mm in trainer.model.shared_modules():
            for param in mm.parameters():
                self.grad_dims.append(param.data.numel())
        self.grads = torch.zeros((sum(self.grad_dims), self.task_num), requires_grad=True).to(self.device)

    def update_step(self, losses, **kwargs):
        model = kwargs["model"]
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
        # modify and overwrite gradients of shared modules according to CAGrad
        g = self.cagrad(self.grads, self.calpha, self.rescale)
        self.overwrite_grad(model, g, self.grad_dims)

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

    def cagrad(self, grads, alpha=0.5, rescale=0):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.task_num) / self.task_num
        bnds = tuple((0, 1) for x in x_start)
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
        A = GG.detach().numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (x.reshape(1, self.task_num).dot(A).dot(b.reshape(self.task_num, 1)) + c * np.sqrt(
                x.reshape(1, self.task_num).dot(A).dot(x.reshape(self.task_num, 1)) + 1e-8)).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return g
        elif rescale == 1:
            return g / (1 + alpha ** 2)
        else:
            return g / (1 + alpha)

    def overwrite_grad(self, m, newgrad, grad_dims):
        newgrad = newgrad * self.task_num  # to match the sum loss
        cnt = 0
        for mm in m.shared_modules():
            for param in mm.parameters():
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                this_grad = newgrad[beg:en].contiguous().view(param.data.size())
                param.grad = this_grad.data.clone()
                cnt += 1


class UW_CAGrad(CAGrad):
    def __init__(self, trainer, **kwargs):
        CAGrad.__init__(self, trainer, **kwargs)

    def update_step(self, losses, **kwargs):
        loss_scale = kwargs["loss_scale"]
        losses = (losses / (2 * loss_scale.exp()) + loss_scale / 2)
        super().update_step(losses, **kwargs)
        return loss_scale.detach()


class DCUW_CAGrad(CAGrad):
    def __init__(self, trainer, **kwargs):
        CAGrad.__init__(self, trainer, **kwargs)

    def update_step(self, losses, **kwargs):
        loss_scale = kwargs["loss_scale"]
        iter_ctr = kwargs["iter_ctr"]
        model = kwargs["model"]
        epoch = kwargs["epoch"]
        losses = (losses / (2 * loss_scale.exp()) + loss_scale / 2)
        super().update_step(losses, **kwargs)
        return loss_scale.detach()


class CUW_CAGrad(CAGrad):
    def __init__(self, trainer, **kwargs):
        CAGrad.__init__(self, trainer, **kwargs)

    def update_step(self, losses, **kwargs):
        loss_scale = kwargs["loss_scale"]
        iter_ctr = kwargs["iter_ctr"]
        model = kwargs["model"]
        epoch = kwargs["epoch"]
        losses = (losses / (2 * loss_scale.exp()) + loss_scale / 2)
        super().update_step(losses, **kwargs)
        return loss_scale.detach()
