import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from _record import _PerformanceMeter_Cityscapes, _PerformanceMeter_NYUv2, _PerformanceMeter_celebA
from utils import count_parameters
from weighting import EW, DWA, DCUW, CUW, CAGrad, PCGrad, RLW, STL, UW, GLS, DUW, UW_mlp


class Trainer(nn.Module):
    def __init__(self, task_dict, weighting, distribution, model,
                 rep_grad, optim_param, scheduler_param, writer, epochs, dataset, **kwargs):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda:0')
        self.kwargs = kwargs
        self.epochs = epochs
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())
        self.rep_grad = rep_grad
        self.dataset = dataset
        self.model = model.to(self.device, non_blocking=True)
        weighting_methods = {"EW": EW.EW, "DWA": DWA.DWA, "GLS": GLS.GLS, "DCUW": DCUW.DCUW, "CUW": CUW.CUW,
                             "UW": UW.UW, "RLW": RLW.RLW, "STL": STL.STL, "CAGrad": CAGrad.CAGrad,
                             "DCUW_CAGrad": CAGrad.DCUW_CAGrad, "UW_CAGrad": CAGrad.UW_CAGrad, "PCGrad": PCGrad.PCGrad,
                             "DCUW_PCGrad": PCGrad.DCUW_PCGrad, "UW_PCGrad": PCGrad.UW_PCGrad, "DUW": DUW.DUW,
                             "CUW_CAGrad": CAGrad.CUW_CAGrad, "CUW_PCGrad": PCGrad.CUW_PCGrad, "UW_mlp": UW_mlp.UW_mlp}
        self.weighting_method = weighting_methods[weighting]
        self._prepare_optimizer(optim_param, scheduler_param)
        if scheduler_param is not None:
            self.scheduler_name = scheduler_param["scheduler"]
        else:
            self.scheduler_name = 'None'
        self.distribution = distribution
        self.meters = {"Cityscapes": _PerformanceMeter_Cityscapes, "CelebA": _PerformanceMeter_celebA,
                       "NYUv2": _PerformanceMeter_NYUv2}
        self.meter = self.meters[self.dataset](self.task_dict, writer, self.epochs)
        self.writer = writer
        count_parameters(self.model)

    def _prepare_optimizer(self, optim_param, scheduler_param):
        optim_dict = {
                'sgd': torch.optim.SGD,
                'adam': torch.optim.Adam,
                'adagrad': torch.optim.Adagrad,
                'rmsprop': torch.optim.RMSprop,
            }
        scheduler_dict = {
                'exp': torch.optim.lr_scheduler.ExponentialLR,
                'step': torch.optim.lr_scheduler.StepLR,
                'cos': torch.optim.lr_scheduler.CosineAnnealingLR,
                'cyclic': torch.optim.lr_scheduler.CyclicLR,
                '1cycle': torch.optim.lr_scheduler.OneCycleLR,
            }
        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_param['optim']](self.model.parameters(), **optim_arg)
        if scheduler_param is not None:
            scheduler_arg = {k: v for k, v in scheduler_param.items() if k != 'scheduler'}
            self.scheduler = scheduler_dict[scheduler_param['scheduler']](self.optimizer, **scheduler_arg)
        else:
            self.scheduler = None

    def _process_data(self, loader):
        if self.dataset == "Cityscapes":
            try:
                data, sem_label, depth_label = loader[1].next()
            except:
                loader[1] = iter(loader[0])
                data, sem_label, depth_label = loader[1].next()

            label = {self.task_name[0]: sem_label,
                     self.task_name[1]: depth_label}

            data = data.to(self.device, non_blocking=True)
            label[self.task_name[0]] = label[self.task_name[0]].to(self.device, non_blocking=True)
            label[self.task_name[1]] = label[self.task_name[1]].to(self.device, non_blocking=True)
            return data, label

        elif self.dataset == "CelebA":
            try:
                label = {}
                for i, e in enumerate(loader[1].next()):
                    if i == 0:
                        data = e
                    else:
                        label[self.task_name[i-1]] = e
            except:
                loader[1] = iter(loader[0])
                label = {}
                for i, e in enumerate(loader[1].next()):
                    if i == 0:
                        data = e
                    else:
                        label[self.task_name[i-1]] = e

            data = data.to(self.device, non_blocking=True)
            for key in label:
                label[key] = label[key].to(self.device, non_blocking=True)
            return data, label

        elif self.dataset == "NYUv2":
            try:
                data, label = loader[1].next()
            except:
                loader[1] = iter(loader[0])
                data, label = loader[1].next()
            data = data.to(self.device, non_blocking=True)
            for task in self.task_name:
                label[task] = label[task].to(self.device, non_blocking=True)
            return data, label

    def _compute_loss(self, preds, gts, task_name=None):
        train_losses = torch.zeros(self.task_num).to(self.device)
        for tn, task in enumerate(self.task_name):
            train_losses[tn] = self.meter.losses[task]._update_loss(preds[task], gts[task])
        return train_losses

    def _prepare_dataloaders(self, dataloaders):
        loader = [dataloaders, iter(dataloaders)]
        return loader, len(dataloaders)

    def train(self, train_dataloaders, test_dataloaders, val_dataloaders=None, return_weight=False):
        start = time.time()
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        self.weighting = self.weighting_method(trainer=self, weight_args=self.kwargs["weight_args"],
                                               train_batch=train_batch, writer=self.writer)
        self.batch_weight = np.zeros([self.task_num, self.epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, self.epochs])

        iter_ctr = 1
        for epoch in range(self.epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')
            for batch_index in range(train_batch):
                train_inputs, train_gts = self._process_data(train_loader)
                train_preds, loss_scale = self.model(train_inputs)
                train_losses = self._compute_loss(train_preds, train_gts)
                self.meter.update(train_preds, train_gts)

                self.optimizer.zero_grad()
                task_weights = self.weighting.update_step(losses=train_losses, epoch=epoch, batch_index=batch_index,
                                                          loss_scale=loss_scale, iter_ctr=iter_ctr, model=self.model)

                if task_weights is not None:
                    for i, weight in enumerate(task_weights):
                        self.writer.add_scalar(f"weight{i+1}", weight.item(), iter_ctr)
                self.optimizer.step()
                if self.scheduler_name == "cyclic" or self.scheduler_name == "1cycle":
                    self.scheduler.step()
                iter_ctr += 1

            if self.scheduler_name == "step":
                self.scheduler.step()

            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()

            if val_dataloaders is not None:
                self.meter.has_val = True
                self.test(val_dataloaders, epoch, mode='val')
            self.test(test_dataloaders, epoch, mode='test')

        self.meter.display_best_result()
        end = time.time()
        train_time = round((end - start) / 60 / 60, 3)
        self.writer.add_scalar("/train_time", train_time)

    def test(self, test_dataloaders, epoch=None, mode='test'):
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            for batch_index in range(test_batch):
                test_inputs, test_gts = self._process_data(test_loader)
                test_preds, loss_scale = self.model(test_inputs)
                test_losses = self._compute_loss(test_preds, test_gts)
                self.meter.update(test_preds, test_gts)
        self.meter.record_time('end')
        self.meter.get_score()
        self.meter.display(epoch=epoch, mode=mode)
        self.meter.reinit()
