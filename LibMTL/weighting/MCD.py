import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
cwd = os.getcwd()
path = cwd + '/LibMTL/weighting'
sys.path.insert(1, path)
from abstract_weighting import AbsWeighting


class MCD(AbsWeighting):
    def __init__(self, trainer, **kwargs):
        super(MCD, self).__init__(trainer, kwargs["train_batch"], kwargs["writer"])
        self.uncertainty_measure = kwargs["weight_args"]["uncertainty_measure"]
        self.num_samples = kwargs["weight_args"]["num_samples"]
        self.sigma_scaling = kwargs["weight_args"]["sigma_scaling"]
        self.train_bs = kwargs["train_bs"]
        self.img_height = kwargs["img_height"]
        self.img_width = kwargs["img_width"]

    def update_step(self, losses, **kwargs):
        model = kwargs["model"]
        train_preds = kwargs["train_preds"]
        train_inputs = kwargs["train_inputs"]
        train_gts = kwargs["train_gts"]

        # store all predictions across forward passes
        preds_depth_all = torch.zeros(size=(self.num_samples, self.train_bs, self.img_height, self.img_width),
                                      dtype=torch.float32,
                                      requires_grad=False).to(self.device)

        probs_seg_gts_class_all = torch.zeros(size=(self.num_samples, self.train_bs, self.img_height, self.img_width),
                                              dtype=torch.float32,
                                              requires_grad=False).to(self.device)

        k = 0
        preds_depth_all, probs_seg_gts_class_all, gts_seg, gts_depth = self.save_preds(preds_depth_all,
                                                                                       probs_seg_gts_class_all,
                                                                                       train_preds, train_gts, k)

        # Multiple forward passes
        for k in range(1, self.num_samples):
            with torch.no_grad():
                train_preds, _ = model(train_inputs)
                preds_depth_all, probs_seg_gts_class_all, gts_seg, gts_depth = self.save_preds(preds_depth_all,
                                                                                               probs_seg_gts_class_all,
                                                                                               train_preds, train_gts,
                                                                                               k)

        # Determine task-specific uncertainties
        seg_uncertainty = self.get_uncertainty_std_seg(probs_seg_gts_class_all, gts_seg)
        depth_uncertainty = self.get_uncertainty_std_depth(preds_depth_all, gts_depth)
        if self.sigma_scaling:  # sigmas sum to 1
            uncertainty_sum = torch.add(seg_uncertainty, depth_uncertainty)
            seg_uncertainty = torch.div(seg_uncertainty, uncertainty_sum)
            depth_uncertainty = torch.div(depth_uncertainty, uncertainty_sum)

        eps = torch.tensor(1e-08, requires_grad=False).to(self.device)
        loss = ((1 / seg_uncertainty.pow(2) + eps) * losses[0]) + (
                    (1 / (2 * depth_uncertainty.pow(2) + eps)) * losses[1])
        loss.backward()
        loss_scale = torch.zeros(2, requires_grad=False).to(self.device)
        loss_scale[0] = (1 / seg_uncertainty.clone().detach().pow(2) + eps)
        loss_scale[1] = (1 / (2 * depth_uncertainty.clone().detach().pow(2) + eps))
        return loss_scale.detach()

    def save_preds(self, preds_depth_all, probs_seg_gts_class_all, train_preds, train_gts, k):
        # .clone() to get new memory address for tensor and .detach() to remove from graph
        preds_seg = train_preds["segmentation"].clone().detach()  # [batch_size, 7, 128, 256]
        probs_seg = F.softmax(preds_seg, dim=1)
        # add a new channel dimension / new fictitious class for all unlabeld pixels and assign "probability" 2. to mask them later
        fictitious_class = torch.full((self.train_bs, 1, self.img_height, self.img_width), 2.).to(self.device)
        probs_seg = torch.cat((probs_seg, fictitious_class), 1)  # [batch_size, 8, 128, 256]
        gts_seg = train_gts["segmentation"].clone().detach()
        gts_seg[gts_seg == -1.] = 7.  # Assign class 7 to all unlabeled pixels (pixels with label -1.)
        preds_depth = train_preds["depth"].clone().detach()  # [batch_size, 1, 128, 256]
        gts_depth = train_gts["depth"].clone().detach()

        preds_depth_per_forward_pass = preds_depth.reshape(shape=(self.train_bs, self.img_height, self.img_width))
        preds_depth_all[k] = preds_depth_per_forward_pass

        # Get the probability of ground truth class of segmentation task for each pixel for each instance
        probs_seg_gts_class_per_forward_pass = torch.zeros(size=(self.train_bs, self.img_height, self.img_width), dtype=torch.float32,
                                                           requires_grad=False).to(self.device)
        for j in range(self.train_bs):  # iterate over instance in batch
            gts_seg_j = torch.unsqueeze(gts_seg[j], 0).long()  # expand dim to [1, 128, 256]
            # Pixel-wise probabilities for ground truth class of instance j in batch
            probs_seg_gts_class_j = torch.gather(input=probs_seg[j], dim=0, index=gts_seg_j)  # [128, 256]
            probs_seg_gts_class_j = torch.squeeze(probs_seg_gts_class_j, 0)  # [1, 128, 256]
            probs_seg_gts_class_per_forward_pass[j] = probs_seg_gts_class_j
        probs_seg_gts_class_all[k] = probs_seg_gts_class_per_forward_pass
        return preds_depth_all, probs_seg_gts_class_all, gts_seg, gts_depth

    def get_uncertainty_std_seg(self, x, y):
        """Expects a tensor containing all forward pass predictions of all instances in a batch
            of shape (Num_samples, batch_size, img_height, img_width) and the corresponding labels of
            shape (batch_size, img_height, img_width)"""
        # Boolean tensor whether pixel has gt class (=True) or not (=False)
        binary_mask = (y != 7)  # [batch_size, 128, 256]
        # get pixel-wise std for each instance in batch
        std_pixel_wise_per_instance = torch.std(input=x, dim=0, unbiased=False)  # [batch_size, 128, 256]
        # Extract all pixels with label resulting in 1D-tensor
        std_pixel_wise_per_instance_with_label = torch.masked_select(std_pixel_wise_per_instance, binary_mask)
        uncertainty = torch.mean(input=std_pixel_wise_per_instance_with_label, dim=0)
        return uncertainty

    def get_uncertainty_std_depth(self, x, y):
        """Expects a tensor containing all forward pass predictions of all instances in a batch
            of shape (Num_samples, batch_size, img_height, img_width) and the corresponding labels of
            shape (batch_size, i, mg_height, img_width)"""
        y = y.squeeze(1)  # [batch_size, 128, 256]
        # Boolean tensor whether pixel has gt class (=True) or not (=False)
        binary_mask = (y != 0.)  # [batch_size, 128, 256]
        # get pixel-wise std for each instance in batch
        std_pixel_wise_per_instance = torch.std(input=x, dim=0, unbiased=False)  # [batch_size, 128, 256]
        # Extract all pixels with label resulting in 1D-tensor
        std_pixel_wise_per_instance_with_label = torch.masked_select(std_pixel_wise_per_instance, binary_mask)
        uncertainty = torch.mean(input=std_pixel_wise_per_instance_with_label, dim=0)
        return uncertainty
