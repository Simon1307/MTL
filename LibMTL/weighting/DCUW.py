import numpy as np
import torch
import io
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import PIL.Image
from torchvision.transforms import ToTensor
import sys
import os
cwd = os.getcwd()
path = cwd + '/LibMTL/weighting'
sys.path.insert(1, path)
from abstract_weighting import AbsWeighting


class DCUW(AbsWeighting):
    def __init__(self, trainer, **kwargs):
        AbsWeighting.__init__(self, trainer, kwargs["train_batch"], kwargs["writer"])
        self.sigma_means_over_iterations = np.zeros(shape=(self.epochs * self.train_batch, self.task_num))
        self.sigma_stds_over_iterations = np.zeros(shape=(self.epochs * self.train_batch, self.task_num))

    def update_step(self, losses, **kwargs):
        loss_scale = kwargs["loss_scale"]
        epoch = kwargs["epoch"]
        iter_ctr = kwargs["iter_ctr"]
        model = kwargs["model"]
        num_tasks = len(losses)
        if iter_ctr == 1:  # Plot initial distribution of sigmas
            self.plot_kde_sigmas(current_epoch=-1, model=model, num_tasks=num_tasks)

        loss = (losses / (2 * loss_scale.exp()) + loss_scale / 2).sum()
        loss.backward()
        self.plot_sigma_density(current_epoch=epoch, current_iteration=iter_ctr,
                                 num_iterations_per_epoch=self.train_batch, model=model, num_tasks=num_tasks)

        if num_tasks == 2:
            if ((epoch + 1) % 1 == 0) and (iter_ctr % self.train_batch == 0):  # every epoch for Cityscapes
                 self.plot_kde_sigmas(current_epoch=epoch, model=model, num_tasks=num_tasks)
        elif num_tasks == 3:
            if ((epoch + 1) % 1 == 0) and (iter_ctr % self.train_batch == 0):  # every epoch for NYU
                 self.plot_kde_sigmas(current_epoch=epoch, model=model, num_tasks=num_tasks)
        return loss_scale.detach()

    def plot_sigma_density(self, current_epoch, current_iteration, num_iterations_per_epoch, model, num_tasks):
        current_epoch += 1
        df = self.sample_sigmas(num_samples=100, model=model)  # Sample 100 sigmas from current learned distribution
        # For every iteration write mean and standard deviation for each sigma based on 100 sigmas
        # drawn from learned distribution at time step t to tensorboard
        sigma_means = df.groupby('$\sigma$', as_index=False)['$\sigma_t$'].mean()
        sigma_stds = df.groupby('$\sigma$', as_index=False)['$\sigma_t$'].std()
        self.sigma_means_over_iterations[current_iteration-1] = sigma_means['$\sigma_t$']
        self.sigma_stds_over_iterations[current_iteration-1] = sigma_stds['$\sigma_t$']

        # Having the mean and std for each sigma per iteration, plot mean and +- 1std over iterations
        if current_iteration == self.epochs * num_iterations_per_epoch:  # plot sigma density in last iteration
            small_sigma_means_over_iterations = self.sigma_means_over_iterations[0::100]  # every 100th datapoint
            small_sigma_stds_over_iterations = self.sigma_means_over_iterations[0::100]
            t = np.arange(small_sigma_means_over_iterations.shape[0])
            fig, ax = plt.subplots(1, figsize=(6,6))
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            # Every row stores the mean sigma value of 1 iteration for all sigmas
            for i in range(self.task_num):
                mu = small_sigma_means_over_iterations[:, i]  # mean of sigma{i} over the iterations
                std = small_sigma_stds_over_iterations[:, i]  # std of sigma{i} over the iterations
                plt.plot(t, mu, lw=2, label=f'$\sigma_{i + 1}$', alpha=0.5)
                plt.fill_between(t, mu + std, mu - std, alpha=0.5)
            plt.legend(loc='upper right', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            if num_tasks == 2:
                plt.xticks(np.linspace(min(t), max(t) + 1, 7))
                locs, labels = plt.xticks()
                plt.xticks(locs, labels=[0, 50, 100, 150, 200, 250, 300])
            elif num_tasks == 3:
                plt.xticks(np.linspace(min(t), max(t) + 1, 5))
                locs, labels = plt.xticks()
                plt.xticks(locs, labels=[0, 50, 100, 150, 200])
            plt.xlabel('Epoch', size=20)
            plt.ylabel('$\sigma_t$', size=25)
            if num_tasks == 2:
                path_ = f'/home/kus1rng/MTL-playground/dcuw_plots/cs/sigma_density_over_epochs.pdf'
            elif num_tasks == 3:
                path_ = f'/home/kus1rng/MTL-playground/dcuw_plots/nyu/sigma_density_over_epochs.pdf'
            plt.savefig(path_, format='pdf', bbox_inches='tight')
            plt.close(fig)


    def plot_kde_sigmas(self, current_epoch, model, num_tasks):
        df = self.sample_sigmas(num_samples=1000, model=model)
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        g = sns.histplot(data=df, x="$\sigma_t$", hue="$\sigma$", kde=True, ax=ax, alpha=0.5)
        if current_epoch == -1:
            g.legend_.set_title(None)
            plt.setp(ax.get_legend().get_texts(), fontsize='20')
        else:
            plt.legend([], [], frameon=False)

        if num_tasks == 2:
            path_ = f'/home/kus1rng/MTL-playground/dcuw_plots/cs/sigma_dist_afer_{current_epoch+1}_epochs.pdf'
        elif num_tasks == 3:
            path_ = f'/home/kus1rng/MTL-playground/dcuw_plots/nyu/sigma_dist_afer_{current_epoch+1}_epochs.pdf'
        # plt.legend(loc='upper right', fontsize=20)
        plt.xlabel('$\sigma_t$', size=25)
        plt.ylabel('Count', size=20)
        plt.xticks(fontsize=20)
        for ind, label in enumerate(g.get_xticklabels()):
            if ind % 2 == 0:  # every 2nd label is kept
                label.set_visible(True)
            else:
                label.set_visible(False)

        plt.yticks(fontsize=20)
        plt.savefig(path_, format='pdf', bbox_inches='tight')
        plt.close(fig)


    def sample_sigmas(self, num_samples, model):
        with torch.no_grad():
            df = pd.DataFrame(columns=["$\sigma_t$", "$\sigma$"])
            dfs = [df]
            for i in range(num_samples):
                loss_scale, _ = model.compute_uncertainty()
                loss_scale = loss_scale.detach().cpu().numpy()
                for j in range(self.task_num):
                    dfs.append(pd.DataFrame([[loss_scale[j], f'$\sigma_{j+1}$']], columns=["$\sigma_t$", "$\sigma$"]))
                df = pd.concat(dfs)
            df = df.set_index(np.arange(len(df)))
        return df
