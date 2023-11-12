import sys
import torch, time
import numpy as np
from utils import count_improvement


class _PerformanceMeter_Cityscapes(object):
    def __init__(self, task_dict, writer, epochs, base_result=None):
        
        self.task_dict = task_dict
        self.task_num = len(self.task_dict)
        self.task_name = list(self.task_dict.keys())
        self.epochs = epochs

        self.weight = {task: self.task_dict[task]['weight'] for task in self.task_name}
        self.base_result = base_result
        self.best_result = {'improvement': -1e+2, 'epoch': 0, 'result': 0}
        
        self.losses = {task: self.task_dict[task]['loss_fn'] for task in self.task_name}
        self.metrics = {task: self.task_dict[task]['metrics_fn'] for task in self.task_name}
        
        self.results = {task:[] for task in self.task_name}
        self.loss_item = np.zeros(self.task_num)
        self.mean_test_metrics = {"mIoU": [],
                                  "pixAcc": [],
                                  "abs_err": [],
                                  "rel_err": []}
        self.has_val = False
        
        self._init_display()
        self.writer = writer

    def record_time(self, mode='begin'):
        if mode == 'begin':
            self.beg_time = time.time()
        elif mode == 'end':
            self.end_time = time.time()
        else:
            raise ValueError('No support time mode {}'.format(mode))
        
    def update(self, preds, gts, task_name=None):
        with torch.no_grad():
            if task_name is None:
                for tn, task in enumerate(self.task_name):
                    self.metrics[task].update_fun(preds[task], gts[task])
            else:
                self.metrics[task_name].update_fun(preds, gts)
        
    def get_score(self):
        with torch.no_grad():
            for tn, task in enumerate(self.task_name):
                self.results[task] = self.metrics[task].score_fun()
                self.loss_item[tn] = self.losses[task]._average_loss()
    
    def _init_display(self):
        print('='*40)
        print('LOG FORMAT | ', end='')
        for tn, task in enumerate(self.task_name):
            print(task+'_LOSS ', end='')
            for m in self.task_dict[task]['metrics']:
                print(m+' ', end='')
            print('| ', end='')
        print('TIME')
    
    def display(self, mode, epoch):
        if epoch == 0 and self.base_result is None and (mode=='val' if self.has_val else 'test'):
            self.base_result = self.results
        if mode == 'train':
            print('Epoch: {:04d} | '.format(epoch), end='')
        if not self.has_val and mode == 'test':
            self._update_best_result(self.results, epoch)
        if self.has_val and mode != 'train':
            self._update_best_result_by_val(self.results, epoch, mode)
        if mode == 'train':
            p_mode = 'TRAIN'
        elif mode == 'val':
            p_mode = 'VAL'
        else:
            p_mode = 'TEST'
        print('{}: '.format(p_mode), end='')
        for tn, task in enumerate(self.task_name):
            print('{:.4f} '.format(self.loss_item[tn]), end='')
            for i in range(len(self.results[task])):
                print('{:.4f} '.format(self.results[task][i]), end='')
            print('| ', end='')
        print('Time: {:.4f}'.format(self.end_time-self.beg_time), end='')
        print(' | ', end='') if mode!='test' else print()

        mIoU = self.results["segmentation"][0]
        pixAcc = self.results["segmentation"][1]
        abs_err = self.results["depth"][0]
        rel_err = self.results["depth"][1]

        # bookkeeping
        self.writer.add_scalar(p_mode.lower() + "/semantic_loss", self.loss_item[0], epoch)
        self.writer.add_scalar(p_mode.lower() + "/semantic_mIoU", mIoU, epoch)
        self.writer.add_scalar(p_mode.lower() + "/semantic_pixAcc", pixAcc, epoch)

        self.writer.add_scalar(p_mode.lower() + "/depth_loss", self.loss_item[1], epoch)
        self.writer.add_scalar(p_mode.lower() + "/depth_abs_err", abs_err, epoch)
        self.writer.add_scalar(p_mode.lower() + "/depth_rel_err", rel_err, epoch)
        # calculate delta m per epoch on test set
        if p_mode == 'TEST':
            delta_m_per_epoch = calc_delta_m_cityscapes(mIoU, pixAcc, abs_err, rel_err)
            self.writer.add_scalar("/delta_m_per_epoch", delta_m_per_epoch, epoch)
        # calculate delta m based on test metrics from last 10 epochs
        if epoch >= self.epochs-10 and p_mode == 'TEST':  # epoch >= 290
            self.mean_test_metrics["mIoU"].append(mIoU)
            self.mean_test_metrics["pixAcc"].append(pixAcc)
            self.mean_test_metrics["abs_err"].append(abs_err)
            self.mean_test_metrics["rel_err"].append(rel_err)
            if epoch == self.epochs-1:  # epoch == 299
                mean_test_mIoU = sum(self.mean_test_metrics["mIoU"]) / len(self.mean_test_metrics["mIoU"])
                mean_test_pixAcc = sum(self.mean_test_metrics["pixAcc"]) / len(self.mean_test_metrics["pixAcc"])
                mean_test_abs_err = sum(self.mean_test_metrics["abs_err"]) / len(self.mean_test_metrics["abs_err"])
                mean_test_rel_err = sum(self.mean_test_metrics["rel_err"]) / len(self.mean_test_metrics["rel_err"])
                print(f"Mean test metrics: mIoU={mean_test_mIoU}, pixAcc={mean_test_pixAcc}, abs_err={mean_test_abs_err}, rel_err={mean_test_rel_err}")
                delta_m = calc_delta_m_cityscapes(mean_test_mIoU, mean_test_pixAcc, mean_test_abs_err, mean_test_rel_err)
                print("Delta_M:", delta_m)
                self.writer.add_scalar("/delta_m", delta_m)
                self.writer.add_scalar("/mean_test_mIoU", mean_test_mIoU)
                self.writer.add_scalar("/mean_test_pixAcc", mean_test_pixAcc)
                self.writer.add_scalar("/mean_test_abs_err", mean_test_abs_err)
                self.writer.add_scalar("/mean_test_rel_err", mean_test_rel_err)

    def display_best_result(self):
        print('='*40)
        print('Best Result: Epoch {}, result {}'.format(self.best_result['epoch'], self.best_result['result']))
        print('='*40)
        
    def _update_best_result_by_val(self, new_result, epoch, mode):
        if mode == 'val':
            improvement = count_improvement(self.base_result, new_result, self.weight)
            if improvement > self.best_result['improvement']:
                self.best_result['improvement'] = improvement
                self.best_result['epoch'] = epoch
        else:
            if epoch == self.best_result['epoch']:
                self.best_result['result'] = new_result
        
    def _update_best_result(self, new_result, epoch):
        improvement = count_improvement(self.base_result, new_result, self.weight)
        if improvement > self.best_result['improvement']:
            self.best_result['improvement'] = improvement
            self.best_result['epoch'] = epoch
            self.best_result['result'] = new_result
        
    def reinit(self):
        for task in self.task_name:
            self.losses[task]._reinit()
            self.metrics[task].reinit()
        self.loss_item = np.zeros(self.task_num)
        self.results = {task:[] for task in self.task_name}


class _PerformanceMeter_NYUv2(object):
    def __init__(self, task_dict, writer, epochs, base_result=None):

        self.task_dict = task_dict
        self.task_num = len(self.task_dict)
        self.task_name = list(self.task_dict.keys())
        self.epochs = epochs

        self.weight = {task: self.task_dict[task]['weight'] for task in self.task_name}
        self.base_result = base_result
        self.best_result = {'improvement': -1e+2, 'epoch': 0, 'result': 0}

        self.losses = {task: self.task_dict[task]['loss_fn'] for task in self.task_name}
        self.metrics = {task: self.task_dict[task]['metrics_fn'] for task in self.task_name}

        self.results = {task: [] for task in self.task_name}
        self.loss_item = np.zeros(self.task_num)
        self.mean_test_metrics = {"mIoU": [],
                                  "pixAcc": [],
                                  "abs_err": [],
                                  "rel_err": [],
                                  "n_mean": [],
                                  "n_median": [],
                                  "n_11_25": [],
                                  "n_22_5": [],
                                  "n_30": []}
        self.has_val = False

        self._init_display()
        self.writer = writer

    def record_time(self, mode='begin'):
        if mode == 'begin':
            self.beg_time = time.time()
        elif mode == 'end':
            self.end_time = time.time()
        else:
            raise ValueError('No support time mode {}'.format(mode))

    def update(self, preds, gts, task_name=None):
        with torch.no_grad():
            if task_name is None:
                for tn, task in enumerate(self.task_name):
                    self.metrics[task].update_fun(preds[task], gts[task])
            else:
                self.metrics[task_name].update_fun(preds, gts)

    def get_score(self):
        with torch.no_grad():
            for tn, task in enumerate(self.task_name):
                self.results[task] = self.metrics[task].score_fun()
                self.loss_item[tn] = self.losses[task]._average_loss()

    def _init_display(self):
        print('=' * 40)
        print('LOG FORMAT | ', end='')
        for tn, task in enumerate(self.task_name):
            print(task + '_LOSS ', end='')
            for m in self.task_dict[task]['metrics']:
                print(m + ' ', end='')
            print('| ', end='')
        print('TIME')

    def display(self, mode, epoch):
        if epoch == 0 and self.base_result is None and (mode == 'val' if self.has_val else 'test'):
            self.base_result = self.results
        if mode == 'train':
            print('Epoch: {:04d} | '.format(epoch), end='')
        if not self.has_val and mode == 'test':
            self._update_best_result(self.results, epoch)
        if self.has_val and mode != 'train':
            self._update_best_result_by_val(self.results, epoch, mode)
        if mode == 'train':
            p_mode = 'TRAIN'
        elif mode == 'val':
            p_mode = 'VAL'
        else:
            p_mode = 'TEST'
        print('{}: '.format(p_mode), end='')
        for tn, task in enumerate(self.task_name):
            print('{:.4f} '.format(self.loss_item[tn]), end='')
            for i in range(len(self.results[task])):
                print('{:.4f} '.format(self.results[task][i]), end='')
            print('| ', end='')
        print('Time: {:.4f}'.format(self.end_time - self.beg_time), end='')
        print(' | ', end='') if mode != 'test' else print()

        mIoU = self.results["segmentation"][0]
        pixAcc = self.results["segmentation"][1]
        abs_err = self.results["depth"][0]
        rel_err = self.results["depth"][1]
        n_mean = self.results["normal"][0]
        n_median = self.results["normal"][1]
        n_11_25 = self.results["normal"][2]
        n_22_5 = self.results["normal"][3]
        n_30 = self.results["normal"][4]

        # bookkeeping
        self.writer.add_scalar(p_mode.lower() + "/semantic_loss", self.loss_item[0], epoch)
        self.writer.add_scalar(p_mode.lower() + "/semantic_mIoU", mIoU, epoch)
        self.writer.add_scalar(p_mode.lower() + "/semantic_pixAcc", pixAcc, epoch)

        self.writer.add_scalar(p_mode.lower() + "/depth_loss", self.loss_item[1], epoch)
        self.writer.add_scalar(p_mode.lower() + "/depth_abs_err", abs_err, epoch)
        self.writer.add_scalar(p_mode.lower() + "/depth_rel_err", rel_err, epoch)

        self.writer.add_scalar(p_mode.lower() + "/normal_loss", self.loss_item[2], epoch)
        self.writer.add_scalar(p_mode.lower() + "/normal_mean", n_mean, epoch)
        self.writer.add_scalar(p_mode.lower() + "/normal_median", n_median, epoch)
        self.writer.add_scalar(p_mode.lower() + "/normal_11_25", n_11_25, epoch)
        self.writer.add_scalar(p_mode.lower() + "/normal_22_5", n_22_5, epoch)
        self.writer.add_scalar(p_mode.lower() + "/normal_30", n_30, epoch)
        if p_mode == 'TEST':
            delta_m_per_epoch = calc_delta_m_nyuv2(mIoU, pixAcc, abs_err, rel_err, n_mean, n_median, n_11_25,
                                                        n_22_5,n_30)
            self.writer.add_scalar("/delta_m_per_epoch", delta_m_per_epoch, epoch)
        if epoch >= self.epochs - 10 and p_mode == 'TEST':  # epoch >= 190
            self.mean_test_metrics["mIoU"].append(mIoU)
            self.mean_test_metrics["pixAcc"].append(pixAcc)
            self.mean_test_metrics["abs_err"].append(abs_err)
            self.mean_test_metrics["rel_err"].append(rel_err)
            self.mean_test_metrics["rel_err"].append(rel_err)
            self.mean_test_metrics["n_mean"].append(n_mean)
            self.mean_test_metrics["n_median"].append(n_median)
            self.mean_test_metrics["n_11_25"].append(n_11_25)
            self.mean_test_metrics["n_22_5"].append(n_22_5)
            self.mean_test_metrics["n_30"].append(n_30)

            if epoch == self.epochs - 1:  # epoch == 199
                mean_test_mIoU = sum(self.mean_test_metrics["mIoU"]) / len(self.mean_test_metrics["mIoU"])
                mean_test_pixAcc = sum(self.mean_test_metrics["pixAcc"]) / len(self.mean_test_metrics["pixAcc"])
                mean_test_abs_err = sum(self.mean_test_metrics["abs_err"]) / len(self.mean_test_metrics["abs_err"])
                mean_test_rel_err = sum(self.mean_test_metrics["rel_err"]) / len(self.mean_test_metrics["rel_err"])
                mean_test_n_mean = sum(self.mean_test_metrics["n_mean"]) / len(self.mean_test_metrics["n_mean"])
                mean_test_n_median = sum(self.mean_test_metrics["n_median"]) / len(self.mean_test_metrics["n_median"])
                mean_test_n_11_25 = sum(self.mean_test_metrics["n_11_25"]) / len(self.mean_test_metrics["n_11_25"])
                mean_test_n_22_5 = sum(self.mean_test_metrics["n_22_5"]) / len(self.mean_test_metrics["n_22_5"])
                mean_test_n_30 = sum(self.mean_test_metrics["n_30"]) / len(self.mean_test_metrics["n_30"])

                print(
                    f"Mean test metrics: mIoU={mean_test_mIoU}, pixAcc={mean_test_pixAcc}, abs_err={mean_test_abs_err},"
                    f" rel_err={mean_test_rel_err}, n_mean={mean_test_n_mean}, n_median={mean_test_n_median},"
                    f" n_11_25={mean_test_n_11_25}, n_22_5={mean_test_n_22_5}, n_30={mean_test_n_30}")
                delta_m = calc_delta_m_nyuv2(mean_test_mIoU, mean_test_pixAcc, mean_test_abs_err, mean_test_rel_err,
                                             mean_test_n_mean, mean_test_n_median, mean_test_n_11_25, mean_test_n_22_5,
                                             mean_test_n_30)
                print("Delta_M:", delta_m)
                self.writer.add_scalar("/delta_m", delta_m)
                self.writer.add_scalar("/mean_test_mIoU", mean_test_mIoU)
                self.writer.add_scalar("/mean_test_pixAcc", mean_test_pixAcc)
                self.writer.add_scalar("/mean_test_abs_err", mean_test_abs_err)
                self.writer.add_scalar("/mean_test_rel_err", mean_test_rel_err)
                self.writer.add_scalar("/mean_test_n_mean", mean_test_n_mean)
                self.writer.add_scalar("/mean_test_n_median", mean_test_n_median)
                self.writer.add_scalar("/mean_test_n_11_25", mean_test_n_11_25)
                self.writer.add_scalar("/mean_test_n_22_5", mean_test_n_22_5)
                self.writer.add_scalar("/mean_test_n_30", mean_test_n_30)


    def display_best_result(self):
        print('=' * 40)
        print('Best Result: Epoch {}, result {}'.format(self.best_result['epoch'], self.best_result['result']))
        print('=' * 40)

    def _update_best_result_by_val(self, new_result, epoch, mode):
        if mode == 'val':
            improvement = count_improvement(self.base_result, new_result, self.weight)
            if improvement > self.best_result['improvement']:
                self.best_result['improvement'] = improvement
                self.best_result['epoch'] = epoch
        else:
            if epoch == self.best_result['epoch']:
                self.best_result['result'] = new_result

    def _update_best_result(self, new_result, epoch):
        improvement = count_improvement(self.base_result, new_result, self.weight)
        if improvement > self.best_result['improvement']:
            self.best_result['improvement'] = improvement
            self.best_result['epoch'] = epoch
            self.best_result['result'] = new_result

    def reinit(self):
        for task in self.task_name:
            self.losses[task]._reinit()
            self.metrics[task].reinit()
        self.loss_item = np.zeros(self.task_num)
        self.results = {task: [] for task in self.task_name}


class _PerformanceMeter_celebA(object):
    def __init__(self, task_dict, writer, epochs, base_result=None):

        self.task_dict = task_dict
        self.task_num = len(self.task_dict)
        self.task_name = list(self.task_dict.keys())
        self.epochs = epochs

        self.weight = {task: self.task_dict[task]['weight'] for task in self.task_name}
        self.base_result = base_result
        self.best_result = {'improvement': -1e+2, 'epoch': 0, 'result': 0}

        self.losses = {task: self.task_dict[task]['loss_fn'] for task in self.task_name}
        self.metrics = {task: self.task_dict[task]['metrics_fn'] for task in self.task_name}

        self.results = {task: [] for task in self.task_name}
        self.loss_item = np.zeros(self.task_num)

        self.mean_test_metrics = {}
        for name in self.task_name:
            self.mean_test_metrics[name] = []

        self.has_val = False

        self._init_display()
        self.writer = writer

    def record_time(self, mode='begin'):
        if mode == 'begin':
            self.beg_time = time.time()
        elif mode == 'end':
            self.end_time = time.time()
        else:
            raise ValueError('No support time mode {}'.format(mode))

    def update(self, preds, gts, task_name=None):
        with torch.no_grad():
            if task_name is None:
                for tn, task in enumerate(self.task_name):
                    self.metrics[task].update_fun(preds[task], gts[task])
            else:
                self.metrics[task_name].update_fun(preds, gts)

    def get_score(self):
        with torch.no_grad():
            for tn, task in enumerate(self.task_name):
                self.results[task] = self.metrics[task].score_fun()
                self.loss_item[tn] = self.losses[task]._average_loss()

    def _init_display(self):
        print('=' * 40)
        print('LOG FORMAT | ', end='')
        for tn, task in enumerate(self.task_name):
            print(task + '_LOSS ', end='')
            for m in self.task_dict[task]['metrics']:
                print(m + ' ', end='')
            print('| ', end='')
        print('TIME')

    def display(self, mode, epoch):
        if epoch == 0 and self.base_result is None and (mode == 'val' if self.has_val else 'test'):
            self.base_result = self.results
        if mode == 'train':
            print('Epoch: {:04d} | '.format(epoch), end='')
        if not self.has_val and mode == 'test':
            self._update_best_result(self.results, epoch)
        if self.has_val and mode != 'train':
            self._update_best_result_by_val(self.results, epoch, mode)
        if mode == 'train':
            p_mode = 'TRAIN'
        elif mode == 'val':
            p_mode = 'VAL'
        else:
            p_mode = 'TEST'
        print('{}: '.format(p_mode), end='')
        for tn, task in enumerate(self.task_name):
            print('{:.4f} '.format(self.loss_item[tn]), end='')
            for i in range(len(self.results[task])):
                print('{:.4f} '.format(self.results[task][i]), end='')
            print('| ', end='')
        print('Time: {:.4f}'.format(self.end_time - self.beg_time), end='')
        print(' | ', end='') if mode != 'test' else print()

        average_accuracy_error = 0.0
        accuracy_errors = []
        for i, attribute in enumerate(self.results):
            acc_err = self.results[attribute][0]
            accuracy_errors.append(acc_err)
            loss = self.loss_item[i]
            average_accuracy_error += acc_err
            path = p_mode.lower() + "/" + attribute
            self.writer.add_scalar(path + "_acc_err", acc_err, epoch)
            self.writer.add_scalar(path + "_bce_loss", loss, epoch)
            if epoch >= self.epochs - 10 and p_mode == 'TEST':  # last 10 epochs
                self.mean_test_metrics[attribute].append(acc_err)
        path = p_mode.lower() + "/average_acc_err"
        average_accuracy_error /= self.task_num
        self.writer.add_scalar(path, average_accuracy_error, epoch)
        if p_mode == 'TEST':
            delta_m_per_epoch = calc_delta_m_celebA(accuracy_errors)
            self.writer.add_scalar("/delta_m_per_epoch", delta_m_per_epoch, epoch)
        if epoch == self.epochs - 1 and p_mode == 'TEST':  # last epoch metrics on test set
            mean_accuracy_errors = []
            for name in self.task_name:
                mean_acc_err = sum(self.mean_test_metrics[name]) / len(self.mean_test_metrics[name])
                path = "/mean_test_" + name + "_acc_err"
                self.writer.add_scalar(path, mean_acc_err)
                mean_accuracy_errors.append(mean_acc_err)
            mean_test_metrics = " ".join(str(e) for e in mean_accuracy_errors)
            print('=' * 40)
            print(
                f"Mean test metrics: {mean_test_metrics}")
            print('=' * 40)
            print(f"Average accuracy error in last epoch on test set: {average_accuracy_error}")
            print('=' * 40)
            delta_m = calc_delta_m_celebA(mean_accuracy_errors)
            print("Delta_M:", delta_m)
            self.writer.add_scalar("/delta_m", delta_m)

    def display_best_result(self):
        print('=' * 40)
        print('Best Result: Epoch {}, result {}'.format(self.best_result['epoch'], self.best_result['result']))
        print('=' * 40)

    def _update_best_result_by_val(self, new_result, epoch, mode):
        if mode == 'val':
            improvement = count_improvement(self.base_result, new_result, self.weight)
            if improvement > self.best_result['improvement']:
                self.best_result['improvement'] = improvement
                self.best_result['epoch'] = epoch
        else:
            if epoch == self.best_result['epoch']:
                self.best_result['result'] = new_result

    def _update_best_result(self, new_result, epoch):
        improvement = count_improvement(self.base_result, new_result, self.weight)
        if improvement > self.best_result['improvement']:
            self.best_result['improvement'] = improvement
            self.best_result['epoch'] = epoch
            self.best_result['result'] = new_result

    def reinit(self):
        for task in self.task_name:
            self.losses[task]._reinit()
            self.metrics[task].reinit()
        self.loss_item = np.zeros(self.task_num)
        self.results = {task: [] for task in self.task_name}


def calc_delta_m_cityscapes(mean_iu: float, pix_acc: float, abs_err: float, rel_err: float):
    """Calculates delta-M according to our single-task baseline (SegNet, 300 epochs). Results are hardcoded below.
    Args:
        mean_iu (float): Mean intersection over union, must be in [0, 1]
        pix_acc (float): Accuracy of segmentation per pixel, must be in [0, 1]
        abs_err (float): Mean L1 error of depth estimates
        rel_err (float): L1 error relative to true depth, then mean
    Returns:
        [type]: [description]
    """
    base_mean_iu = 0.7403
    base_pix_acc = 0.9314

    base_abs_err = 0.0124
    base_rel_err = 25.5362

    delta_m = (
        1
        / 4
        * (
            (-1) * (mean_iu - base_mean_iu) / base_mean_iu
            + (-1) * (pix_acc - base_pix_acc) / base_pix_acc
            + 1 * (abs_err - base_abs_err) / base_abs_err
            + 1 * (rel_err - base_rel_err) / base_rel_err
        )
    )
    return delta_m * 100


def calc_delta_m_nyuv2(mean_iu: float, pix_acc: float, abs_err: float, rel_err: float, n_mean: float, n_median: float,
                       n_11_25: float, n_22_5: float, n_30: float):
    """Calculates delta-M according to our single-task baseline (SegNet, 200 epochs). Results are hardcoded below.
    Args:
        mean_iu (float): Mean intersection over union, must be in [0, 1]
        pix_acc (float): Accuracy of segmentation per pixel, must be in [0, 1]
        abs_err (float): Mean L1 error of depth estimates
        rel_err (float): L1 error relative to true depth, then mean
        n_mean (float)
        n_median (float)
        n_11_25 (float)
        n_22_5 (float)
        n_30 (float)
    Returns:
        [type]: [description]
    """
    # seg
    base_mean_iu = 0.3614
    base_pix_acc = 0.6223
    # depth
    base_abs_err = 0.6935
    base_rel_err = 0.2730
    # normal
    base_n_mean = 25.5863
    base_n_median = 19.5904
    base_n_11_25 = 0.2884
    base_n_22_5 = 0.5583
    base_n_30 = 0.6801

    delta_m = (
        1
        / 9
        * (
            (-1) * (mean_iu - base_mean_iu) / base_mean_iu
            + (-1) * (pix_acc - base_pix_acc) / base_pix_acc
            + 1 * (abs_err - base_abs_err) / base_abs_err
            + 1 * (rel_err - base_rel_err) / base_rel_err
            + 1 * (n_mean - base_n_mean) / base_n_mean
            + 1 * (n_median - base_n_median) / base_n_median
            + (-1) * (n_11_25 - base_n_11_25) / base_n_11_25
            + (-1) * (n_22_5 - base_n_22_5) / base_n_22_5
            + (-1) * (n_30 - base_n_30) / base_n_30
        )
    )
    return delta_m * 100


def calc_delta_m_celebA(mtl_results):
    """Calculates delta-M according to our single-task baseline (ResNet-18, 100 epochs). Results are hardcoded below.
    """
    baseline_results = [6.71,18.74,20.43,17.77,1.22,4.65,31.11,17.77,11.71,5.0,4.56,13.57,8.61,4.99,4.14,0.42,3.03,2.01,
                        10.26,14.54,1.9,7.07,3.6,13.94,4.99,28.91,3.34,26.18,7.26,5.63,2.67,8.31,18.99,18.29,11.61,1.01,
                        7.37,15.22,3.39,13.54]

    delta_m = 0.0
    for i in range(len(mtl_results)):
        delta_m += (mtl_results[i] - baseline_results[i]) / baseline_results[i]
    delta_m /= 40
    return delta_m * 100
