from torch.utils.tensorboard import SummaryWriter
from cs_utils import *
from create_dataset import CityScapes
from model import SegNet_MTAN, SegNet, DeepLabV3, wrap_with_dc_uncertainty, wrap_with_c_uncertainty, \
    wrap_with_d_uncertainty, wrap_with_uw_mlp_uncertainty
import argparse
import sys
import os
cwd = os.getcwd()
path = cwd + '/LibMTL'
sys.path.insert(1, path)
from trainer import Trainer
from utils import set_random_seed, set_device, get_exp_name
from config import prepare_args


def parse_args(parser):
    parser.add_argument('--aug', action=argparse.BooleanOptionalAction, default=False, help='data augmentation')
    parser.add_argument('--train_mode', default='trainval', type=str, help='trainval, train')
    parser.add_argument('--train_bs', default=8, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=8, type=int, help='batch size for test')
    parser.add_argument('--dataset_path', default='/fs/scratch/rng_cr_bcai_dl/OpenData/cityscapes_mtan/small', type=str,
                        help='dataset path')
    parser.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=False, help='save model')
    parser.add_argument('--pretrained_backbone', action=argparse.BooleanOptionalAction, default=False,
                        help='Pretrained Backbone')
    parser.add_argument('--epochs', default=200, type=int, help='Nb epochs of training')

    # general
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpu_id', default='0', type=str, help='gpu_id')
    parser.add_argument('--weighting', type=str, default='EW',
                         help='loss weighing strategies')
    parser.add_argument('--arch', type=str, default='SegNet',
                         help='architecture for MTL')
    parser.add_argument('--rep_grad', action='store_true', default=False,
                         help='computing gradient for representation or sharing parameters')
    ## optim
    parser.add_argument('--optim', type=str, default='adam',
                         help='optimizer for training, option: adam, sgd, adagrad, rmsprop')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for all types of optim')
    parser.add_argument('--momentum', type=float, default=0, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for all types of optim')
    parser.add_argument('--beta_1', type=float, default=0.9, help='beta_1 for Adam optimizer')
    parser.add_argument('--beta_2', type=float, default=0.999, help='beta_2 for Adam optimizer')
    parser.add_argument('--rms_alpha', type=float, default=0.99, help='beta_2 for Adam optimizer')

    ## scheduler
    parser.add_argument('--scheduler', type=str,  # default='step',
                         help='learning rate scheduler for training, option: step, cos, exp, cycle, 1cycle')
    parser.add_argument('--step_size', type=int, default=100, help='step size for StepLR')
    parser.add_argument('--mode', default='triangular', type=str,
                         help='Mode of CyclicLR: triangular, triangular2, exp_range')
    parser.add_argument('--exp_range_gamma', type=float, default=0.99995, help='Gamma in exp_range mode of CyclicLR')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma for StepLR')
    parser.add_argument('--max_lr', default=0.1, type=float, help='max lr for CyclicLR and OneCycleLR')
    parser.add_argument('--base_lr', default=0.00001, type=float, help='lowest lr for CyclicLR')
    parser.add_argument('--step_size_up', default=20, type=int, help='Nb iteratins to max_lr for CyclicLR')
    parser.add_argument('--step_size_down', default=20, type=int, help='Nb iterations to base_lr for CyclicLR')
    parser.add_argument('--steps_per_epoch', default=20, type=int, help='Nb steps per epoch for OneCycleLR')

    # args for weighting
    ## DWA
    parser.add_argument('--T', type=float, default=2.0, help='T for DWA')
    ## MGDA
    parser.add_argument('--mgda_gn', default='none', type=str,
                         help='type of gradient normalization for MGDA, option: l2, none, loss, loss+')
    ## GradVac
    parser.add_argument('--beta', type=float, default=0.5, help='beta for GradVac')
    ## GradNorm
    parser.add_argument('--alpha', type=float, default=1.5, help='alpha for GradNorm')
    ## GradDrop
    parser.add_argument('--leak', type=float, default=0.0, help='leak for GradDrop')
    ## CAGrad
    parser.add_argument('--calpha', type=float, default=0.4, help='calpha for CAGrad')
    parser.add_argument('--rescale', type=int, default=1, help='rescale for CAGrad')
    ## RLW
    parser.add_argument('--distribution', type=str, default='uniform', help='Distribution to sample weights for RLW')
    ## STL
    parser.add_argument('--task', type=int, default=1, help='Task to learn for STL')
    # args for architecture
    ## CGC
    parser.add_argument('--img_size', nargs='+', help='image size for CGC')
    parser.add_argument('--num_experts', nargs='+', help='the number of experts for sharing and task-specific')
    ## DSelect_k
    parser.add_argument('--num_nonzeros', type=int, default=2, help='num_nonzeros for DSelect-k')
    parser.add_argument('--kgamma', type=float, default=1.0, help='gamma for DSelect-k')
    # tensorboard_logs
    parser.add_argument('--logdir', type=str, default="./testruns/", help='log directory for tensorboard_logs')
    return parser.parse_args()


def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    # bookkeeping
    exp_name = get_exp_name(params, optim_param, scheduler_param, kwargs, dataset="Cityscapes")
    print("experiment name", exp_name)

    writer = SummaryWriter(params.logdir + exp_name)

    for arg, val in params.__dict__.items():
        writer.add_text(arg, str(val), 0)

    # prepare dataloaders
    cityscapes_train_set = CityScapes(root=params.dataset_path, train=True, augmentation=params.aug)
    cityscapes_test_set = CityScapes(root=params.dataset_path, train=False, augmentation=False)

    cityscapes_train_loader = torch.utils.data.DataLoader(
        dataset=cityscapes_train_set,
        batch_size=params.train_bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True)

    cityscapes_test_loader = torch.utils.data.DataLoader(
        dataset=cityscapes_test_set,
        batch_size=params.test_bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    # define tasks
    task_dict = {'segmentation': {'metrics': ['mIoU', 'pixAcc'],
                                  'metrics_fn': SegMetric(),
                                  'loss_fn': SegLoss(),
                                  'weight': [1, 1]},  # for mIoU and pixAcc --> the higher the better
                 'depth': {'metrics': ['abs_err', 'rel_err'],
                           'metrics_fn': DepthMetric(),
                           'loss_fn': DepthLoss(),
                           'weight': [0, 0]}  # for abs_err and rel_err --> the lower the better
                 }

    model_class = {'MTAN': SegNet_MTAN, 'SegNet': SegNet, 'RESNET101': DeepLabV3, 'RESNET50': DeepLabV3}[params.arch]
    if 'DCUW' in params.weighting:
        model_class = wrap_with_dc_uncertainty(model_class)
    elif 'CUW' in params.weighting:
        model_class = wrap_with_c_uncertainty(model_class)
    elif 'DUW' in params.weighting:
        model_class = wrap_with_d_uncertainty(model_class)
    elif 'UW_mlp' in params.weighting:
        model_class = wrap_with_uw_mlp_uncertainty(model_class)

    # initialize model
    if params.arch == 'MTAN':
        model = model_class(tasks=list(task_dict.keys()), weighting=params.weighting)
    elif params.arch == 'SegNet':
        model = model_class(tasks=list(task_dict.keys()), weighting=params.weighting)
    elif params.arch == 'RESNET101' or params.arch == 'RESNET50':
        model = model_class(tasks=list(task_dict.keys()), weighting=params.weighting, num_classes=[7, 1],
                                        pretrained=params.pretrained_backbone, resnet=params.arch)

    class CityScapesTrainer(Trainer):
        def __init__(self, task_dict, weighting, distribution, model, rep_grad, optim_param,
                     scheduler_param, writer, epochs, dataset, **kwargs):
            super(CityScapesTrainer, self).__init__(task_dict=task_dict,
                                                     weighting=weighting,
                                                     distribution=distribution,
                                                     model=model,
                                                     rep_grad=rep_grad,
                                                     optim_param=optim_param,
                                                     scheduler_param=scheduler_param,
                                                     writer=writer,
                                                     epochs=epochs,
                                                     dataset=dataset,
                                                     **kwargs)

    CityScapesModel = CityScapesTrainer(task_dict=task_dict,
                                          weighting=params.weighting,
                                          distribution=params.distribution,
                                          architecture=params.arch,
                                          model=model,
                                          rep_grad=params.rep_grad,
                                          optim_param=optim_param,
                                          scheduler_param=scheduler_param,
                                          writer=writer,
                                          epochs=params.epochs,
                                          dataset="Cityscapes",
                                          **kwargs)

    CityScapesModel.train(cityscapes_train_loader, cityscapes_test_loader)

    if bool(params.__dict__["save_model"]):
        torch.save(model.state_dict(), f"{writer.log_dir}/{exp_name}.pt")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configurations')
    params = parse_args(parser)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)
