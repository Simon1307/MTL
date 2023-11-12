import argparse
import torch


def prepare_args(params):
    r"""Return the configuration of hyperparameters, optimizer, and learning rate scheduler.

    Args:
        params (argparse.Namespace): The command-line arguments.
    """
    kwargs = {'weight_args': {}, 'arch_args': {}}
    if params.weighting in ['EW', 'UW', 'GLS', 'RLW', 'PCGrad', 'CAGrad', 'DWA', 'UW_CAGrad', 'DCUW', 'DCUW_CAGrad',
                            'STL', 'DCUW_PCGrad', 'UW_PCGrad', 'CUW', 'DUW', 'CUW_CAGrad', 'CUW_PCGrad', 'UW_mlp']:
        if params.weighting in ['DWA']:
            if params.T is not None:
                kwargs['weight_args']['T'] = params.T
            else:
                raise ValueError('DWA needs keywaord T')
        elif params.weighting in ['RLW']:
            kwargs['weight_args']['distribution'] = params.distribution
        elif params.weighting in ["STL"]:
            kwargs['weight_args']['task'] = params.task
        elif params.weighting in ['CAGrad'] or params.weighting in ["UW_CAGrad"] or params.weighting in ["DCUW_CAGrad"] \
                or params.weighting in ["CUW_CAGrad"]:
            if params.calpha is not None and params.rescale is not None:
                kwargs['weight_args']['calpha'] = params.calpha
                kwargs['weight_args']['rescale'] = params.rescale
            else:
                raise ValueError('CAGrad needs keyword calpha and rescale')
    else:
        raise ValueError('No support weighting method {}'.format(params.weighting))
        
    if params.optim in ['adam', 'sgd', 'adagrad', 'rmsprop']:
        if params.optim == 'adam':
            optim_param = {'optim': 'adam', 'lr': params.lr, 'weight_decay': params.weight_decay,
                           'betas': [params.beta_1, params.beta_2]}
        elif params.optim == 'sgd':
            optim_param = {'optim': 'sgd', 'lr': params.lr, 
                           'weight_decay': params.weight_decay, 'momentum': params.momentum}
        elif params.optim == 'rmsprop':
            optim_param = {'optim': 'rmsprop', 'lr': params.lr, 'weight_decay': params.weight_decay,
                           'momentum': params.momentum, 'alpha': params.rms_alpha}
    else:
        raise ValueError('No support optim method {}'.format(params.optim))
        
    if params.scheduler is not None:
        if params.scheduler in ['step', 'cyclic', '1cycle']:
            if params.scheduler == 'step':
                scheduler_param = {'scheduler': 'step', 'step_size': params.step_size, 'gamma': params.gamma}
            elif params.scheduler == 'cyclic':
                scheduler_param = {'scheduler': 'cyclic', 'mode': params.mode, 'step_size_up': params.step_size_up,
                                   'step_size_down': params.step_size_down, 'max_lr': params.max_lr,
                                   'base_lr': params.base_lr, 'gamma': params.exp_range_gamma}
            elif params.scheduler == '1cycle':
                scheduler_param = {'scheduler': '1cycle', 'max_lr': params.max_lr,
                                   "steps_per_epoch": params.steps_per_epoch, "epochs": params.epochs}
        else:
            raise ValueError('No support scheduler method {}'.format(params.scheduler))
    else:
        scheduler_param = None
    
    _display(params, kwargs, optim_param, scheduler_param)
    
    return kwargs, optim_param, scheduler_param


def _display(params, kwargs, optim_param, scheduler_param):
    print('='*40)
    print('General Configuration:')
    print('\tWeighting:', params.weighting)
    print('\tArchitecture:', params.arch)
    print('\tOptimizer:', params.optim)
    print('\tScheduler:', params.scheduler)
    print('\tRep_Grad:', params.rep_grad)
    print('\tSeed:', params.seed)
    print('\tDevice: {}'.format('cuda:'+params.gpu_id if torch.cuda.is_available() else 'cpu'))
    for wa, p in zip(['weight_args', 'arch_args'], [params.weighting, params.arch]):
        if kwargs[wa] != {}:
            print('{} Configuration:'.format(p))
            for k, v in kwargs[wa].items():
                print('\t'+k+':', v)
    print('Optimizer Configuration:')
    for k, v in optim_param.items():
        print('\t'+k+':', v)
    if scheduler_param is not None:
        print('Scheduler Configuration:')
        for k, v in scheduler_param.items():
            print('\t'+k+':', v)
