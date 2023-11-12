import random, torch, os
import numpy as np
from datetime import datetime


def set_random_seed(seed):
    r"""Set the random seed for reproducibility.

    Args:
        seed (int, default=0): The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def set_device(gpu_id):
    r"""Set the device where model and data will be allocated. 

    Args:
        gpu_id (str, default='0'): The id of gpu.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

def count_parameters(model):
    r'''Calculates the number of parameters for a model.

    Args:
        model (torch.nn.Module): A neural network module.
    '''
    trainable_params = 0
    non_trainable_params = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        else:
            non_trainable_params += p.numel()
    print('='*40)
    print('Total Params:', trainable_params + non_trainable_params)
    print('Trainable Params:', trainable_params)
    print('Non-trainable Params:', non_trainable_params)
        
def count_improvement(base_result, new_result, weight):
    r"""Calculate the improvement between two results,

    .. math::
        \Delta_{\mathrm{p}}=100\%\times \frac{1}{T}\sum_{t=1}^T 
        \frac{1}{M_t}\sum_{m=1}^{M_t}\frac{(-1)^{w_{t,m}}(B_{t,m}-N_{t,m})}{N_{t,m}}.

    Args:
        base_result (dict): A dictionary of scores of all metrics of all tasks.
        new_result (dict): The same structure with ``base_result``.
        weight (dict): The same structure with ``base_result`` while each elements is binary integer representing whether higher or lower score is better.

    Returns:
        float: The improvement between ``new_result`` and ``base_result``.

    Examples::

        base_result = {'A': [96, 98], 'B': [0.2]}
        new_result = {'A': [93, 99], 'B': [0.5]}
        weight = {'A': [1, 0], 'B': [1]}

        print(count_improvement(base_result, new_result, weight))
    """
    improvement = 0
    count = 0
    for task in list(base_result.keys()):
        improvement += (((-1)**np.array(weight[task]))*\
                        (np.array(base_result[task])-np.array(new_result[task]))/\
                         np.array(base_result[task])).mean()
        count += 1
    return improvement/count


def get_exp_name(params, optimizer_params, scheduler_params, kwargs, dataset):
    params = params.__dict__
    weighting = params["weighting"]
    arch = params["arch"]
    seed = params["seed"]
    epochs = params["epochs"]

    if params["pretrained_backbone"]:
        exp_name = dataset + "--pretrained_backbone--" + weighting + "--"
    else:
        exp_name = dataset + "--" + weighting + "--"

    if kwargs is not None:
        for key, val in kwargs["weight_args"].items():
            exp_name += key + "_" + str(val) + "--"
        exp_name += arch + "--" + "Seed" + "_" + str(seed) + '--' + "Epochs" + "_" + str(epochs) + "--"

    for key, val in  optimizer_params.items():
        if key == "lr":
            if scheduler_params is not None:
                if scheduler_params["scheduler"] == "cyclic" or scheduler_params["scheduler"] == "1cycle":
                    continue
        if key == "betas":
            exp_name += 'beta1' + '_' + str(val[0]) + '--' + 'beta2' + '_' + str(val[1]) + '--'
            continue
        if key == 'alpha':
            exp_name += 'rms_alpha' + '_' + str(val) + '--'
            continue
        if isinstance(key, float) or isinstance(key, int):
            exp_name += str(key) + '_'
        else:
            exp_name += key + '_'
        if isinstance(val, float) or isinstance(val, int):
            exp_name += str(val) + '--'
        else:
            exp_name += val + '--'

    if scheduler_params is not None:
        for key, val in scheduler_params.items():
            if isinstance(key, str) and key == "epochs":
                continue
            if isinstance(key, float) or isinstance(key, int):
                exp_name += str(key) + '_'
            else:
                exp_name += key + '_'
            if isinstance(val, float) or isinstance(val, int):
                exp_name += str(val) + '--'
            else:
                exp_name += val + '--'

    return exp_name + f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
