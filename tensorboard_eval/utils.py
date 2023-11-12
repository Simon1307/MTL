import sys

from tensorboard.backend.event_processing import event_accumulator


def calc_delta_m_cityscapes(mean_iu: float, pix_acc: float, abs_err: float, rel_err: float):
    """Calculates delta-M according to our single-task baseline.
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


def calc_epoch_specific_delta_m_cityscapes(experiments, experiment, epoch):
    """Given a specific epoch and experiment calculate Delta M"""
    mIoU = experiments[experiment]["test/semantic_mIoU"][epoch]
    pixAcc = experiments[experiment]["test/semantic_pixAcc"][epoch]
    abs_err = experiments[experiment]["test/depth_abs_err"][epoch]
    rel_err = experiments[experiment]["test/depth_rel_err"][epoch]
    delta_m = calc_delta_m_cityscapes(mean_iu=mIoU, pix_acc=pixAcc, abs_err=abs_err, rel_err=rel_err)
    return delta_m, mIoU, pixAcc, abs_err, rel_err


def calc_delta_m_celebA(mtl_results):
    """Calculates delta-M according to our single-task baseline."""
    baseline_results = [6.71,18.74,20.43,17.77,1.22,4.65,31.11,17.77,11.71,5.0,4.56,13.57,8.61,4.99,4.14,0.42,3.03,2.01,
                        10.26,14.54,1.9,7.07,3.6,13.94,4.99,28.91,3.34,26.18,7.26,5.63,2.67,8.31,18.99,18.29,11.61,1.01,
                        7.37,15.22,3.39,13.54]

    delta_m = 0.0
    for i in range(len(mtl_results)):
        delta_m += (mtl_results[i] - baseline_results[i]) / baseline_results[i]
    delta_m /= 40
    return delta_m * 100


def calc_epoch_specific_delta_m_celebA(experiments, experiment, epoch, attributes):
    """Given a specific epoch and experiment calculate Delta M"""
    accuracy_errors = []
    for attribute in attributes:
        identifier = "test/" + attribute + "_acc_err"
        acc_err = experiments[experiment][identifier][epoch]
        accuracy_errors.append(acc_err)
    delta_m = calc_delta_m_celebA(mtl_results=accuracy_errors)
    return delta_m, accuracy_errors


def calc_delta_m_nyuv2(mean_iu: float, pix_acc: float, abs_err: float, rel_err: float, n_mean: float, n_median: float, n_11_25: float, n_22_5: float, n_30: float):
    """Calculates delta-M according to our single-task baseline.
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


def calc_epoch_specific_delta_m_nyu(experiments, experiment, epoch):
    """Given a specific epoch and experiment calculate Delta M"""
    mIoU = experiments[experiment]["test/semantic_mIoU"][epoch]
    pixAcc = experiments[experiment]["test/semantic_pixAcc"][epoch]
    abs_err = experiments[experiment]["test/depth_abs_err"][epoch]
    rel_err = experiments[experiment]["test/depth_rel_err"][epoch]
    n_mean = experiments[experiment]["test/normal_mean"][epoch]
    n_median = experiments[experiment]["test/normal_median"][epoch]
    n_11_25 = experiments[experiment]["test/normal_11_25"][epoch]
    n_22_5 = experiments[experiment]["test/normal_22_5"][epoch]
    n_30 = experiments[experiment]["test/normal_30"][epoch]
    delta_m = calc_delta_m_nyuv2(mean_iu=mIoU, pix_acc=pixAcc, abs_err=abs_err, rel_err=rel_err, n_mean=n_mean, n_median=n_median, n_11_25=n_11_25, n_22_5=n_22_5, n_30=n_30)
    return delta_m, mIoU, pixAcc, abs_err, rel_err, n_mean, n_median, n_11_25, n_22_5, n_30


def replace_exp_name(exp, num_runs_per_method):
    if '--Seed_30' in exp:
        exp = exp.replace('--Seed_30', '--NbSeeds_' + str(num_runs_per_method))
    elif '--Seed_50' in exp:
        exp = exp.replace('--Seed_50', '--NbSeeds_' + str(num_runs_per_method))
    elif '--Seed_5' in exp:
        exp = exp.replace('--Seed_5', '--NbSeeds_' + str(num_runs_per_method))
    return exp


def get_experiment_groups(exp_names):
    """Store each experiment group in a list. An exp group consists of 5 runs with 5 different seeds. Only one
     character of the seed argument must differ across runs belonging to the same experiment group for this
     function to work"""
    experiment_groups = []
    for i, exp_name in enumerate(exp_names):
        exp_name = exp_names[i]
        # collect all experiment indices already assigned to any group
        assigned_indices = []
        if experiment_groups is not None:
            for group in experiment_groups:
                for index in group:
                    assigned_indices.append(index)
            if i in assigned_indices:
                continue

        experiment_group = []
        for j, exp_name_ in enumerate(exp_names):
            if compare(exp_name, exp_name_):
                # check if differing character is seed value
                name_split_exp_name = exp_name.split(sep='--')
                name_split_exp_name_ = exp_name_.split(sep='--')
                for e in range(len(name_split_exp_name)):
                    if name_split_exp_name[e][:4] == "Seed":
                        if name_split_exp_name[e] != name_split_exp_name_[e]:
                            if not experiment_group:
                                experiment_group.append(i)
                            experiment_group.append(j)
        experiment_groups.append(experiment_group)
    return experiment_groups


def compare(string1, string2):
    if len(string1) == len(string2):
        count_diffs = 0
        for a, b in zip(string1, string2):
            if a != b:
                if count_diffs:
                    return False
                count_diffs += 1
        return True


def read_tb(tb_path):
    ea = event_accumulator.EventAccumulator(tb_path,
                                            size_guidance={
                                                event_accumulator.COMPRESSED_HISTOGRAMS: 1,
                                                event_accumulator.IMAGES: 1,
                                                event_accumulator.AUDIO: 1,
                                                event_accumulator.SCALARS: 0, # 0 = all events
                                                event_accumulator.HISTOGRAMS: 1})
    ea.Reload()
    return ea


def get_all_scalars(ea):
    """For one experiment collect all scalar values (all epoch values) in a dict.
    Key value pair example:
    'train/semantic_mIoU': [list of all epoch values]
    """
    scalar_names = []
    for scalar_name in ea.Tags()["scalars"]:
        if 'weight' in scalar_name or 'train' in scalar_name:
            continue
        # filter scalar_names: do no collect train scalar values
        scalar_names.append(scalar_name)
    scalar_values = dict.fromkeys(scalar_names)
    for key in scalar_values.keys():
        scalar_event_list = ea.Scalars(key)
        values = []
        for event in scalar_event_list:
            values.append(event.value)
        scalar_values[key] = values
    return scalar_values


def get_column_names(dataset_name, num_tasks):
    if dataset_name == 'Cityscapes':
        columns = ['exp name',
                   'Train Time',
                   'Sem. mIoU mean',
                   'Sem. pixAcc mean',
                   'Depth Abs Err mean',
                   'Depth Rel Err mean',
                   'Delta M mean',
                   'Delta M std']
        attributes = None
    elif dataset_name == 'NYU':
        columns = ['exp name', 'Train Time', 'Sem. mIoU mean', 'Sem. pixAcc mean', 'Depth Abs Err mean',
                   'Depth Rel Err mean', 'Normal Mean mean', 'Normal Median mean', 'Normal 11_25 mean',
                   'Normal 22_5 mean', 'Normal 30 mean', 'Delta M mean', 'Delta M std']
        attributes = None
    elif dataset_name == 'CelebA':
        columns = ['exp name',
                   'Train Time']
        for i in range(num_tasks):
            columns.append(f'Att{i+1}')
        columns.append('Delta M mean')
        columns.append('Delta M std')
        columns.append('Avg Acc Err')
        attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
                      'Big_Lips',
                      'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
                      'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                      'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                      'Pointy_Nose',
                      'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                      'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie',
                      'Young']

    return columns, attributes


def rename_methods(df):
    for i in range(len(df)):
        exp_name = df["exp name"][i].split("--")
        weighting_method = exp_name[1]
        if weighting_method == "DWA" or weighting_method == "RLW":
            weighting_arg = exp_name[2]
            arch = exp_name[3]
            df["exp name"][i] = weighting_method + " + " + arch
        elif weighting_method == "CAGrad" or weighting_method == "UW_CAGrad" or weighting_method == "DCUW_CAGrad":
            weighting_arg = exp_name[2]
            arch = exp_name[4]
            if "_" in weighting_method:
                weighting_method_ = weighting_method.split('_')
                weighting_method = weighting_method_[0] + " + " + weighting_method_[1]
            df["exp name"][i] = weighting_method + " + " + arch
        elif weighting_method == "pretrained_backbone":
            weighting_method = exp_name[2]
            if weighting_method == "DWA" or weighting_method == "RLW":
                weighting_arg = exp_name[3]
                arch = exp_name[4]
                df["exp name"][i] = weighting_method + " + " + arch
            elif weighting_method == "CAGrad" or weighting_method == "UW_CAGrad" or weighting_method == "DCUW_CAGrad":
                weighting_arg = exp_name[3]
                arch = exp_name[5]
                if "_" in weighting_method:
                    weighting_method_ = weighting_method.split('_')
                    weighting_method = weighting_method_[0] + " + " + weighting_method_[1]
                df["exp name"][i] = weighting_method + " + " + arch
            else:
                arch = exp_name[3]
                df["exp name"][i] = weighting_method + " + " + arch
        else:
            arch = exp_name[2]
            df["exp name"][i] = weighting_method + " + " + arch
        return df


def prepare_df(df, dataset_name):
    df['Delta M mean'] = df['Delta M mean'].astype('string')
    df['Delta M std'] = df['Delta M std'].astype('string')
    for i in range(len(df)):
        df['Delta M mean'][i] = '$' + df['Delta M mean'][i] + ' \pm ' + df['Delta M std'][i] + '$'
    df = df.drop(columns=['Train Time', 'Delta M std'])
    if dataset_name == 'Cityscapes':
        df = df.rename(columns={'exp name': "Method",
                                'Sem. mIoU mean': "mIoU",
                                'Sem. pixAcc mean': 'Pix Acc',
                                'Depth Abs Err mean': 'Abs Err',
                                'Depth Rel Err mean': 'Rel Err',
                                'Delta M mean': '$\Delta_m\downarrow$'})
    elif dataset_name == 'NYU':
        df = df.rename(columns={'exp name': "Method",
                                'Sem. mIoU mean': "mIoU",
                                'Sem. pixAcc mean': 'Pix Acc',
                                'Depth Abs Err mean': 'Abs Err',
                                'Depth Rel Err mean': 'Rel Err',
                                'Normal Mean mean': 'Mean',
                                'Normal Median mean': 'Median',
                                'Normal 11_25 mean': 11.25,
                                'Normal 22_5 mean': 22.5,
                                'Normal 30 mean': 30,
                                'Delta M mean': '$\Delta_m\downarrow$'})
    elif dataset_name == 'CelebA':
        df = df.rename(columns={'exp name': "Method", 'Delta M mean': '$\Delta_m\downarrow$'})

    for col in df.columns:
        if col == "Method" or col == '$\Delta_m\downarrow$':
            continue
        df[col] = df[col].astype('string')
        for i in range(len(df)):
            df[col][i] = "$" + df[col][i] + "$"

    if dataset_name == 'Cityscapes':
        header = [['Method',
                   'Segmentation',
                   'Segmentation',
                   'Depth',
                   'Depth',
                   '$\Delta_m \downarrow$',
                   ],
                  [
                      '',
                      '(Higher Better)',
                      '(Higher Better)',
                      '(Lower Better)',
                      '(Lower Better)',
                      '',
                  ],
                  [
                      '',
                      'mIoU',
                      'Pix Acc',
                      'Abs Err',
                      'Rel Err',
                      '',
                  ]
                  ]
        df.columns = header
    elif dataset_name == 'NYU':
        header = [['Method',
                   'Segmentation',
                   'Segmentation',
                   'Depth',
                   'Depth',
                   'Normal',
                   'Normal',
                   'Normal',
                   'Normal',
                   'Normal',
                   '$\Delta_m \downarrow$',
                   ],
                  [
                      '',
                      '(Higher Better)',
                      '(Higher Better)',
                      '(Lower Better)',
                      '(Lower Better)',
                      '(Lower Better)',
                      '(Lower Better)',
                      '(Higher Better)',
                      '(Higher Better)',
                      '(Higher Better)',
                      '',
                  ],
                  [
                      '',
                      'mIoU',
                      'Pix Acc',
                      'Abs Err',
                      'Rel Err',
                      'Mean',
                      'Median',
                      11.25,
                      22.5,
                      30,
                      '',
                  ]
                  ]
        df.columns = header
    elif dataset_name == 'CelebA':
        df = df.transpose()
        columns = df.iloc[0]
        df.columns = columns
        df = df.iloc[1:]
        df = df[['DWA + RESNET18',
                 'EW + RESNET18',
                 'GLS + RESNET18',
                 'RLW + RESNET18',
                 'UW + RESNET18',
                 'CUW + RESNET18',
                 'DCUW + RESNET18']]
        df = df.reset_index()
    return df


def get_table_meta_data(dataset_name):
    if dataset_name == "Cityscapes":
        caption = "Methods on Cityscapes"
        label = "tab:cs_methods",
        column_format = "ccccccc"
    elif dataset_name == "NYU":
        caption = "Methods on NYUv2"
        column_format = "ccccccccccc"
        label = "tab:nyuv2_methods"
    elif dataset_name == 'CelebA':
        caption = "Methods on CelebA"
        column_format = "cccccccc"
        label = "tab:celebA_methods"
    return caption, column_format, label
