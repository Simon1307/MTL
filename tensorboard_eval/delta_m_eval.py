import sys
import numpy as np
from os import listdir, path
import statistics
import pandas as pd
import argparse
from utils import *


def parse_args(arg_parser):
    parser.add_argument('--base_path', default='/cityscapes_table2/',
                        type=str, help='Directory containing tensorboard_logs events of experiments')
    parser.add_argument('--result_path', default='/results/cs/cityscapes_table2.csv',
                        type=str, help='Path to save table')
    parser.add_argument('--dataset_name', default='Cityscapes', type=str, help='Name of the dataset')
    parser.add_argument('--num_runs_per_method', type=int, default=5, help='Number of runs per method')
    parser.add_argument('--num_tasks', type=int, default=2, help='Number of tasks of dataset')
    return arg_parser.parse_args()


def main(params):
    base_path = params.base_path
    result_path = params.result_path
    dataset_name = params.dataset_name
    num_tasks = params.num_tasks
    num_runs_per_method = params.num_runs_per_method
    r = [f for f in sorted(listdir(base_path))]
    scalars_per_exp = dict.fromkeys(r)
    for i, directory in enumerate(r):  # loop over all experiments
        ea = read_tb(path.join(base_path, directory))
        # Extract all scalar values of all epochs in a dict
        scalar_values = get_all_scalars(ea)
        scalars_per_exp[directory] = scalar_values
    full_identifier_exps = list(scalars_per_exp.keys())
    # remove date from experiment names to compare experiment names
    all_experiment_names = [name[:-21] for name in full_identifier_exps]
    # group runs together where only the seed is differing
    experiment_groups = get_experiment_groups(all_experiment_names)

    print("Calculating delta m and task-specific metrics for the following experiment groups")
    columns, attributes = get_column_names(dataset_name, num_tasks)

    df = pd.DataFrame(columns=columns)
    for e, group in enumerate(experiment_groups):
        print(f"Experiment group {e}:\n")
        experiments = []
        for index in group:
            print(full_identifier_exps[index])
            experiments.append(full_identifier_exps[index])
        print("-"*20)

        if dataset_name == 'Cityscapes':
            train_times = []
            all_delta_m = []
            all_mIoU = []
            all_pixAcc = []
            all_abs_err = []
            all_rel_err = []
            for n, exp in enumerate(experiments):
                # Calculate delta m of each experiment 10 times based on the last 10 epoch test
                epochs = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
                for r, epoch in enumerate(epochs):
                    delta_m, mIoU, pixAcc, abs_err, rel_err = calc_epoch_specific_delta_m_cityscapes(scalars_per_exp,
                                                                                                     exp, epoch)
                    all_delta_m.append(delta_m)
                    all_mIoU.append(mIoU)
                    all_pixAcc.append(pixAcc)
                    all_abs_err.append(abs_err)
                    all_rel_err.append(rel_err)
                try:
                    train_time = scalars_per_exp[exp]["/train_time"][0]
                except:
                    train_time = -1.0
                train_times.append(train_time)
            if not train_times:
                train_times.append(-1.0)
            train_time_mean = statistics.mean(train_times)
            delta_m_mean = np.asarray(all_delta_m).mean(axis=0)
            delta_m_std = np.asarray(all_delta_m).std(axis=0, ddof=0)
            mIoU_mean = np.asarray(all_mIoU).mean(axis=0)
            pixAcc_mean = np.asarray(all_pixAcc).mean(axis=0)
            abs_err_mean = np.asarray(all_abs_err).mean(axis=0)
            rel_err_mean = np.asarray(all_rel_err).mean(axis=0)
            # create list of values in same order as column names and write list as row to dataframe
            exp = replace_exp_name(exp, num_runs_per_method)
            # split exp name to parameters and values of params
            params_values = exp.split(sep='--')
            exp = '--'.join(params_values[:-1])
            values = [exp, round(train_time_mean, 2), round(mIoU_mean, 3), round(pixAcc_mean, 3),
                      round(abs_err_mean, 4), round(rel_err_mean, 1), round(delta_m_mean, 1), round(delta_m_std, 1)]

        elif dataset_name == 'NYU':
            train_times = []
            all_delta_m = []
            all_mIoU = []
            all_pixAcc = []
            all_abs_err = []
            all_rel_err = []
            all_n_mean = []
            all_n_median = []
            all_n_11_25 = []
            all_n_22_5 = []
            all_n_30 = []
            for n, exp in enumerate(experiments):
                # Calculate delta m of each experiment 10 times based on the last 10 epoch test
                epochs = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
                for r, epoch in enumerate(epochs):
                    delta_m, mIoU, pixAcc, abs_err, rel_err, n_mean, n_median, n_11_25, n_22_5, n_30 = \
                        calc_epoch_specific_delta_m_nyu(
                        scalars_per_exp, exp, epoch)
                    all_delta_m.append(delta_m)
                    all_mIoU.append(mIoU)
                    all_pixAcc.append(pixAcc)
                    all_abs_err.append(abs_err)
                    all_rel_err.append(rel_err)
                    all_n_mean.append(n_mean)
                    all_n_median.append(n_median)
                    all_n_11_25.append(n_11_25)
                    all_n_22_5.append(n_22_5)
                    all_n_30.append(n_30)
                try:
                    train_time = scalars_per_exp[exp]["/train_time"][0]
                except:
                    train_time = -1.0
                train_times.append(train_time)
            train_time_mean = statistics.mean(train_times)
            delta_m_mean = np.asarray(all_delta_m).mean(axis=0)
            delta_m_std = np.asarray(all_delta_m).std(axis=0, ddof=0)
            mIoU_mean = np.asarray(all_mIoU).mean(axis=0)
            pixAcc_mean = np.asarray(all_pixAcc).mean(axis=0)
            abs_err_mean = np.asarray(all_abs_err).mean(axis=0)
            rel_err_mean = np.asarray(all_rel_err).mean(axis=0)
            n_mean_mean = np.asarray(all_n_mean).mean(axis=0)
            n_median_mean = np.asarray(all_n_median).mean(axis=0)
            n_11_25_mean = np.asarray(all_n_11_25).mean(axis=0)
            n_22_5_mean = np.asarray(all_n_22_5).mean(axis=0)
            n_30_mean = np.asarray(all_n_30).mean(axis=0)
            # create list of values in same order as column names and write list as row to dataframe
            exp = replace_exp_name(exp, num_runs_per_method)
            # split exp name to parameters and values of params
            params_values = exp.split(sep='--')
            exp = '--'.join(params_values[:-1])
            values = [exp, round(train_time_mean, 2), round(mIoU_mean, 3), round(pixAcc_mean, 3),
                      round(abs_err_mean, 3), round(rel_err_mean, 3), round(n_mean_mean, 1), round(n_median_mean, 1),
                      round(n_11_25_mean, 3), round(n_22_5_mean, 3), round(n_30_mean, 3), round(delta_m_mean, 2),
                      round(delta_m_std, 1)]

        elif dataset_name == 'CelebA':
            train_times = []
            all_delta_m = []
            all_accuracy_errors = {attribute: [] for attribute in attributes}
            average_accuracy_error = 0.0
            ctr = 0
            for n, exp in enumerate(experiments):
                # Calculate delta m of each experiment 10 times based on the last 10 epoch test
                epochs = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]

                for r, epoch in enumerate(epochs):
                    delta_m, accuracy_errors = calc_epoch_specific_delta_m_celebA(scalars_per_exp, exp, epoch, attributes)
                    all_delta_m.append(delta_m)
                    for i, attribute in enumerate(attributes):
                        all_accuracy_errors[attribute].append(accuracy_errors[i])
                        average_accuracy_error += accuracy_errors[i]
                        ctr += 1
                try:
                    train_time = scalars_per_exp[exp]["/train_time"][0]
                except:
                    train_time = -1.0
                train_times.append(train_time)
            train_time_mean = statistics.mean(train_times)
            delta_m_mean = np.asarray(all_delta_m).mean(axis=0)
            delta_m_std = np.asarray(all_delta_m).std(axis=0, ddof=0)
            average_accuracy_error /= ctr
            # create list of values in same order as column names and write list as row to dataframe
            exp = replace_exp_name(exp, num_runs_per_method)
            # split exp name to parameters and values of params
            params_values = exp.split(sep='--')
            exp = '--'.join(params_values[:-1])
            values = [exp, round(train_time_mean, 2)]
            for attribute in attributes:
                arr = np.asarray(all_accuracy_errors[attribute])
                acc_err_mean = arr.mean(axis=0)
                values.append(round(acc_err_mean, 2))
            values.append(round(delta_m_mean, 1))
            values.append(round(delta_m_std, 1))
            values.append(round(average_accuracy_error, 2))

        df.loc[e] = values

    df = rename_methods(df)
    df = prepare_df(df, dataset_name)
    caption, column_format, label = get_table_meta_data(dataset_name)
    with open(result_path, "w") as tf:
        tf.write(df.to_latex(index=False,
                             caption=caption,
                             label=label,
                             column_format=column_format,
                             multicolumn_format="c",
                             escape=False,
                             multirow=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configurations')
    parameters = parse_args(parser)
    main(parameters)
