import pandas as pd
import os
import numpy as np
import yaml
from yacs.config import CfgNode as CN

path = './data/'

task = 'label_skew'
attack_type = 'None'
dataset = 'fl_cifar10'  # 'fl_cifar10, PACS
averaging = 'Weight'
# fl_cifar10: MNIST, USPS, SVHN, SYN
# PACS: 'photo', 'art_painting', 'cartoon', 'sketch'
# OfficeCaltech 'caltech', 'amazon','webcam','dslr'

method_list = ['FedAVG', 'FedProx', 'FedProc']

Dataset_info = {
    'fl_cifar10': {
        'parti_num':10,
        'communication_epoch': 100
    }
}

metrics_dict = \
    {
        'label_skew' : ['in_domain_mean_acc','performance_fairness_mean_acc'],
        'domain_skew': ['in_domain_mean_acc','in_domain_all_acc'],
        'ood': ['in_domain_mean_acc', 'in_domain_all_acc','out_domain_all_acc']
    }

aim_args_dict = {
    # 'parti_num': 1,
}

aim_cfg_dict = {
    'DATASET': {
        # 'use_two_crop': "WEAK"
        # 'parti_num': 20
    },
}


def mean_metric(structure_path, metric):
    acc_dict = {}
    experiment_index = 0
    for model in os.listdir(structure_path):
        if model != '' and model in method_list:
            model_path = os.path.join(structure_path, model)
            if os.path.isdir(model_path):
                for para in os.listdir(model_path):
                    para_path = os.path.join(model_path, para)
                    args_path = para_path + '/args.csv'
                    cfg_path = para_path + '/cfg.yaml'
                    is_same = select_para(args_path, cfg_path)
                    if is_same:
                        if len(os.listdir(para_path)) > 3:
                            data = pd.read_table(para_path + '/' + metric + '.csv', sep=",")
                            data = data.loc[:, data.columns]
                            acc_value = data.values
                            mean_acc_value = np.mean(acc_value, axis=0)
                            mean_acc_value = mean_acc_value.tolist()
                            mean_acc_value = [round(item, 3) for item in mean_acc_value]
                            last_acc_vale = mean_acc_value[-5:]  # 取最后五轮结果
                            last_acc_vale = np.mean(last_acc_vale)
                            mean_acc_value.append(round(last_acc_vale, 3))
                            acc_dict[experiment_index] = [model, para] + mean_acc_value
                            experiment_index += 1
    return acc_dict


def all_metric(structure_path, metric, scale_num):
    acc_dict = {}
    experiment_index = 0
    for model in os.listdir(structure_path):
        if model != '' and model in method_list:
            model_path = os.path.join(structure_path, model)
            if os.path.isdir(model_path):  # Check this path = path to folder
                for para in os.listdir(model_path):
                    para_path = os.path.join(model_path, para)
                    args_path = para_path + '/args.csv'
                    cfg_path = para_path + '/cfg.yaml'
                    is_same = select_para(args_path, cfg_path)
                    if is_same:
                        if len(os.listdir(para_path)) > 3:
                            data = pd.read_table(para_path + '/' + metric + '_domain_all_acc.csv', sep=",")
                            data = data.loc[:, data.columns]
                            acc_value = data.values[:, 1:]
                            # parti_num = args_pd['parti_num'][0]
                            times = int(len(acc_value) / scale_num)
                            mean_acc_value = []
                            for i in range(scale_num):
                                domain_acc_value = acc_value[[scale_num * j + i for j in range(times)]]
                                domain_mean_acc_value = np.mean(domain_acc_value, axis=0)
                                last_mean_acc_value = domain_mean_acc_value[-5:]
                                # last_mean_acc_value = np.max(domain_mean_acc_value)
                                last_mean_acc_value = np.mean(last_mean_acc_value)
                                mean_acc_value.append(last_mean_acc_value)  # 添加accuracy
                            mean_acc_value = [round(item, 3) for item in mean_acc_value]
                            mean_acc_value.append(np.mean(mean_acc_value))
                            # Specific parameter value
                            acc_dict[experiment_index] = [model, para] + mean_acc_value
                            experiment_index += 1
    return acc_dict, scale_num


def select_para(args_path, cfg_path):
    args_pd = pd.read_table(args_path, sep=",")
    # aim_cfg.merge_from_file(cfg_path)
    args_pd = args_pd.loc[:, args_pd.columns]

    now_arg_dict = {}
    for k in aim_args_dict:
        now_arg_dict[k] = args_pd[k][0]

    # 判断是否全等
    is_same = True
    for k in aim_args_dict:
        if now_arg_dict[k] != aim_args_dict[k]:
            is_same = False
            break

    now_cfg = CN()

    with open(cfg_path, encoding="utf-8") as f:
        result = f.read()
        now_dict = yaml.full_load(result)

    for sub_k in aim_cfg_dict:
        try:
            now_sub_dict = now_dict[sub_k]
            aim_sub_dict = aim_cfg_dict[sub_k]
            for para_name in aim_sub_dict:
                if aim_sub_dict[para_name] != now_sub_dict[para_name]:
                    is_same = False
                    break
        except:
            pass

        if not is_same:
            break

    return is_same


if __name__ == '__main__':
    print('**************************************************************')
    specific_path = os.path.join(path, task,attack_type,dataset,averaging)
    for _, metric in enumerate(metrics_dict[task]):
        print("Task: {} Attack: {} Dataset: {} Averaging: {} Metric {}".format(task,attack_type,dataset,averaging,metric))
        # if "all" in metric:
        #     each_acc_dict, n_participants = all_metric(specific_path, metric, scale_num=scale_dict[metric])
        #     each_df = pd.DataFrame(each_acc_dict)
        #     each_df = each_df.T
        #     # pd.set_option('display.max_columns', None)
        #     column_each_acc_list = ['method', 'para'] + [str(i) for i in range(n_participants)] + ['avg']
        #     each_df.columns = column_each_acc_list
        #     print(each_df)
        if "mean" in metric:
            mean_acc_dict = mean_metric(specific_path, metric)
            mean_df = pd.DataFrame(mean_acc_dict)
            mean_df = mean_df.T
            column_mean_acc_list = ['method', 'para'] + ['E: ' + str(i) for i in
            range(Dataset_info[dataset]['communication_epoch'])] + ['MEAN']
            mean_df.columns = column_mean_acc_list
            print(mean_df)
    print('**************************************************************')
