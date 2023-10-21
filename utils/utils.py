import functools
from collections import Counter
import numpy as np
import torch
import os


def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "TEST": 31,
        "OOD": 33,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def create_if_not_exists(path: str) -> None:
    """
    Creates the specified folder if it does not exist.
    :param path: the complete path of the folder to be created
    """
    if not os.path.exists(path):
        os.makedirs(path)


def set_requires_grad(net, requires_grad):
    for param in net.parameters():
        param.requires_grad = requires_grad


def ini_client_domain(rand_domain_select, domains_list, parti_num):
    domains_len = len(domains_list)
    # 是否随机采样数据集
    if rand_domain_select:
        # 每个数据集最大数量
        max_num = 10
        is_ok = False
        while not is_ok:
            selected_domain_list = np.random.choice(domains_list, size=parti_num - domains_len, replace=True, p=None)
            selected_domain_list = list(selected_domain_list) + domains_list
            # selected_domain_list = np.random.choice(domains_list, size=parti_num, replace=True, p=None)
            selected_domain_list = list(selected_domain_list)
            result = dict(Counter(selected_domain_list))
            for k in result:
                if result[k] > max_num:
                    is_ok = False
                    break
            else:
                is_ok = True
    else:
        selected_domain_dict = {}
        for domain in domains_list:
            selected_domain_dict[domain] = parti_num // domains_len
        # selected_domain_dict = {'MNIST': 6, 'USPS': 4, 'SVHN': 3, 'SYN': 7}  # base
        selected_domain_list = []
        for k in selected_domain_dict:
            domain_num = selected_domain_dict[k]
            for i in range(domain_num):
                selected_domain_list.append(k)
        selected_domain_list = np.random.permutation(selected_domain_list)
    result = Counter(selected_domain_list)
    print(log_msg(selected_domain_list))
    print(log_msg(result))
    return selected_domain_list


def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "TEST": 31,
        "OOD": 33,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def cal_client_weight(online_clients_list, client_domain_list, freq):
    client_weight = {}
    for index, item in enumerate(online_clients_list):  # 遍历循环当前的参与者
        client_domain = client_domain_list[item]
        client_freq = freq[index]
        client_weight[str(item) + ':' + client_domain] = round(client_freq, 3)
    return client_weight


def row_into_parameters(row, parameters):
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x, y: x * y, param.shape)
        current_data = row[offset:offset + new_size]

        param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size
