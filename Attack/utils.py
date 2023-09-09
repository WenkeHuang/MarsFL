import copy

import numpy as np
import torch

from Datasets.utils.utils import noisify
from utils.utils import row_into_parameters

# attack攻击类型
attack_type_dict = {
    'PairFlip': 'dataset',
    'SymFlip': 'dataset',
    'RandomNoise': 'model_para',
    'lie_attack': 'model_para',
    'min_max': 'model_para',
    'min_sum': 'model_para',
}


# 数据集攻击
def attack_dataset(args, cfg, private_dataset, client_type):
    # 攻击类型是数据集攻击 那么修改数据集的内容
    if attack_type_dict[cfg[args.task].evils] == 'dataset':
        for i in range(len(client_type)):
            if not client_type[i]:
                dataset = private_dataset.train_loaders[i].dataset
                train_labels = np.asarray([[dataset.targets[i]] for i in range(len(dataset.targets))])
                train_noisy_labels, actual_noise_rate = noisify(
                    train_labels=train_labels,
                    noise_type=cfg[args.task].evils,
                    noise_rate=cfg[args.task].noise_data_rate,
                    nb_classes=len(np.unique(train_labels)))

                train_noisy_labels = train_noisy_labels.reshape(-1)
                dataset.targets = train_noisy_labels


# 网络参数攻击
def attack_net_para(args, cfg, fed_method):
    temp_net = copy.deepcopy(fed_method.global_net)
    if cfg[args.task].evils == 'RandomNoise':
        for i in fed_method.online_clients_list:
            if fed_method.client_type[i] == False:
                random_net = copy.deepcopy(fed_method.random_net)
                fed_method.nets_list[i] = random_net

    elif cfg[args.task].evils == 'AddNoise':
        for i in fed_method.online_clients_list:
            if fed_method.client_type[i] == False:
                sele_net = fed_method.nets_list[i]
                random_net = copy.deepcopy(fed_method.random_net)
                noise_weight = 0.5
                for name, param in sele_net.state_dict().items():
                    param += torch.tensor(copy.deepcopy(noise_weight * (random_net.state_dict()[name] - param)), dtype=param.dtype)

    elif cfg[args.task].evils == 'lie_attack':
        # 计算z的值
        n = len(fed_method.online_clients_list)
        m = n - sum(fed_method.client_type)
        s = n // 2 + 1 - m
        z = np.exp(-1 * (s ** 2)) / (2 / ((2 * np.pi) ** 0.5))

        all_net_delta = []
        with torch.no_grad():
            for i in fed_method.online_clients_list:
                if fed_method.client_type[i] == True:
                    sele_net = fed_method.nets_list[i]

                    net_delta = []
                    for name, param0 in temp_net.state_dict().items():
                        param1 = sele_net.state_dict()[name]
                        delta = (param1.detach() - param0.detach())
                        net_delta.append(copy.deepcopy(delta.view(-1)))
                    net_delta = torch.cat(net_delta, dim=0).view(1, -1)

                    all_net_delta.append(net_delta)
            all_net_delta = torch.cat(all_net_delta, dim=0)
            avg = torch.mean(all_net_delta, dim=0)
            std = torch.std(all_net_delta, dim=0)
        for i in fed_method.online_clients_list:
            if fed_method.client_type[i] == False:
                sele_net = fed_method.nets_list[i]
                row_into_parameters((avg + z * std).cpu().numpy(), sele_net.parameters())

    elif cfg[args.task].evils == 'min_sum':

        # 计算好客户端的模型变化量和对应均值
        all_net_delta = []
        with torch.no_grad():
            for i in fed_method.online_clients_list:
                if fed_method.client_type[i] == True:
                    sele_net = fed_method.nets_list[i]

                    net_delta = []
                    for name, param0 in temp_net.state_dict().items():
                        param1 = sele_net.state_dict()[name]
                        delta = (param1.detach() - param0.detach())
                        net_delta.append(copy.deepcopy(delta.view(-1)))
                    net_delta = torch.cat(net_delta, dim=0).view(1, -1)

                    all_net_delta.append(net_delta)
            all_net_delta = torch.cat(all_net_delta, dim=0)
            avg_delta = torch.mean(all_net_delta, dim=0)

        if cfg[args.task].dev_type == 'unit_vec':
            deviation = avg_delta / torch.norm(avg_delta)  # unit vector, dir opp to good dir
        elif cfg[args.task].dev_type == 'sign':
            deviation = torch.sign(avg_delta)
        elif cfg[args.task].dev_type == 'std':
            deviation = torch.std(all_net_delta, 0)

        lamda_fail = torch.Tensor([cfg[args.task].lamda]).float().to(fed_method.device)
        lamda = torch.Tensor([cfg[args.task].lamda]).float().to(fed_method.device)
        lamda_succ = 0

        distances = []
        for update in all_net_delta:
            distance = torch.norm((all_net_delta - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        scores = torch.sum(distances, dim=1)
        min_score = torch.min(scores)
        del distances

        while torch.abs(lamda_succ - lamda) > cfg[args.task].threshold_diff:
            mal_update = (avg_delta - lamda * deviation)
            distance = torch.norm((all_net_delta - mal_update), dim=1) ** 2
            score = torch.sum(distance)

            if score <= min_score:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        # print(lamda_succ)
        mal_update = (avg_delta - lamda_succ * deviation)
        for i in fed_method.online_clients_list:
            if fed_method.client_type[i] == False:
                sele_net = fed_method.nets_list[i]
                row_into_parameters(mal_update.cpu().numpy(), sele_net.parameters())

    elif cfg[args.task].evils == 'min_max':
        # 计算好客户端的模型变化量和对应均值
        all_net_delta = []
        with torch.no_grad():
            for i in fed_method.online_clients_list:
                if fed_method.client_type[i] == True:
                    sele_net = fed_method.nets_list[i]

                    net_delta = []
                    for name, param0 in temp_net.state_dict().items():
                        param1 = sele_net.state_dict()[name]
                        delta = (param1.detach() - param0.detach())
                        net_delta.append(copy.deepcopy(delta.view(-1)))
                    net_delta = torch.cat(net_delta, dim=0).view(1, -1)

                    all_net_delta.append(net_delta)
            all_net_delta = torch.cat(all_net_delta, dim=0)
            avg_delta = torch.mean(all_net_delta, dim=0)

        if cfg[args.task].dev_type == 'unit_vec':
            deviation = avg_delta / torch.norm(avg_delta)  # unit vector, dir opp to good dir
        elif cfg[args.task].dev_type == 'sign':
            deviation = torch.sign(avg_delta)
        elif cfg[args.task].dev_type == 'std':
            deviation = torch.std(all_net_delta, 0)

        lamda_fail = torch.Tensor([cfg[args.task].lamda]).float().to(fed_method.device)
        lamda = torch.Tensor([cfg[args.task].lamda]).float().to(fed_method.device)
        lamda_succ = 0

        distances = []
        for update in all_net_delta:
            distance = torch.norm((all_net_delta - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        max_distance = torch.max(distances)
        del distances

        while torch.abs(lamda_succ - lamda) > cfg[args.task].threshold_diff:
            mal_update = (avg_delta - lamda * deviation)
            distance = torch.norm((all_net_delta - mal_update), dim=1) ** 2
            max_d = torch.max(distance)

            if max_d <= max_distance:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = (avg_delta - lamda_succ * deviation)
        for i in fed_method.online_clients_list:
            if fed_method.client_type[i] == False:
                sele_net = fed_method.nets_list[i]
                row_into_parameters(mal_update.cpu().numpy(), sele_net.parameters())
