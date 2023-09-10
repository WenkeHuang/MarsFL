import copy

from Attack.utils import attack_net_para
from Methods.utils.meta_methods import FederatedMethod
from utils.logger import CsvWriter

import torch
import numpy as np
from utils.utils import log_msg


def cal_top_one_five(net, test_dl, device):
    correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
    for batch_idx, (images, labels) in enumerate(test_dl):
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, max5 = torch.topk(outputs, 5, dim=-1)
            labels = labels.view(-1, 1)
            top1 += (labels == max5[:, 0:1]).sum().item()
            top5 += (labels == max5).sum().item()
            total += labels.size(0)
    top1acc = round(100 * top1 / total, 2)
    top5acc = round(100 * top5 / total, 2)
    return top1acc, top5acc


def global_personal_evaluation(optimizer: FederatedMethod, test_loader: dict, client_domain_list: list):
    personal_domain_accs = []
    for client_index in range(optimizer.cfg.DATASET.parti_num):
        client_domain = client_domain_list[client_index]
        client_net = optimizer.nets_list[client_index]
        client_test_loader = test_loader[client_domain]
        client_net.eval()
        top1acc, _ = cal_top_one_five(net=client_net, test_dl=client_test_loader, device=optimizer.device)
        personal_domain_accs.append(top1acc)
        client_net.train()
    mean_personal_domain_acc = round(np.mean(personal_domain_accs, axis=0), 3)
    return personal_domain_accs, mean_personal_domain_acc


def global_in_evaluation(optimizer: FederatedMethod, test_loader: dict, in_domain_list: list):
    in_domain_accs = []
    # global_net = optimizer.global_net
    # global_net.eval()
    for in_domain in in_domain_list:
        # 如果有给不同domain不同网络 那么就使用
        if hasattr(optimizer, 'global_net_dict'):
            global_net = optimizer.global_net_dict[in_domain]

        else:
            global_net = optimizer.global_net
        global_net.eval()

        test_domain_dl = test_loader[in_domain]
        top1acc, _ = cal_top_one_five(net=global_net, test_dl=test_domain_dl, device=optimizer.device)
        in_domain_accs.append(top1acc)

        global_net.train()

    mean_in_domain_acc = round(np.mean(in_domain_accs, axis=0), 3)

    return in_domain_accs, mean_in_domain_acc


def global_out_evaluation(optimizer: FederatedMethod, test_loader: dict, out_domain: str):
    test_out_domain_dl = test_loader[out_domain]

    # 如果有给不同domain不同网络 那么就使用
    if hasattr(optimizer, 'global_net_dict'):
        global_net = optimizer.global_net_dict[out_domain]

    else:
        global_net = optimizer.global_net
    global_net.eval()

    top1acc, _ = cal_top_one_five(net=global_net, test_dl=test_out_domain_dl, device=optimizer.device)
    out_acc = top1acc
    global_net.train()
    return out_acc


def train(fed_method, private_dataset, args, cfg, client_domain_list) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, cfg)

    # GA需要测聚合前后的本地效果
    fed_method.test_loaders = private_dataset.test_loader

    if hasattr(fed_method, 'ini'):
        # 要在这一步 把方法的个性化绑定进去
        fed_method.ini()

    if args.task == 'OOD':
        in_domain_accs_dict = {}
        mean_in_domain_acc_list = []
        out_domain_accs_dict = {}
    elif args.task == 'label_skew':
        accs_list = []
    elif args.task =='domain_skew':
        in_domain_accs_dict = {}
        mean_in_domain_acc_list = []
    elif args.task == 'attack':
        if cfg.attack.dataset_type == 'single_domain':
            accs_list = []
        elif cfg.attack.dataset_type == 'multi_domain':
            in_domain_accs_dict = {}
            mean_in_domain_acc_list = []

    communication_epoch = cfg.DATASET.communication_epoch
    for epoch_index in range(communication_epoch):
        fed_method.epoch_index = epoch_index

        fed_method.local_update(private_dataset.train_loaders)
        if args.attack_type == 'byzantine':
            attack_net_para(args, cfg, fed_method)

        fed_method.sever_update(private_dataset.train_loaders)

        if args.task == 'OOD':
            '''
            测试分为三种：
            person_domain_accs: 私有模型在本地Domain的精度测试 + 提供mean的值
            domain_accs: 全局模型在InDomain的精度测试 + 提供mean的值
            out_domain_acc: 全局模型在OutDomain的精度测试
            '''

            '''
            全局模型在参与者的Domain上的精度 & 存储
            '''
            if hasattr(fed_method, 'weight_dict'):
                weight_dict = fed_method.weight_dict
                if args.csv_log:
                    csv_writer.write_weight(weight_dict, epoch_index, client_domain_list)

            domain_accs, mean_in_domain_acc = global_in_evaluation(fed_method, private_dataset.test_loader, private_dataset.in_domain_list)
            mean_in_domain_acc_list.append(mean_in_domain_acc)
            for index, in_domain in enumerate(private_dataset.in_domain_list):
                if in_domain in in_domain_accs_dict:
                    in_domain_accs_dict[in_domain].append(domain_accs[index])
                else:
                    in_domain_accs_dict[in_domain] = [domain_accs[index]]
            print(log_msg(f"The {epoch_index} Epoch: In Domain Mean Acc: {mean_in_domain_acc} Method: {args.method} CSV: {args.csv_name}", "TEST"))
            '''
            全局模型在未知的Domain上的精度 & 存储
            '''
            if cfg[args.task].out_domain != "NONE":
                out_domain_acc = global_out_evaluation(fed_method, private_dataset.test_loader, cfg[args.task].out_domain)
                if cfg[args.task].out_domain in out_domain_accs_dict:
                    out_domain_accs_dict[cfg[args.task].out_domain].append(out_domain_acc)
                else:
                    out_domain_accs_dict[cfg[args.task].out_domain] = [out_domain_acc]
                print(log_msg(f"The {epoch_index} Epoch: Out Domain {cfg[args.task].out_domain} Acc: {out_domain_acc} Method: {args.method} CSV: {args.csv_name}", "OOD"))

            if args.save_checkpoint:
                # 每十个交流周期进行一次存储
                if epoch_index % 10 == 0 or epoch_index == communication_epoch - 1:
                    fed_method.save_checkpoint()

        elif args.task == 'label_skew':
            top1acc, _ = cal_top_one_five(fed_method.global_net, private_dataset.test_loader, fed_method.device)
            accs_list.append(top1acc)
            print(log_msg(f'The {epoch_index} Epoch: Acc:{top1acc}'))
        elif args.task == 'domain_skew':
            domain_accs, mean_in_domain_acc = global_in_evaluation(fed_method, private_dataset.test_loader, private_dataset.domain_list)
            mean_in_domain_acc_list.append(mean_in_domain_acc)
            for index, in_domain in enumerate(private_dataset.domain_list):
                if in_domain in in_domain_accs_dict:
                    in_domain_accs_dict[in_domain].append(domain_accs[index])
                else:
                    in_domain_accs_dict[in_domain] = [domain_accs[index]]

            print(log_msg(f"The {epoch_index} Epoch: Domain Mean Acc: {mean_in_domain_acc} Method: {args.method} CSV: {args.csv_name}", "TEST"))

        # if args.attack == 'byzantine':
        #     if cfg.attack.dataset_type == 'single_domain':
        #         top1acc, _ = cal_top_one_five(fed_method.global_net, private_dataset.test_loader, fed_method.device)
        #         accs_list.append(top1acc)
        #         print(log_msg(f'The {epoch_index} Epoch: Acc:{top1acc}'))
        #     elif cfg.attack.dataset_type == 'multi_domain':
        #         domain_accs, mean_in_domain_acc = global_in_evaluation(fed_method, private_dataset.test_loader, private_dataset.domain_list)
        #         mean_in_domain_acc_list.append(mean_in_domain_acc)
        #         for index, in_domain in enumerate(private_dataset.domain_list):
        #             if in_domain in in_domain_accs_dict:
        #                 in_domain_accs_dict[in_domain].append(domain_accs[index])
        #             else:
        #                 in_domain_accs_dict[in_domain] = [domain_accs[index]]
        #
        #         print(log_msg(f"The {epoch_index} Epoch: Domain Mean Acc: {mean_in_domain_acc} Method: {args.method} CSV: {args.csv_name}", "TEST"))

    if args.csv_log:
        if args.task == 'OOD':
            csv_writer.write_acc(mean_in_domain_acc_list, name='in_domain', mode='MEAN')
            csv_writer.write_acc(in_domain_accs_dict, name='in_domain', mode='ALL')
            if cfg[args.task].out_domain != "NONE":
                csv_writer.write_acc(out_domain_accs_dict, name='out_domain', mode='ALL')

        elif args.task == 'label_skew':
            csv_writer.write_acc(accs_list, name='label_skew', mode='MEAN')

        elif args.task == 'domain_skew':
            csv_writer.write_acc(mean_in_domain_acc_list, name='in_domain', mode='MEAN')
            csv_writer.write_acc(in_domain_accs_dict, name='in_domain', mode='ALL')

        # if args.attack == 'byzantine':
        #     if cfg.attack.dataset_type == 'multi_domain':
        #         csv_writer.write_acc(mean_in_domain_acc_list, name='in_domain', mode='MEAN')
        #         csv_writer.write_acc(in_domain_accs_dict, name='in_domain', mode='ALL')
        #
        #     elif cfg.attack.dataset_type == 'single_domain':
        #         csv_writer.write_acc(accs_list, name='label_skew', mode='MEAN')
