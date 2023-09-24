import copy

from Attack.byzantine.utils import attack_net_para
from Methods.utils.meta_methods import FederatedMethod
from utils.logger import CsvWriter
import torch
import numpy as np
from utils.utils import log_msg


def cal_top_one_five(net, test_dl, device):
    net.eval()
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
    net.train()
    top1acc = round(100 * top1 / total, 2)
    top5acc = round(100 * top5 / total, 2)
    return top1acc, top5acc


# def global_personal_evaluation(optimizer: FederatedMethod, test_loader: dict, client_domain_list: list):
#     personal_domain_accs = []
#     for client_index in range(optimizer.cfg.DATASET.parti_num):
#         client_domain = client_domain_list[client_index]
#         client_net = optimizer.nets_list[client_index]
#         client_test_loader = test_loader[client_domain]
#         client_net.eval()
#         top1acc, _ = cal_top_one_five(net=client_net, test_dl=client_test_loader, device=optimizer.device)
#         personal_domain_accs.append(top1acc)
#         client_net.train()
#     mean_personal_domain_acc = round(np.mean(personal_domain_accs, axis=0), 3)
#     return personal_domain_accs, mean_personal_domain_acc

def global_in_evaluation(optimizer: FederatedMethod, test_loader: dict, in_domain_list: list):
    in_domain_accs = []
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


def cal_sim_con_weight(**kwargs):
    # para
    optimizer = kwargs['optimizer']
    test_loader = kwargs['test_loader']
    task = kwargs['task']
    domain_list = kwargs['domain_list']

    global_net = optimizer.global_net
    if task == 'label_skew':
        overall_acc = cal_top_one_five(net=global_net, test_dl=test_loader, device=optimizer.device)
        overall_top1_acc = overall_acc[0]
    elif task == 'domain_skew':
        accs = []
        for in_domain in domain_list:

            test_domain_dl = test_loader[in_domain]
            top1acc, _ = cal_top_one_five(net=global_net, test_dl=test_domain_dl, device=optimizer.device)
            accs.append(top1acc)
        overall_top1_acc = round(np.mean(accs, axis=0), 3)

    partial_acc_list = []
    nets_list = optimizer.nets_list_before_agg

    if hasattr(optimizer, 'aggregation_weight_list'):
        aggregation_weight_list = optimizer.aggregation_weight_list
    else:
        print('not support this method')
        return

    for index_out, _ in enumerate(optimizer.online_clients_list):
        global_w = {}
        temp_global_net = copy.deepcopy(global_net)
        temp_freq = copy.deepcopy(aggregation_weight_list)
        temp_freq[index_out] = 0
        temp_freq = temp_freq / np.sum(temp_freq)
        first = True
        for index, net_id in enumerate(optimizer.online_clients_list):
            net = nets_list[net_id]  # 获取 online client 中对应的网络的索引
            net_para = net.state_dict()
            # 排除所有不用的的部分
            except_part = []
            used_net_para = {}
            for k, v in net_para.items():
                is_in = False
                for part_str_index in range(len(except_part)):
                    if except_part[part_str_index] in k:
                        is_in = True
                        break
                # 只有不在的排除范围内的 才选中
                if not is_in:
                    used_net_para[k] = v
            # 只加载需要的参数
            if first:
                first = False
                for key in used_net_para:
                    global_w[key] = used_net_para[key] * temp_freq[index]
            else:
                for key in used_net_para:
                    global_w[key] += used_net_para[key] * temp_freq[index]
        temp_global_net.load_state_dict(global_w, strict=False)

        if task == 'label_skew':
            partial_top1_acc, partial_top5_acc = cal_top_one_five(net=temp_global_net, test_dl=test_loader, device=optimizer.device)
        elif task == 'domain_skew':
            accs = []
            for in_domain in domain_list:
                test_domain_dl = test_loader[in_domain]
                top1acc, _ = cal_top_one_five(net=temp_global_net, test_dl=test_domain_dl, device=optimizer.device)
                accs.append(top1acc)
            partial_top1_acc = round(np.mean(accs, axis=0), 3)

        partial_acc_list.append(partial_top1_acc)

    overall_top1_acc_list = [overall_top1_acc] * len(partial_acc_list)
    dif_ac = [a - b + 1e-5 for a, b in zip(overall_top1_acc_list, partial_acc_list)]
    dif_ac = dif_ac / (np.sum(dif_ac))
    print(partial_acc_list)

    sim_con_weight = dif_ac.dot(aggregation_weight_list) / (
            np.linalg.norm(dif_ac) * np.linalg.norm(aggregation_weight_list))
    return sim_con_weight


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

    if hasattr(fed_method, 'ini'):
        # 要在这一步 把方法的个性化绑定进去
        fed_method.ini()

    if args.task == 'OOD':
        in_domain_accs_dict = {}  # Query-Client Accuracy A^u
        mean_in_domain_acc_list = []  # Cross-Client Accuracy A^U
        out_domain_accs_dict = {}  # Out-Client Accuracy A^o
        fed_method.out_train_loader = private_dataset.out_train_loader
    elif args.task == 'label_skew':
        mean_in_domain_acc_list = []
        if args.attack_type == 'None':
            contribution_match_degree_list = []  # Contribution Match Degree \bm{\mathcal{E}}
        fed_method.net_cls_counts = private_dataset.net_cls_counts # label stastic
    elif args.task == 'domain_skew':
        in_domain_accs_dict = {}  # Query-Client Accuracy \bm{\mathcal{A}}}^{u}
        mean_in_domain_acc_list = []  # Cross-Client Accuracy A^U \bm{\mathcal{A}}}^{\mathcal{U}
        performance_variane_list = []
        if args.attack_type == 'None':
            contribution_match_degree_list = []
    if args.attack_type == 'backdoor':
        attack_success_rate = []

    communication_epoch = cfg.DATASET.communication_epoch
    for epoch_index in range(communication_epoch):
        fed_method.epoch_index = epoch_index

        # Client 端操作
        fed_method.local_update(private_dataset.train_loaders)
        fed_method.nets_list_before_agg = copy.deepcopy(fed_method.nets_list)

        if args.attack_type == 'byzantine':
            attack_net_para(args, cfg, fed_method)

        # Server 端操作
        fed_method.sever_update(private_dataset.train_loaders)

        if args.task == 'OOD':
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

        else:
            if 'mean_in_domain_acc_list' in locals():
                print("进行 mean_in_domain_acc_list 评估")
                top1acc, _ = cal_top_one_five(fed_method.global_net, private_dataset.test_loader, fed_method.device)
                mean_in_domain_acc_list.append(top1acc)
                print(log_msg(f'The {epoch_index} Epoch: Acc:{top1acc}', "TEST"))

            if 'contribution_match_degree_list' in locals():
                print("进行 contribution_match_degree_list 评估")
                if epoch_index % 10 == 0 or epoch_index == communication_epoch - 1:
                    con_fair_metric = cal_sim_con_weight(optimizer=fed_method, test_loader=private_dataset.test_loader,
                                                         domain_list=None, task=args.task)
                    contribution_match_degree_list.append(con_fair_metric)
                else:
                    con_fair_metric = 0
                    contribution_match_degree_list.append(con_fair_metric)
                print(log_msg(f'The {epoch_index} Method: {args.method} Epoch: Con Fair:{con_fair_metric}', "TEST"))

            if 'in_domain_accs_dict' in locals():
                print("进行 in_domain_accs_dict 评估")
                domain_accs, mean_in_domain_acc = global_in_evaluation(fed_method, private_dataset.test_loader, private_dataset.domain_list)
                perf_var = np.var(domain_accs, ddof=0)
                performance_variane_list.append(perf_var)
                mean_in_domain_acc_list.append(mean_in_domain_acc)

                for index, in_domain in enumerate(private_dataset.domain_list):
                    if in_domain in in_domain_accs_dict:
                        in_domain_accs_dict[in_domain].append(domain_accs[index])
                    else:
                        in_domain_accs_dict[in_domain] = [domain_accs[index]]
                print(log_msg(f"The {epoch_index} Epoch: Mean Acc: {mean_in_domain_acc} Method: {args.method} Per Var: {perf_var} ", "TEST"))

            if 'attack_success_rate' in locals():
                top1acc, _ = cal_top_one_five(fed_method.global_net, private_dataset.backdoor_test_loader, fed_method.device)
                attack_success_rate.append(top1acc)
                print(log_msg(f'The {epoch_index} Epoch: attack success rate:{top1acc}'))

    if args.csv_log:
        if args.task == 'OOD':
            csv_writer.write_acc(mean_in_domain_acc_list, name='in_domain', mode='MEAN')
            csv_writer.write_acc(in_domain_accs_dict, name='in_domain', mode='ALL')
            if cfg[args.task].out_domain != "NONE":
                csv_writer.write_acc(out_domain_accs_dict, name='out_domain', mode='ALL')

        elif args.task == 'label_skew':
            csv_writer.write_acc(mean_in_domain_acc_list, name='in_domain', mode='MEAN')
            csv_writer.write_acc(contribution_match_degree_list, name='contribution_fairness', mode='MEAN')

        elif args.task == 'domain_skew':
            csv_writer.write_acc(mean_in_domain_acc_list, name='in_domain', mode='MEAN')
            csv_writer.write_acc(in_domain_accs_dict, name='in_domain', mode='ALL')
            csv_writer.write_acc(contribution_match_degree_list, name='contribution_fairness', mode='MEAN')
            csv_writer.write_acc(performance_variane_list, name='performance_variance', mode='MEAN')

        if args.attack_type == 'backdoor':
            csv_writer.write_acc(attack_success_rate, name='attack_success_rate', mode='MEAN')

        # if args.save_checkpoint:
        #     # 每十个交流周期进行一次存储
        #     if epoch_index % 10 == 0 or epoch_index == communication_epoch - 1:
        #         fed_method.save_checkpoint()