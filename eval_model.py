import numpy as np
import torch

from Datasets import Priv_NAMES, get_prive_dataset
from Methods import Fed_Methods_NAMES, get_fed_method
from Aggregations import Aggregation_NAMES, get_fed_aggregation
from utils.utils import ini_client_domain, log_msg
from Backbones import get_private_backbones
from utils.conf import set_random_seed, config_path
from argparse import ArgumentParser
from utils.cfg import CFG as cfg, show_cfg
from utils.training import global_in_evaluation, global_out_evaluation
import argparse
import datetime
import socket
import uuid
import copy
import os


def parse_args():
    parser = ArgumentParser(description='Federated Learning', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=1, help='The Device Id for Experiment')
    parser.add_argument('--dataset', type=str, default='Digits',  # Digits,PACS PACScomb OfficeHome
                        choices=Priv_NAMES, help='Which scenario to perform experiments on.')
    parser.add_argument('--rand_domain_select', type=bool, default=True, help='The Local Domain Selection')

    '''
    Whether Conduct OOD Experiments NONE
    '''
    # NONE
    # Digits: MNIST, USPS, SVHN, SYN
    # PACS: 'photo', 'art_painting', 'cartoon', 'sketch'
    # OfficeCaltech 'caltech', 'amazon','webcam','dslr'
    # OfficeHome 'Art', 'Clipart', 'Product', 'Real World'
    # DomainNet 'clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'
    parser.add_argument('--OOD', type=str, default='MNIST', help='Whether conduct OOD Experiments')

    '''
    Federated Optimizer Hyper-Parameter
    '''
    parser.add_argument('--method', type=str, default='FedProxCOSAddGlobal',
                        help='Federated Method name.', choices=Fed_Methods_NAMES)
    # FedRC FedAVG FedR FedProx FedDyn FedOpt FedProc FedR FedProxRC  FedProxCos
    '''
    Aggregations Strategy Hyper-Parameter
    '''
    parser.add_argument('--averaging', type=str, default='Weight', choices=Aggregation_NAMES, help='The Option for averaging strategy')
    parser.add_argument('--seed', type=int, default=0, help='The random seed.')

    # parser.add_argument('--csv_log', action='store_true', default=False, help='Enable csv logging')
    parser.add_argument('--csv_name', type=str, default='W0.01T1.0mu0.01lr0.01_1_0', help='Predefine the csv name')
    parser.add_argument('--commun_epoch', type=int, default=99, help='Communication epoch')

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

def main(args=None):
    if args is None:
        args = parse_args()

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    # 整合方法的超参数
    cfg_dataset_path = os.path.join(config_path(), args.dataset, 'Default.yaml')
    cfg.merge_from_file(cfg_dataset_path)

    cfg_method_path = os.path.join(config_path(), args.dataset, args.method + '.yaml')
    if os.path.exists(cfg_method_path):
        cfg.merge_from_file(cfg_method_path)

    cfg.merge_from_list(args.opts)

    particial_cfg = show_cfg(cfg, args.method)  # 移除其他方法的cfg

    if args.seed is not None:
        set_random_seed(args.seed)

    '''
    Loading the Private Digits
    '''
    private_dataset = get_prive_dataset(args, particial_cfg)
    '''
    Define clients domain
    '''
    in_domain_list = copy.deepcopy(private_dataset.domain_list)
    if args.OOD != "NONE":
        in_domain_list.remove(args.OOD)
        private_dataset.in_domain_list = in_domain_list

    private_dataset.in_domain_list = in_domain_list  # 参与者能够从哪几个Domain中获取数据

    # 先生成再删
    temp_client_domain_list = ini_client_domain(args.rand_domain_select, private_dataset.domain_list, particial_cfg.DATASET.parti_num)
    client_domain_list = []
    for i in range(len(temp_client_domain_list)):
        if temp_client_domain_list[i] != args.OOD:
            client_domain_list.append(temp_client_domain_list[i])

    # 只用改一次 因为不是deepcopy
    particial_cfg.DATASET.parti_num = len(client_domain_list)
    # particial_cfg.DATASET.n_classes = private_dataset.N_CLASS
    cfg.freeze()

    private_dataset.client_domain_list = client_domain_list  # 参与者具体的Domain选择
    private_dataset.get_data_loaders(client_domain_list)

    '''
    Loading the Private Backbone
    '''
    priv_backbones = get_private_backbones(particial_cfg)

    '''
    Loading the Federated Optimizer
    '''
    fed_method = get_fed_method(priv_backbones, client_domain_list, args, particial_cfg)

    '''
    Loading the Aggregations Solution
    '''
    fed_aggregation = get_fed_aggregation(args)

    # 绑定聚合策略进方法
    fed_method.fed_aggregation = fed_aggregation

    # 加权数据集分配给method
    fed_method.train_eval_loaders = private_dataset.train_eval_loaders


    print(log_msg('{}_{}_{}'.format(args.method, args.OOD, args.csv_name), "TRAIN"))

    if hasattr(fed_method, 'ini'):
        fed_method.ini()

    # 读取参数
    global_net_path = os.path.join(fed_method.net_folder, f'global_net_{fed_method.cfg.DATASET.backbone}_{args.commun_epoch}.pth')
    fed_method.global_net.load_state_dict(torch.load(global_net_path))

    if hasattr(fed_method, 'personalized_fc_list'):
        for i in range(len(fed_method.personalized_fc_list)):
            personalized_fc = fed_method.personalized_fc_list[i]
            personalized_fc_path = os.path.join(fed_method.net_folder, f'personalized_fc_{i}_{args.commun_epoch}.pth')
            personalized_fc.load_state_dict(torch.load(personalized_fc_path))
    if hasattr(fed_method, 'unbiased_fc'):
        unbiased_fc_path = os.path.join(fed_method.net_folder, f'unbiased_fc_{args.commun_epoch}.pth')
        fed_method.unbiased_fc.load_state_dict(torch.load(unbiased_fc_path))

    # 不同域给不同权重
    if hasattr(fed_method, 'global_net_dict'):
        if hasattr(fed_method, 'unbiased_fc'):
            for domain in fed_method.train_eval_loaders:
                temp_global_net = copy.deepcopy(fed_method.global_net)

                # 暂时等分
                freq_fc = np.ones(len(fed_method.personalized_fc_list)+1)/(len(fed_method.personalized_fc_list)+1)
                # freq_fc = [1,0,0,0]
                # [TEST] In Domain  Accs: [95.71, 82.29, 72.45] Method: FedProxCOSAddGlobal
                # [TEST] Out Domain MNIST Acc: 86.32 Method: FedProxCOSAddGlobal

                # freq_fc = [0,1,0,0]
                # [TRAIN] FedProxCOSAddGlobal_MNIST_W0.1T1.0mu0.01lr0.01_1_0
                # [TEST] In Domain  Accs: [90.88, 93.02, 81.25] Method: FedProxCOSAddGlobal
                # [TEST] Out Domain MNIST Acc: 73.55 Method: FedProxCOSAddGlobal

                # freq_fc = [0,0,1,0]
                # [TEST] In Domain  Accs: [89.54, 91.03, 84.95] Method: FedProxCOSAddGlobal
                # [TEST] Out Domain MNIST Acc: 72.11 Method: FedProxCOSAddGlobal

                freq_fc = [0,0,0,1]
                # [TEST] In Domain  Accs: [91.93, 91.59, 80.05] Method: FedProxCOSAddGlobal
                # [TEST] Out Domain MNIST Acc: 73.48 Method: FedProxCOSAddGlobal


                # freq_fc = [0.25,0.25,0.25,0.25]
                # [TEST] In Domain  Accs: [93.57, 92.51, 82.0] Method: FedProxCOSAddGlobal
                # [TEST] Out Domain MNIST Acc: 78.55 Method: FedProxCOSAddGlobal

                freq_fc = [0.41781646, 0.0918538, 0.39786425, 0.092465416]
                # [TRAIN] FedProxCOSAddGlobal_MNIST_W0.1T1.0mu0.01lr0.01_1_0
                # [TEST] In Domain  Accs: [95.27, 89.95, 78.6] Method: FedProxCOSAddGlobal
                # [TEST] Out Domain MNIST Acc: 83.99 Method: FedProxCOSAddGlobal

                # Default
                # [TRAIN] FedProxCOSAddGlobal_MNIST_W0.1T1.0mu0.01lr0.01_1_0
                # [TEST] In Domain  Accs: [91.48, 92.94, 82.1] Method: FedProxCOSAddGlobal
                # [TEST] Out Domain MNIST Acc: 75.13 Method: FedProxCOSAddGlobal

                fed_method.fed_aggregation.agg_parts(online_clients_list=list(range(cfg.DATASET.parti_num)),
                                                     nets_list=fed_method.personalized_fc_list,
                                                     global_net=fed_method.personalized_global_fc, freq=freq_fc[:-1], except_part=[],
                                                     global_only=True,
                                                     use_additional_net=True,additional_net_list=[fed_method.unbiased_fc],
                                                     additional_freq=[freq_fc[-1]])
                temp_global_net.cls = copy.deepcopy(fed_method.personalized_global_fc)
                # temp_global_net.cls = copy.deepcopy(fed_method.global_net.cls)
                fed_method.global_net_dict[domain] = temp_global_net

        else:
            for domain in fed_method.train_eval_loaders:
                temp_global_net = copy.deepcopy(fed_method.global_net)

                # 暂时等分
                freq_fc = np.ones(len(fed_method.personalized_fc_list)) / (len(fed_method.personalized_fc_list))

                fed_method.fed_aggregation.agg_parts(online_clients_list=list(range(cfg.DATASET.parti_num)),
                                                     nets_list=fed_method.personalized_fc_list,
                                                     global_net=fed_method.personalized_global_fc, freq=freq_fc, except_part=[],
                                                     global_only=True)
                temp_global_net.cls = copy.deepcopy(fed_method.personalized_global_fc)
                fed_method.global_net_dict[domain] = temp_global_net

    in_domain_accs, mean_in_domain_acc = global_in_evaluation(fed_method, private_dataset.test_loader, private_dataset.in_domain_list)
    # print(log_msg(f"In Domain Mean Acc: {mean_in_domain_acc} Method: {args.method}", "TEST"))
    print(log_msg(f"In Domain  Accs: {in_domain_accs} Method: {args.method}", "TEST"))

    '''
    全局模型在未知的Domain上的精度 & 存储
    '''
    if args.OOD != "NONE":
        out_domain_acc = global_out_evaluation(fed_method, private_dataset.test_loader, args.OOD)

        print(log_msg(f"Out Domain {args.OOD} Acc: {out_domain_acc} Method: {args.method}", "TEST"))


if __name__ == '__main__':
    main()
