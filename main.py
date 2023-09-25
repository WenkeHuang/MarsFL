import numpy as np
import torch
from torch.utils.data import DataLoader
from Aggregations import Aggregation_NAMES
from Attack.backdoor.utils import BackdoorDataset, backdoor_attack
from Attack.byzantine.utils import attack_dataset
from Datasets.federated_dataset.single_domain import single_domain_dataset_name, get_single_domain_dataset
from Methods import Fed_Methods_NAMES, get_fed_method
from utils.conf import set_random_seed, config_path
from Datasets.federated_dataset.multi_domain import multi_domain_dataset_name, get_multi_domain_dataset
from Backbones import get_private_backbones
from utils.cfg import CFG as cfg, simplify_cfg,show_cfg
from utils.utils import ini_client_domain, log_msg
from argparse import ArgumentParser
from utils.training import train
import setproctitle
import argparse
import datetime
import socket
import uuid
import copy
import os


def parse_args():
    parser = ArgumentParser(description='Federated Learning', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')

    parser.add_argument('--task', type=str, default='domain_skew')
    # OOD label_skew domain_skew
    parser.add_argument('--dataset', type=str, default='OfficeCaltech',
                        help='Which scenario to perform experiments on.')
    # fl_cifar10 fl_cifar100 fl_mnist fl_fashionmnist fl_tinyimagenet
    # Digits,OfficeCaltech, PACS PACScomb OfficeHome
    parser.add_argument('--attack_type', type=str, default='None')
    # byzantine backdoor None

    parser.add_argument('--rand_domain_select', type=bool, default=True, help='The Local Domain Selection')
    parser.add_argument('--structure', type=str, default='homogeneity')  # 'homogeneity' heterogeneity

    '''
    Federated Optimizer Hyper-Parameter 
    '''

    parser.add_argument('--method', type=str, default='FedAVG',
                        help='Federated Method name.', choices=Fed_Methods_NAMES)
    # FedRC FedAVG FedR FedProx FedDyn FedOpt FedProc FedR FedProxRC  FedProxCos FedNTD
    '''
    Aggregations Strategy Hyper-Parameter
    '''
    parser.add_argument('--averaging', type=str, default='Weight', choices=Aggregation_NAMES, help='The Option for averaging strategy')
    # Weight Equal

    parser.add_argument('--seed', type=int, default=0, help='The random seed.')
    # parser.add_argument('--note', type=str,default='DKDWeight', help='Something extra')

    parser.add_argument('--csv_log', action='store_true', default=False, help='Enable csv logging')
    parser.add_argument('--csv_name', type=str, default=None, help='Predefine the csv name')
    parser.add_argument('--save_checkpoint', action='store_true', default=False)
    # parser.add_argument('--use_random_domain', action='store_true',default=False)

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
    cfg_dataset_path = os.path.join(config_path(), args.task, args.dataset, 'Default.yaml')
    cfg.merge_from_file(cfg_dataset_path)

    cfg_method_path = os.path.join(config_path(), args.dataset, args.method + '.yaml')
    if os.path.exists(cfg_method_path):
        cfg.merge_from_file(cfg_method_path)

    cfg.merge_from_list(args.opts)

    particial_cfg = simplify_cfg(args, cfg)

    show_cfg(particial_cfg,args.method)
    if args.seed is not None:
        set_random_seed(args.seed)

    '''
    Loading the dataset
    '''
    if args.dataset in multi_domain_dataset_name:
        private_dataset = get_multi_domain_dataset(args, particial_cfg)
    elif args.dataset in single_domain_dataset_name:
        private_dataset = get_single_domain_dataset(args, particial_cfg)

    if args.task == 'OOD':
        '''
        Define clients domain
        '''

        in_domain_list = copy.deepcopy(private_dataset.domain_list)
        if cfg[args.task].out_domain != "NONE":
            in_domain_list.remove(cfg[args.task].out_domain)
            private_dataset.in_domain_list = in_domain_list

        private_dataset.in_domain_list = in_domain_list  # 参与者能够从哪几个Domain中获取数据

        # 先生成再删
        # 随机采样
        temp_client_domain_list = ini_client_domain(args.rand_domain_select, private_dataset.domain_list, particial_cfg.DATASET.parti_num)

        # 均分
        # temp_client_domain_list = copy.deepcopy(private_dataset.domain_list) * (particial_cfg.DATASET.parti_num // len(private_dataset.domain_list))

        # 是否用随机 如果不随机 那么均分 注意整除
        # if args.use_random_domain:
        #     temp_client_domain_list = ini_client_domain(args.rand_domain_select, private_dataset.domain_list, particial_cfg.DATASET.parti_num)
        # else:
        #     temp_client_domain_list = copy.deepcopy(private_dataset.domain_list) * (particial_cfg.DATASET.parti_num//len(private_dataset.domain_list))

        client_domain_list = []
        for i in range(len(temp_client_domain_list)):
            if temp_client_domain_list[i] != cfg[args.task].out_domain:
                client_domain_list.append(temp_client_domain_list[i])

        # 只用改一次 因为不是deepcopy
        particial_cfg.DATASET.parti_num = len(client_domain_list)

        # cfg.freeze()

        private_dataset.client_domain_list = client_domain_list
        client_domain_list.append(cfg[args.task].out_domain)
        # 参与者具体的Domain选择
        private_dataset.get_data_loaders(client_domain_list)
        private_dataset.out_train_loader = private_dataset.train_loaders.pop()
        client_domain_list.pop()

    elif args.task == 'label_skew':
        private_dataset.get_data_loaders()
        client_domain_list = None

    elif args.task == 'domain_skew':
        client_domain_list = ini_client_domain(args.rand_domain_select, private_dataset.domain_list, particial_cfg.DATASET.parti_num)
        private_dataset.get_data_loaders(client_domain_list)

    if args.attack_type == 'byzantine':

        # 数据集的信息
        if args.dataset in multi_domain_dataset_name:
            # client_domain_list = ini_client_domain(args.rand_domain_select, private_dataset.domain_list, particial_cfg.DATASET.parti_num)
            # private_dataset.get_data_loaders(client_domain_list)
            particial_cfg.attack.dataset_type = 'multi_domain'

        elif args.dataset in single_domain_dataset_name:
            # private_dataset.get_data_loaders()
            # client_domain_list = None
            particial_cfg.attack.dataset_type = 'single_domain'

        # 攻击和未被攻击的客户端数量
        bad_scale = int(particial_cfg.DATASET.parti_num * particial_cfg['attack'].bad_client_rate)
        good_scale = particial_cfg.DATASET.parti_num - bad_scale
        client_type = np.repeat(True, good_scale).tolist() + (np.repeat(False, bad_scale)).tolist()
        # 攻击类型是数据集攻击 那么修改数据集的内容
        attack_dataset(args, cfg, private_dataset, client_type)
    elif args.attack_type == 'backdoor':
        # 攻击和未被攻击的客户端数量
        bad_scale = int(particial_cfg.DATASET.parti_num * particial_cfg['attack'].bad_client_rate)
        good_scale = particial_cfg.DATASET.parti_num - bad_scale
        client_type = np.repeat(True, good_scale).tolist() + (np.repeat(False, bad_scale)).tolist()

        # 攻击训练集
        backdoor_attack(args, particial_cfg, client_type, private_dataset, is_train=True)

        # 攻击测试集用于测试专属指标
        backdoor_attack(args, particial_cfg, client_type, private_dataset, is_train=False)

    '''
    Loading the Private Backbone
    '''
    priv_backbones = get_private_backbones(particial_cfg)

    '''
    Loading the Federated Optimizer
    '''

    fed_method = get_fed_method(priv_backbones, client_domain_list, args, particial_cfg)
    assert args.structure in fed_method.COMPATIBILITY

    # 将部分参数给方法
    if args.task == 'OOD':
        # 加权数据集分配给method
        fed_method.train_eval_loaders = private_dataset.train_eval_loaders

        fed_method.test_loaders = private_dataset.test_loader

    if args.attack_type == 'byzantine':
        fed_method.client_type = client_type

    # print(log_msg("CONFIG:\n{}".format(particial_cfg.dump()), "INFO"))
    if args.csv_name is None:
        setproctitle.setproctitle('{}_{}'.format(args.method, args.task))
    else:
        setproctitle.setproctitle('{}_{}_{}'.format(args.method, args.task, args.csv_name))
    train(fed_method, private_dataset, args, particial_cfg, client_domain_list)


if __name__ == '__main__':
    main()
