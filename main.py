from Aggregations import Aggregation_NAMES
from Datasets.federated_dataset.single_domain import get_single_domain_dataset
from Methods import Fed_Methods_NAMES, get_fed_method
from utils.conf import set_random_seed, config_path
from Datasets.federated_dataset.multi_domain import Priv_NAMES, get_multi_domain_dataset
from Backbones import get_private_backbones
from utils.cfg import CFG as cfg, show_cfg
from utils.utils import ini_client_domain
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
    parser.add_argument('--device_id', type=int, default=7, help='The Device Id for Experiment')
    parser.add_argument('--dataset', type=str, default='fl_cifar10',  # Digits,PACS PACScomb OfficeHome
                        help='Which scenario to perform experiments on.')
    parser.add_argument('--rand_domain_select', type=bool, default=True, help='The Local Domain Selection')

    parser.add_argument('--task', type=str, default='label_skew')
    parser.add_argument('--structure', type=str, default='homogeneity')  # 'homogeneity' heterogeneity

    '''
    Whether Conduct OOD Experiments NONE
    '''
    # NONE
    # Digits: MNIST, USPS, SVHN, SYN
    # PACS: 'photo', 'art_painting', 'cartoon', 'sketch'
    # OfficeCaltech 'caltech', 'amazon','webcam','dslr'
    # OfficeHome 'Art', 'Clipart', 'Product', 'Real_World'
    # DomainNet 'clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'
    parser.add_argument('--OOD', type=str, default='MNIST', help='Whether conduct OOD Experiments')

    '''
    Federated Optimizer Hyper-Parameter 
    '''
    parser.add_argument('--method', type=str, default='FedProx',
                        help='Federated Method name.', choices=Fed_Methods_NAMES)
    # FedRC FedAVG FedR FedProx FedDyn FedOpt FedProc FedR FedProxRC  FedProxCos
    '''
    Aggregations Strategy Hyper-Parameter
    '''
    parser.add_argument('--averaging', type=str, default='Weight', choices=Aggregation_NAMES, help='The Option for averaging strategy')
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

    particial_cfg = show_cfg(cfg, args.method)  # 移除其他方法的cfg

    if args.seed is not None:
        set_random_seed(args.seed)

    '''
    Loading the Private Digits
    '''

    if args.task == 'OOD':
        '''
        Define clients domain
        '''
        private_dataset = get_multi_domain_dataset(args, particial_cfg)
        in_domain_list = copy.deepcopy(private_dataset.domain_list)
        if args.OOD != "NONE":
            in_domain_list.remove(args.OOD)
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
            if temp_client_domain_list[i] != args.OOD:
                client_domain_list.append(temp_client_domain_list[i])

        # 只用改一次 因为不是deepcopy
        particial_cfg.DATASET.parti_num = len(client_domain_list)

        cfg.freeze()

        private_dataset.client_domain_list = client_domain_list  # 参与者具体的Domain选择
        private_dataset.get_data_loaders(client_domain_list)

    elif args.task == 'label_skew':
        private_dataset = get_single_domain_dataset(args, particial_cfg)
        private_dataset.get_data_loaders()
        client_domain_list = None
    elif args.task == 'domain_skew':
        private_dataset = get_multi_domain_dataset(args, particial_cfg)
        client_domain_list = ini_client_domain(args.rand_domain_select, private_dataset.domain_list, particial_cfg.DATASET.parti_num)
        private_dataset.get_data_loaders(client_domain_list)

    '''
    Loading the Private Backbone
    '''
    priv_backbones = get_private_backbones(particial_cfg)

    '''
    Loading the Federated Optimizer
    '''

    fed_method = get_fed_method(priv_backbones, client_domain_list, args, particial_cfg)
    assert args.structure in fed_method.COMPATIBILITY

    if args.task == 'OOD':
        # 加权数据集分配给method
        fed_method.train_eval_loaders = private_dataset.train_eval_loaders

    if args.csv_name == None:
        setproctitle.setproctitle('{}_{}'.format(args.method, args.OOD))
    else:
        setproctitle.setproctitle('{}_{}_{}'.format(args.method, args.OOD, args.csv_name))
    train(fed_method, private_dataset, args, particial_cfg, client_domain_list)


if __name__ == '__main__':
    main()
