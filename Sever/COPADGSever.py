import copy

from Sever.utils.sever_methods import SeverMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
import numpy as np
from itertools import permutations, combinations

from utils.utils import set_requires_grad


class COPADGSever(SeverMethod):
    NAME = 'COPADGSever'
    def __init__(self, args, cfg):
        super(COPADGSever, self).__init__(args, cfg)

    def sever_update(self, **kwargs):
        fed_aggregation = kwargs['fed_aggregation']
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        # client_domain_list = kwargs['client_domain_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']

        # 更新客户端的头
        head_dict = {}
        for i in range(len(nets_list)):
            head_dict[i] = {}
            for j in range(len(nets_list)):
                # 加入其他头 并且不要梯度
                if i != j:
                    head = copy.deepcopy(nets_list[j].cls)
                    set_requires_grad(head, False)
                    head_dict[i][j] = head

        # 获取参与者的聚合权重
        freq = fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        # FedAVG 是聚合Bone + cls
        fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                                  global_net=global_net, freq=freq, except_part=['cls'], global_only=False)

        return freq, head_dict

