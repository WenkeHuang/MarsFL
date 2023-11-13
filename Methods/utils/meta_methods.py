import copy

from Aggregations import get_fed_aggregation
from Sever import get_sever_method
from Local import get_local_method
from utils.conf import get_device
import torch.nn as nn
import numpy as np
import torch
import os


class FederatedMethod(nn.Module):
    """
    Federated learning Methods.
    """
    NAME = None

    def __init__(self, nets_list: list, client_domain_list: list,
                 args, cfg) -> None:
        super(FederatedMethod, self).__init__()
        self.nets_list = nets_list
        self.client_domain_list = client_domain_list

        self.args = args
        self.cfg = cfg

        self.random_state = np.random.RandomState()
        self.online_num = np.ceil(self.cfg.DATASET.parti_num * self.cfg.DATASET.online_ratio).item()
        self.online_num = int(self.online_num)

        self.global_net = None
        self.device = get_device(device_id=self.args.device_id)

        self.local_model = get_local_method(args, cfg)
        self.sever_model = get_sever_method(args, cfg)

        if args.structure == 'homogeneity':
            self.fed_aggregation = get_fed_aggregation(args)
        else:
            self.fed_aggregation = None

        self.epoch_index = 0
        self.random_net = copy.deepcopy(self.nets_list[0]).to(self.device)

        self.net_to_device()

    def net_to_device(self):
        for net in self.nets_list:
            net.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_scheduler(self):
        return

    def ini(self):
        if self.args.structure == 'homogeneity':
            self.global_net = copy.deepcopy(self.nets_list[0])
            self.global_net = self.global_net.to(self.device)

            global_w = self.nets_list[0].state_dict()
            for _, net in enumerate(self.nets_list):
                net.load_state_dict(global_w)

    def col_update(self, publoader):
        pass

    def update(self, priloader_list):
        pass

    def get_gt_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    def load_pretrained_nets(self):
        if self.load:
            for j in range(self.args.parti_num):
                pretrain_path = os.path.join(self.checkpoint_path, 'pretrain')
                save_path = os.path.join(pretrain_path, str(j) + '.ckpt')
                self.nets_list[j].load_state_dict(torch.load(save_path, self.device))
        else:
            pass

    def copy_nets2_prevnets(self):
        nets_list = self.nets_list
        for net_id, net in enumerate(nets_list):
            net_para = net.state_dict()
            self.prev_nets_list[net_id].load_state_dict(net_para)
