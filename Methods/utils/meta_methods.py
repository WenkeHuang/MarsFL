from Aggregations import get_fed_aggregation
from Global import get_global_method
from Local import get_local_method
from utils.conf import get_device, checkpoint_path, net_path
from utils.utils import create_if_not_exists
from argparse import Namespace
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
        self.nets_list = nets_list  # 存储本地网络参数
        self.client_domain_list = client_domain_list  # 存储Domain-用于个性化性能测试

        self.args = args
        self.cfg = cfg

        # 计算一次参与的客户端数量
        self.random_state = np.random.RandomState()
        self.online_num = np.ceil(self.cfg.DATASET.parti_num * self.cfg.DATASET.online_ratio).item()
        self.online_num = int(self.online_num)

        self.global_net = None
        self.device = get_device(device_id=self.args.device_id)

        # 本地和全局模型更新
        self.local_model = get_local_method(args, cfg)
        self.global_model = get_global_method(args, cfg)

        # 用于同构聚合的策略
        if args.structure == 'homogeneity':
            self.fed_aggregation = get_fed_aggregation(args)
        else:
            self.fed_aggregation = None

        self.epoch_index = 0

        # 模型路径

        self.base_net_folder = os.path.join(net_path(), self.args.dataset, self.args.OOD,
                                            self.args.averaging, self.args.method)
        create_if_not_exists(self.base_net_folder)
        para_group_dirs = os.listdir(self.base_net_folder)  # 获取所有存储的参数组
        n_para = len(para_group_dirs)  # 获取长度
        if self.args.csv_name == None:
            path = os.path.join(self.base_net_folder, 'para' + str(n_para + 1))
            k = 1
            while os.path.exists(path):
                path = os.path.join(self.base_net_folder, 'para' + str(n_para + k))
                k = k + 1
        else:
            path = os.path.join(self.base_net_folder, self.args.csv_name)

        self.net_folder = path
        create_if_not_exists(self.net_folder)
        self.net_to_device()

    def net_to_device(self):
        for net in self.nets_list:
            net.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_scheduler(self):
        return

    def ini(self):
        pass

    def col_update(self, publoader):
        pass

    def save_checkpoint(self):
        global_net_path = os.path.join(self.net_folder, f'global_net_{self.cfg.DATASET.backbone}_{self.epoch_index}.pth')
        torch.save(self.global_net.state_dict(), global_net_path)
        print('save global_net over')

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
