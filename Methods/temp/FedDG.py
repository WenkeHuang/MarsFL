from collections import OrderedDict

import torch

from Methods.utils.meta_methods import FederatedMethod
from utils.utils import cal_client_weight
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy


class FedDG(FederatedMethod):
    NAME = 'FedDG'

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(FedDG, self).__init__(nets_list, client_domain_list, args, cfg)

        self.meta_step_size = cfg[self.NAME].meta_step_size
        self.clip_value = cfg[self.NAME].clip_value

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        self.global_net = self.global_net.to(self.device)

        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def update(self, priloader_list):
        total_clients = list(range(self.cfg.DATASET.parti_num))  # 获取所有参与者
        online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()  # 随机选取online的参与者

        for i in online_clients_list:  # 遍历循环当前的参与者
            self.train_net(i, self.nets_list[i], priloader_list[i])

        # 获取参与者的聚合权重
        freq = self.fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        client_weight = cal_client_weight(online_clients_list=online_clients_list, client_domain_list=self.client_domain_list, freq=freq)
        print(client_weight)

        '''这里我建议聚合和分发放到一起 因为有的聚合是只聚合部分 然后分发部分'''
        # FedAVG 是聚合除了Lora的部分 即 Bone + cls
        self.fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=self.nets_list,
                                       global_net=self.global_net, freq=freq, except_part=['lora_'], global_only=False)
        return None

    def train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        for _ in iterator:
            for batch_idx, (images, image_tar_freq, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                image_tar_freq = image_tar_freq.to(self.device)

                output_inner = net(images)
                loss_inner = criterion(output_inner, labels)
                grads = torch.autograd.grad(loss_inner, net.parameters(), retain_graph=True, allow_unused=True)
                meta_step_size = self.meta_step_size
                clip_value = self.clip_value
                fast_weights = OrderedDict((name, param - torch.mul(meta_step_size, torch.clamp(grad, 0 - clip_value, clip_value))) for
                                           ((name, param), grad) in zip(net.named_parameters(), grads) if grad is not None)

                learner = copy.deepcopy(net)
                learner.load_state_dict(fast_weights, strict=False)

                output_outer = learner(image_tar_freq)
                del fast_weights
                loss_outer = criterion(output_outer, labels)

                loss = loss_inner + loss_outer

                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
