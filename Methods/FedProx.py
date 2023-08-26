from Methods.utils.meta_methods import FederatedMethod
from utils.utils import cal_client_weight
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
import copy


class FedProx(FederatedMethod):
    NAME = 'FedProx'

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(FedProx, self).__init__(nets_list, client_domain_list, args, cfg)
        self.mu = cfg[self.NAME].mu

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        self.global_net = self.global_net.to(self.device)

        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.cfg.DATASET.parti_num))  # 获取所有参与者
        online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()  # 随机选取online的参与者

        for i in online_clients_list:  # 遍历循环当前的参与者
            self.train_net(i, self.nets_list[i], priloader_list[i])

        # 获取参与者的聚合权重
        freq = self.fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        client_weight = cal_client_weight(online_clients_list=online_clients_list, client_domain_list=self.client_domain_list, freq=freq)
        print(client_weight)

        # 聚合Bone + cls
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
        self.global_net = self.global_net.to(self.device)
        global_weight_collector = list(self.global_net.parameters())
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                fed_prox_reg = 0.0
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += ((0.01 / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += self.mu * fed_prox_reg
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
