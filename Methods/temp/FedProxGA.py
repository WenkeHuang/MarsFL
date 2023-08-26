import numpy as np

from Methods.utils.meta_methods import FederatedMethod
from utils.utils import cal_client_weight
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
import copy


class FedProxGA(FederatedMethod):
    NAME = 'FedProxGA'

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(FedProxGA, self).__init__(nets_list, client_domain_list, args, cfg)
        self.mu = cfg[self.NAME].mu
        self.agg_weight = np.ones(self.cfg.DATASET.parti_num) / self.cfg.DATASET.parti_num

        self.base_step_size = cfg[self.NAME].base_step_size
        self.step_size_decay = self.base_step_size / self.cfg.DATASET.communication_epoch

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        self.global_net = self.global_net.to(self.device)

        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def update(self, priloader_list):
        total_clients = list(range(self.cfg.DATASET.parti_num))  # 获取所有参与者
        online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()  # 随机选取online的参与者

        accs_before_agg = []
        for i in online_clients_list:  # 遍历循环当前的参与者
            self.train_net(i, self.nets_list[i], priloader_list[i])
            acc_before_agg = self.get_local_test_acc(self.nets_list[i], self.train_eval_loaders[self.client_domain_list[i]])
            accs_before_agg.append(acc_before_agg)

        # FedAVG 是聚合除了Lora的部分 即 Bone + cls
        self.fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=self.nets_list,
                                       global_net=self.global_net, freq=self.agg_weight, except_part=['lora_'], global_only=False)
        accs_after_agg = []
        for i in online_clients_list:
            acc_after_agg = self.get_local_test_acc(self.global_net, self.test_loaders[self.client_domain_list[i]])
            accs_after_agg.append(acc_after_agg)

        self.update_weight_by_GA(accs_before_agg, accs_after_agg)

        return None

    # 用ga更新权重
    def update_weight_by_GA(self, accs_before_agg, accs_after_agg):
        accs_before_agg = np.array(accs_before_agg)
        accs_after_agg = np.array(accs_after_agg)

        # 准确度差值
        accs_diff = accs_after_agg - accs_before_agg

        # 根据等权重修为基础做修正
        step_size = self.base_step_size - (self.epoch_index - 1) * self.step_size_decay
        step_size *= np.ones(self.online_num) / self.online_num

        norm_gap_array = accs_diff / np.max(np.abs(accs_diff))

        # 更新
        self.agg_weight += norm_gap_array * step_size
        self.agg_weight = np.clip(self.agg_weight, 0, 1)

        self.agg_weight = self.agg_weight / np.sum(self.agg_weight)

        return

    # 测自己在自己域上的效果
    def get_local_test_acc(self, net, test_loader):
        total = 0
        top1 = 0
        net.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images[0].to(self.device), labels.to(self.device)
                outputs = net(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()

                total += labels.size(0)
        top1acc = round(100 * top1 / total, 2)
        net.train()
        return top1acc

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
                iterator.desc = "Local Client %d loss = %0.3f" % (index, loss)
                optimizer.step()
