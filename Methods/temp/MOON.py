from Methods.utils.meta_methods import FederatedMethod
from utils.utils import cal_client_weight
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
import copy


class MOON(FederatedMethod):
    NAME = 'MOON'

    def __init__(self, nets_list, client_domain_list, args,cfg):
        super(MOON, self).__init__(nets_list, client_domain_list,args,cfg)
        self.prev_nets_list = []
        self.mu = cfg[self.NAME].mu
        self.temperature_moon = cfg[self.NAME].temperature_moon

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        self.global_net = self.global_net.to(self.device)

        global_w = self.nets_list[0].state_dict()

        for _, net in enumerate(self.nets_list):
            self.prev_nets_list.append(copy.deepcopy(net)) # 初始化在loading global之前
            net.load_state_dict(global_w)

    def update(self, priloader_list):
        total_clients = list(range(self.cfg.DATASET.parti_num))  # 获取所有参与者
        online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()  # 随机选取online的参与者

        for i in online_clients_list:  # 遍历循环当前的参与者
            self.train_net(i, self.nets_list[i],self.prev_nets_list[i], priloader_list[i])

        self.copy_nets2_prevnets()

        # 获取参与者的聚合权重
        freq = self.fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        client_weight = cal_client_weight(online_clients_list=online_clients_list, client_domain_list=self.client_domain_list, freq=freq)
        print(client_weight)

        '''这里我建议聚合和分发放到一起 因为有的聚合是只聚合部分 然后分发部分'''
        # FedAVG 是聚合除了Lora的部分 即 Bone + cls
        self.fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=self.nets_list,
                                       global_net=self.global_net, freq=freq, except_part=['lora_'],global_only=False)
        return None

    def train_net(self, index, net,prev_net, train_loader):
        net = net.to(self.device)
        prev_net = prev_net.to(self.device)
        if self.cfg.OPTIMIZER.type =='SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        self.global_net = self.global_net.to(self.device)
        cos = torch.nn.CosineSimilarity(dim=-1)
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                f = net.features(images)
                pre_f = prev_net.features(images)
                g_f = self.global_net.features(images)
                posi = cos(f, g_f)
                temp = posi.reshape(-1, 1)
                nega = cos(f, pre_f)
                temp = torch.cat((temp, nega.reshape(-1, 1)), dim=1)
                temp /= self.temperature_moon
                temp = temp.to(self.device)
                targets = torch.zeros(labels.size(0)).to(self.device).long()
                lossCON = self.mu * criterion(temp, targets)
                outputs = net(images)
                lossCE = criterion(outputs, labels)
                loss = lossCE + lossCON
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d CE = %0.3f,CON = %0.3f" % (index, lossCE, lossCON)
                optimizer.step()

