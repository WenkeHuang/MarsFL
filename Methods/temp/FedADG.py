from collections import OrderedDict

import torch
from torch.nn import init

from Methods.utils.meta_methods import FederatedMethod
from utils.utils import cal_client_weight, set_requires_grad
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, hidden_size, num_labels, rp_size):
        super(Discriminator, self).__init__()
        self.features_pro = nn.Sequential(
            nn.Linear(rp_size, rp_size),
            nn.LeakyReLU(),
            nn.Linear(rp_size, 1),
            nn.Sigmoid(),
        )
        self.optimizer = None
        self.projection = nn.Linear(hidden_size + num_labels, rp_size, bias=False)
        with torch.no_grad():
            self.projection.weight.div_(torch.norm(self.projection.weight, keepdim=True))

    def forward(self, y, z):
        feature = z.view(z.size(0), -1)
        feature = torch.cat([feature, y], dim=1)
        feature = self.projection(feature)
        logit = self.features_pro(feature)
        return logit


# Distribution Generator
class GeneDistrNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels):
        super(GeneDistrNet, self).__init__()
        self.num_labels = num_labels
        self.latent_size = 4096
        self.genedistri = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(input_size + self.num_labels, self.latent_size)),
            ("relu1", nn.LeakyReLU()),

            ("fc2", nn.Linear(self.latent_size, hidden_size)),
            ("relu2", nn.ReLU()),
        ]))

        self.initial_params()

    def initial_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                init.xavier_uniform_(layer.weight, 0.5)

    def forward(self, x, y, device):
        x = torch.cat([x, y], dim=1)
        x = x.to(device)
        x = self.genedistri(x)
        return x


class FedADG(FederatedMethod):
    NAME = 'FedADG'

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(FedADG, self).__init__(nets_list, client_domain_list, args, cfg)
        self.pretrain_epoch = cfg[self.NAME].pretrain_epoch
        self.train_epoch = cfg[self.NAME].train_epoch
        self.alpha = self.cfg[self.NAME].alpha

        rp_size = cfg[self.NAME].rp_size

        hidden_size = self.nets_list[0].cls.in_features
        num_labels = self.nets_list[0].cls.out_features

        self.global_DG = GeneDistrNet(self.nets_list[0].cls.in_features, hidden_size, num_labels=num_labels).to(self.device)

        self.global_D = Discriminator(hidden_size, num_labels, rp_size).to(self.device)

        self.DG_list = []
        self.D_list = []

        for i in range(len(self.nets_list)):
            self.DG_list.append(copy.deepcopy(self.global_DG))
            self.D_list.append(copy.deepcopy(self.global_D))

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

        # Bone + cls
        self.fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=self.nets_list,
                                       global_net=self.global_net, freq=freq, except_part=[], global_only=False)
        # Distribution Generator
        self.fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=self.DG_list,
                                       global_net=self.global_DG, freq=freq, except_part=[], global_only=False)
        return None

    def train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)

        # 先训backbone和cls一段时间 再训练gan
        set_requires_grad(net, True)
        iterator = tqdm(range(self.pretrain_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()

        # 联合训
        iterator = tqdm(range(self.train_epoch))
        DG_net = self.DG_list[index]
        optimizer_DG = optim.SGD(DG_net.parameters(), lr=self.cfg[self.NAME].DG_train_lr,
                                 momentum=self.cfg[self.NAME].DG_momentum, weight_decay=self.cfg[self.NAME].DG_weight_decay)
        discriminator = self.D_list[index]
        optimizer_D = optim.SGD(discriminator.parameters(), lr=self.cfg[self.NAME].D_train_lr,
                                momentum=self.cfg[self.NAME].D_momentum, weight_decay=self.cfg[self.NAME].D_weight_decay)
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                y_onehot = torch.zeros(labels.size(0), net.cls.out_features).to(self.device)
                y_onehot.scatter_(1, labels.view(-1, 1), 1).to(self.device)

                # 随机采样作为dg的输入
                random_f = torch.rand(labels.size(0), net.cls.in_features).to(self.device)

                # 训练判别器
                set_requires_grad(net, False)
                set_requires_grad(DG_net, False)
                set_requires_grad(discriminator, True)
                optimizer_D.zero_grad()

                net.eval()
                DG_net.eval()

                fakez = net.features(images)
                realz = DG_net(y=y_onehot, x=random_f, device=self.device)
                net.train()
                DG_net.train()

                loss_discri = -torch.mean(torch.pow(discriminator(y_onehot, realz), 2) + torch.pow(1 - discriminator(y_onehot, fakez), 2))
                loss_discri.backward()
                optimizer_D.step()

                # 训练net
                set_requires_grad(net, True)
                set_requires_grad(DG_net, False)
                set_requires_grad(discriminator, False)

                discriminator.eval()
                f = net.features(images)
                outputs = net.classifier(f)
                loss_ce = criterion(outputs, labels)
                loss_enc = torch.mean(torch.pow(1 - discriminator(y_onehot, f), 2))
                loss_cla = self.alpha * loss_ce + (1 - self.alpha) * loss_enc
                optimizer.zero_grad()
                loss_cla.backward()
                optimizer.step()

                # 训练DGnet
                set_requires_grad(net, False)
                set_requires_grad(DG_net, True)
                set_requires_grad(discriminator, False)

                realz = DG_net(y=y_onehot, x=random_f, device=self.device)
                loss_gene = torch.mean(torch.pow(1 - discriminator(y_onehot, realz), 2))
                optimizer_DG.zero_grad()
                loss_gene.backward()
                optimizer_DG.step()

                discriminator.train()
                iterator.desc = "Local %d loss d = %0.3f loss local = %0.3f loss g = %0.3f" % (index, loss_discri, loss_cla, loss_gene)
