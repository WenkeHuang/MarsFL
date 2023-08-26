import torch
import torchvision.models
import torch.nn.functional as F
from torch import distributions

from Methods.utils.meta_methods import FederatedMethod
from utils.utils import cal_client_weight
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy


class FedSR(FederatedMethod):
    NAME = 'FedSR'

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(FedSR, self).__init__(nets_list, client_domain_list, args, cfg)
        self.r_mu = nn.Parameter(torch.zeros(self.cfg.DATASET.parti_num, self.nets_list[0].cls.out_features, self.nets_list[0].cls.in_features))
        self.r_sigma = nn.Parameter(torch.ones(self.cfg.DATASET.parti_num, self.nets_list[0].cls.out_features, self.nets_list[0].cls.in_features))
        self.C = nn.Parameter(torch.ones(self.cfg.DATASET.parti_num))

        self.L2R_coeff = cfg[self.NAME].L2R_coeff
        self.CMI_coeff = cfg[self.NAME].CMI_coeff

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
            optimizer.add_param_group({'params': [self.r_mu, self.r_sigma, self.C], 'lr': self.cfg.OPTIMIZER.local_train_lr, 'momentum': self.cfg.OPTIMIZER.momentum})



        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                client_r_mu = self.r_mu[index].to(self.device)
                client_r_sigma = self.r_sigma[index].to(self.device)
                client_C = self.C[index].to(self.device)

                images = images.to(self.device)
                labels = labels.to(self.device)
                z, (z_mu, z_sigma) = net.featurize(images, return_dist=True)

                outputs = net.classifier(z)
                loss = criterion(outputs, labels)

                # regL2R = torch.zeros_like(loss)
                # regCMI = torch.zeros_like(loss)
                # regNegEnt = torch.zeros_like(loss)

                if self.L2R_coeff != 0.0:
                    regL2R = z.norm(dim=1).mean()
                    loss = loss + self.L2R_coeff * regL2R
                if self.CMI_coeff != 0.0:
                    r_sigma_softplus = F.softplus(client_r_sigma)
                    r_mu = client_r_mu[labels]
                    r_sigma = r_sigma_softplus[labels]
                    z_mu_scaled = z_mu * client_C
                    z_sigma_scaled = z_sigma * client_C
                    regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + \
                             (z_sigma_scaled ** 2 + (z_mu_scaled - r_mu) ** 2) / (2 * r_sigma ** 2) - 0.5
                    regCMI = regCMI.sum(1).mean()
                    loss = loss + self.CMI_coeff * regCMI

                # z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
                # mix_coeff = distributions.categorical.Categorical(x.new_ones(x.shape[0]))
                # mixture = distributions.mixture_same_family.MixtureSameFamily(mix_coeff, z_dist)
                # log_prob = mixture.log_prob(z)
                # regNegEnt = log_prob.mean()

                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
