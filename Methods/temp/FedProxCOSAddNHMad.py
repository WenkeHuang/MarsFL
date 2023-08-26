import os

import numpy as np

from Methods.utils.meta_methods import FederatedMethod
from functorch import make_functional
import torch.nn.functional as F
import torch.optim as optim
from utils.utils import EH
from tqdm import tqdm
import torch.nn as nn
import torch
import copy


class FedProxCOSAddNHMad(FederatedMethod):
    NAME = 'FedProxCOSAddNHMad'

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(FedProxCOSAddNHMad, self).__init__(nets_list, client_domain_list, args, cfg)
        self.mu = cfg[self.NAME].mu
        self.alpha = cfg[self.NAME].alpha
        self.beta = cfg[self.NAME].beta
        self.temperature = cfg[self.NAME].temperature
        self.w = cfg[self.NAME].w

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        self.global_net = self.global_net.to(self.device)

        self.unbiased_fc = copy.deepcopy(self.global_net.cls)

        self.personalized_global_fc = copy.deepcopy(self.global_net.cls)

        self.sub_fc_list = []

        self.personalized_fc_list = []

        # 不同域测试的global
        self.global_net_dict = {}

        # 初始化的时候赋值
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)
            self.sub_fc_list.append(copy.deepcopy(self.unbiased_fc))
            self.personalized_fc_list.append(copy.deepcopy(self.personalized_global_fc))

    # 更新头
    def update_head(self, index, net, train_loader):
        '''
        更新Personalized FC 分支
        '''
        net.eval()
        personalized_fc = self.personalized_fc_list[index].to(self.device)
        personalized_fc.train()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(personalized_fc.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    local_feat = net.features(images)
                local_logits = personalized_fc(local_feat)
                loss_ce = criterion(local_logits, labels)
                optimizer.zero_grad()
                iterator.desc = "Local Personalized FC %d ce = %0.3f" % (index, loss_ce)
                loss_ce.backward()
                optimizer.step()
        personalized_fc.eval()
        self.personalized_fc_list[index] = copy.deepcopy(personalized_fc)

        '''
        更新Unbiased FC 分支
        '''
        sub_fc = self.sub_fc_list[index].to(self.device)
        sub_fc.train()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(sub_fc.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    local_feat = net.features(images)
                    local_logits = net.classifier(local_feat) # 调用global的fc
                query_logits = sub_fc(local_feat)
                loss_ce = criterion(query_logits, labels)

                # bs = labels.shape[0]
                # class_num = self.cfg.DATASET.n_classes
                query_pred = F.log_softmax(query_logits/self.temperature, dim=1)
                local_pred = F.softmax(local_logits/self.temperature, dim=1)
                kd_loss = F.kl_div(query_pred, local_pred, reduction='batchmean') * (self.temperature ** 2)
                kd_loss = 1 / (kd_loss + 1e-5)

                kd_loss = self.w *kd_loss
                loss = loss_ce + kd_loss

                # mad_loss = self.w * self.mad(personalized_fc, sub_fc)
                # loss = loss_ce + mad_loss
                optimizer.zero_grad()
                iterator.desc = "Local Unbiased FC %d ce = %0.3f kd = %0.3f" % (index, loss_ce, kd_loss)
                loss.backward()
                optimizer.step()

        net.train()
        sub_fc.eval()
        self.sub_fc_list[index] = copy.deepcopy(sub_fc)

    # def mad(self, per_fc, un_fc):
    #     g1_w = per_fc.weight  # (K, M, D)
    #     g2_w = un_fc.weight  # (K, D)
    #     w1 = F.normalize(g1_w, dim=-1)  # l2-normalize
    #     w2 = F.normalize(g2_w, dim=-1)  # l2-normalize
    #     reg = torch.einsum("kd,kd->k", w1, w2)
    #     reg = torch.mean(reg ** 2)
    #     return reg

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

    def update(self, priloader_list):
        # 获取所有参与者
        total_clients = list(range(self.cfg.DATASET.parti_num))
        # 随机选取online的参与者
        online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()

        # # 固定方便查看
        # online_clients_list = total_clients

        for i in online_clients_list:
            self.train_net(i, self.nets_list[i], priloader_list[i])
        self.agg_nets(online_clients_list, priloader_list)
        return None

    # 聚合模型 包括头和backbone
    def agg_nets(self, online_clients_list, priloader_list):
        # 获取参与者的聚合权重
        freq_backbone = self.fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        # 更新global的整体参数 需要分发下去！ 维持方法的一致性
        self.fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=self.nets_list,
                                       global_net=self.global_net, freq=freq_backbone, except_part=[], global_only=False)
        # 跟新各种头
        for i in online_clients_list:
            self.update_head(i, self.nets_list[i], priloader_list[i])

        # 更新无偏的全局global fc 需要分发下去！
        self.fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=self.sub_fc_list,
                                       global_net=self.unbiased_fc, freq=freq_backbone, except_part=[], global_only=False)

        # 不同域用不同权重
        self.global_net.eval()
        self.weight_dict = {}
        for domain in self.train_eval_loaders:
            temp_global_net = copy.deepcopy(self.global_net)
            # 计算fc的权重
            freq_fc = self.get_fc_weight(domain, online_clients_list)

            # 更新无偏的全局personalized fc 不需要分发下去！
            self.fed_aggregation.agg_parts(online_clients_list=online_clients_list,nets_list=self.personalized_fc_list,
                                           global_net=self.personalized_global_fc, freq=freq_fc[:-1], except_part=[], global_only=True,
                                           use_additional_net=True, additional_net_list=[self.unbiased_fc],
                                           additional_freq=[freq_fc[-1]])

            # 将权重按照客户端大小排序然后保存
            freq_fc_sorted = freq_fc[np.argsort(online_clients_list + [len(online_clients_list)])]
            self.weight_dict[domain] = freq_fc_sorted

            temp_global_net.cls = copy.deepcopy(self.personalized_global_fc)
            self.global_net_dict[domain] = temp_global_net
        self.global_net.train()

    def cosine_similarity(self,a, b, eps=1e-8):
        return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)

    def pearson_correlation(self,a, b, eps=1e-8): # per 越大 两个越相似
        return self.cosine_similarity(a - a.mean(1).unsqueeze(1),
                                 b - b.mean(1).unsqueeze(1), eps)

    # def inter_class_relation(self,y_s, y_t):
    #     return 1 - self.pearson_correlation(y_s, y_t).mean()

    def get_fc_weight(self, domain, online_clients_list):
        aggregation_weight = torch.nn.Parameter(torch.FloatTensor(len(online_clients_list) + 1), requires_grad=True)
        aggregation_weight.data.fill_(1 / len(aggregation_weight))
        if self.cfg[self.NAME].weight_opt_type == 'SGD':
            optimizer = optim.SGD([aggregation_weight], lr=self.cfg[self.NAME].weight_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        elif self.cfg[self.NAME].weight_opt_type == 'Adam':
            optimizer = torch.optim.Adam([aggregation_weight], lr=self.cfg[self.NAME].weight_lr,
                                         weight_decay=self.cfg.OPTIMIZER.weight_decay)

        iterator = tqdm(range(self.cfg[self.NAME].weight_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(self.train_eval_loaders[domain]):
                imgs_0 = images[0].to(self.device)
                imgs_1 = images[1].to(self.device)

                aggregation_softmax = torch.softmax(aggregation_weight, dim=0)

                # 将fc层转换为保留梯度的形式
                func_g, _ = make_functional(self.global_net.cls)
                para_list = []

                # 聚合fc层 网络参数 这里需要调用personalized的 头
                for _, net_id in enumerate(online_clients_list):
                    cls = self.personalized_fc_list[net_id]

                    _, params = make_functional(cls)
                    para_list.append(params)

                # 无偏头参数
                cls = self.unbiased_fc
                _, params = make_functional(cls)
                para_list.append(params)

                # 参数赋值
                new_g_para = []
                for j in range(len(para_list[0])):
                    new_g_para.append(
                        torch.sum(torch.stack([aggregation_softmax[i] * para_list[i][j] for i in range(len(para_list))]), dim=0))

                # 计算特征
                with torch.no_grad():
                    f_0 = self.global_net.features(imgs_0)
                    f_1 = self.global_net.features(imgs_1)

                # 计算logits
                logits_0 = func_g(new_g_para, f_0)
                logits_1 = func_g(new_g_para, f_1)

                # 计算predi
                pred_0 = F.softmax(logits_0, dim=1)  #
                pred_1 = F.softmax(logits_1, dim=1)  # 这里调用default的弱aug

                # 计算每个样本的MSE损失，并应用权重

                # 计算均方误差损失
                # loss_ins = F.mse_loss(logits_0, logits_1, reduction='mean')

                loss_cos = - torch.nn.CosineSimilarity(dim=1)((pred_0), (pred_1))
                weighted_loss = loss_cos  # 使用unsqueeze将权重扩展到与损失相同的维度
                # 计算加权损失的平均值
                loss_ins = torch.mean(weighted_loss)

                loss_eh = 0.5 * (EH(pred_0) + EH(pred_1))

                loss = self.alpha * loss_ins + self.beta * loss_eh
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iterator.desc = f"{domain} weight = {torch.softmax(aggregation_weight, dim=0).detach().cpu().numpy()}"

        # 训练后的结果softmax
        aggregation_softmax = F.softmax(aggregation_weight, dim=0).detach().cpu().numpy()
        return aggregation_softmax

    def save_checkpoint(self):
        super().save_checkpoint()

        for i in range(len(self.personalized_fc_list)):
            personalized_fc = self.personalized_fc_list[i]
            personalized_fc_path = os.path.join(self.net_folder, f'personalized_fc_{i}_{self.epoch_index}.pth')

            torch.save(personalized_fc.state_dict(), personalized_fc_path)

            print(f'save personalized_fc {i} over')

        unbiased_fc_path = os.path.join(self.net_folder, f'unbiased_fc_{self.epoch_index}.pth')
        torch.save(self.unbiased_fc.state_dict(), unbiased_fc_path)
        print(f'save unbiased_fc over')