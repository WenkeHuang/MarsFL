import os

import numpy as np
from functorch import make_functional

from Methods.utils.meta_methods import FederatedMethod
from utils.utils import EH
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
import torch
import torch.nn.functional as F


class MOONCOSAddGlobal(FederatedMethod):
    NAME = 'MOONCOSAddGlobal'

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(MOONCOSAddGlobal, self).__init__(nets_list, client_domain_list, args, cfg)
        self.prev_nets_list = []
        self.mu = cfg[self.NAME].mu
        self.temperature_moon = cfg[self.NAME].temperature_moon
        self.alpha = cfg[self.NAME].alpha
        self.beta = cfg[self.NAME].beta
        self.w = cfg[self.NAME].w
        self.temperature = cfg[self.NAME].temperature

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        self.global_net = self.global_net.to(self.device)

        self.unbiased_fc = copy.deepcopy(self.global_net.cls)

        self.personalized_global_fc = copy.deepcopy(self.global_net.cls)

        self.sub_fc_list = []

        self.personalized_fc_list = []

        # 不同域测试的global
        self.global_net_dict = {}

        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            self.prev_nets_list.append(copy.deepcopy(net))  # 初始化在loading global之前
            net.load_state_dict(global_w)
            self.sub_fc_list.append(copy.deepcopy(self.unbiased_fc))
            self.personalized_fc_list.append(copy.deepcopy(self.personalized_global_fc))

    def loc_update(self, priloader_list):
        total_clients = list(range(self.cfg.DATASET.parti_num))  # 获取所有参与者
        online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()  # 随机选取online的参与者

        for i in online_clients_list:  # 遍历循环当前的参与者
            self.train_net(i, self.nets_list[i], self.prev_nets_list[i], priloader_list[i])

        self.copy_nets2_prevnets()

        self.agg_nets(online_clients_list, priloader_list)
        return None

    def train_net(self, index, net, prev_net, train_loader):
        net = net.to(self.device)
        prev_net = prev_net.to(self.device)
        if self.cfg.OPTIMIZER.type == 'SGD':
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
                gt_mask = self.get_gt_mask(query_logits, labels)
                local_pred = F.softmax(local_logits / self.temperature - 1000.0 * gt_mask, dim=1)
                query_pred = F.log_softmax(query_logits / self.temperature - 1000.0 * gt_mask, dim=1)
                nckd_loss = F.kl_div(query_pred, local_pred, reduction='batchmean') * (self.temperature ** 2)
                nckd_loss = 1 / (nckd_loss + 1e-5)
                loss = loss_ce + self.w * nckd_loss
                optimizer.zero_grad()
                iterator.desc = "Local Unbiased FC %d ce = %0.3f dkd = %0.3f" % (index, loss_ce, nckd_loss)
                loss.backward()
                optimizer.step()

        net.train()
        sub_fc.eval()
        self.sub_fc_list[index] = copy.deepcopy(sub_fc)

    def get_fc_weight(self, domain, online_clients_list):
        aggregation_weight = torch.nn.Parameter(torch.FloatTensor(len(online_clients_list) + 1), requires_grad=True)
        aggregation_weight.data.fill_(1 / len(aggregation_weight))
        if self.cfg[self.NAME].weight_opt_type == 'SGD':
            optimizer = optim.SGD([aggregation_weight], lr=self.cfg[self.NAME].weight_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.cfg.OPTIMIZER.weight_decay)
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

                # 计算每个样本的MSE损失，并应用权重
                loss_cos = 1 - torch.nn.CosineSimilarity(dim=1)((logits_0), (logits_1))
                weighted_loss = loss_cos  # 使用unsqueeze将权重扩展到与损失相同的维度

                # 计算加权损失的平均值
                loss_ins = torch.mean(weighted_loss)

                # 单个样本输出Sharpness
                softmax_aggregation_output0 = F.softmax(logits_0, dim=1)  # 这里调用default的弱aug
                loss_eh = EH(softmax_aggregation_output0)

                loss = self.alpha * loss_ins + self.beta * loss_eh
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iterator.desc = f"{domain} weight = {torch.softmax(aggregation_weight, dim=0).detach().cpu().numpy()}"

        # 训练后的结果softmax
        aggregation_softmax = F.softmax(aggregation_weight, dim=0).detach().cpu().numpy()
        return aggregation_softmax

    def save_checkpoint(self):
        global_net_path = os.path.join(self.net_folder, f'global_net_{self.cfg.DATASET.backbone}.pth')

        torch.save(self.global_net.state_dict(), global_net_path)
        print('save global_net over')

        for i in range(len(self.personalized_fc_list)):
            personalized_fc = self.personalized_fc_list[i]
            personalized_fc_path = os.path.join(self.net_folder, f'personalized_fc_{i}.pth')

            torch.save(personalized_fc.state_dict(), personalized_fc_path)

            print(f'save personalized_fc {i} over')

        unbiased_fc_path = os.path.join(self.net_folder, f'unbiased_fc.pth')
        torch.save(self.unbiased_fc.state_dict(), unbiased_fc_path)
        print(f'save unbiased_fc over')
