from Methods.utils.meta_methods import FederatedMethod
from functorch import make_functional
import torch.nn.functional as F
import torch.optim as optim
from utils.utils import EH
from tqdm import tqdm
import torch.nn as nn
import torch
import copy


class FedCe(FederatedMethod):
    NAME = 'FedCe'

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(FedCe, self).__init__(nets_list, client_domain_list, args, cfg)
        self.mu = cfg[self.NAME].mu
        self.alpha = cfg[self.NAME].alpha
        self.beta = cfg[self.NAME].beta
        self.temperature = cfg[self.NAME].temperature

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        self.global_net = self.global_net.to(self.device)

        self.unbiased_fc = copy.deepcopy(self.global_net.cls)

        self.sub_fc_list = []

        # 不同域测试的global
        self.global_net_dict = {}

        # 初始化的时候赋值
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)
            self.sub_fc_list.append(copy.deepcopy(self.unbiased_fc))

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
                images = images[0].to(self.device)
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

        '''
        更新另一个分支
        '''
        sub_fc = self.sub_fc_list[index].to(self.device)
        sub_fc.train()
        net.eval()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(sub_fc.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images[0].to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    local_feat = net.features(images)
                    local_logits = net.classifier(local_feat)

                query_logits = sub_fc(local_feat)
                loss_ce = criterion(query_logits, labels)

                gt_mask = self.get_gt_mask(query_logits, labels)

                local_pred = F.softmax(local_logits / self.temperature - 1000.0 * gt_mask, dim=1)

                query_pred = F.log_softmax(query_logits / self.temperature - 1000.0 * gt_mask, dim=1)

                nckd_loss = F.kl_div(query_pred, local_pred, reduction='batchmean') * (self.temperature ** 2)

                nckd_loss = - nckd_loss  # 最大化这一项
                loss = loss_ce + nckd_loss
                optimizer.zero_grad()
                iterator.desc = "Local Pariticipant %d ce = %0.3f dkd = %0.3f" % (index, loss_ce, nckd_loss)
                loss.backward()
                optimizer.step()

        net.train()
        sub_fc.eval()
        self.sub_fc_list[index] = copy.deepcopy(sub_fc)

    def get_gt_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    def loc_update(self, priloader_list):
        # 获取所有参与者
        total_clients = list(range(self.cfg.DATASET.parti_num))
        # 随机选取online的参与者
        online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()

        # 固定方便查看
        online_clients_list = total_clients

        for i in online_clients_list:
            self.train_net(i, self.nets_list[i], priloader_list[i])

        # 获取参与者的聚合权重
        freq_backbone = self.fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        # 聚合backbone 并分发给local
        # except_part = ['lora_', 'cls']
        except_part = ['cls']
        # 更新每个local的backbone
        self.fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=self.nets_list,
                                       global_net=self.global_net, freq=freq_backbone, except_part=except_part, global_only=False)

        # 更新global的整体参数
        self.fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=self.nets_list,
                                       global_net=self.global_net, freq=freq_backbone, except_part=[], global_only=True)

        # 更新无偏的全局global fc
        self.fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=self.sub_fc_list,
                                       global_net=self.unbiased_fc, freq=freq_backbone, except_part=[], global_only=False)

        temp_global_net = copy.deepcopy(self.global_net)

        # 不同域用不同权重
        for domain in self.train_eval_loaders:
            # 计算fc的权重
            freq_fc = self.get_fc_weight(domain, online_clients_list)

            # backbone一样情况下 聚合全部的 相当于聚合lora和fc 不过lora和fc不分发
            self.fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=self.nets_list,
                                           global_net=self.global_net, freq=freq_fc, except_part=[], global_only=True)

            self.global_net_dict[domain] = copy.deepcopy(self.global_net)

        self.global_net = temp_global_net
        return None

    def get_fc_weight(self, domain, online_clients_list):
        aggregation_weight = torch.nn.Parameter(torch.FloatTensor(len(online_clients_list)), requires_grad=True)
        aggregation_weight.data.fill_(1 / len(online_clients_list))
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

                # 聚合fc层 网络参数
                for _, net_id in enumerate(online_clients_list):
                    cls = self.nets_list[net_id].cls
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

                    unbiased_logits = self.unbiased_fc(f_0)
                    unbiased_pred = F.softmax(unbiased_logits, dim=1)
                    unbiased_ent = - (unbiased_pred * (unbiased_pred + 1e-5).log()).sum(dim=1)
                    ins_weight = torch.exp(torch.mean(unbiased_ent) - unbiased_ent)

                # 计算logits
                logits_0 = func_g(new_g_para, f_0)  # 弱增强
                logits_1 = func_g(new_g_para, f_1)  # 强增强

                with torch.no_grad():
                    logits_aim = func_g(new_g_para, f_0)  # 弱增强

                # kl 强aug往弱的拉
                loss_kl = F.kl_div(F.log_softmax(logits_1, dim=1), F.softmax(logits_aim, dim=1), reduction='none')
                weighted_loss = loss_kl * ins_weight.unsqueeze(1)  # 使用unsqueeze将权重扩展到与损失相同的维度
                # 计算加权损失的平均值
                loss_ins = torch.mean(weighted_loss)

                # 单个样本输出Sharpness
                softmax_aggregation_output0 = F.softmax(logits_0, dim=1)  # 这里调用default的弱aug
                loss_eh = EH(softmax_aggregation_output0, ins_weight)

                # 整体样本输出Diversity
                # loss_he = HE(softmax_aggregation_output0)

                loss = self.alpha * loss_ins + self.beta * loss_eh
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iterator.desc = f"{domain} weight = {torch.softmax(aggregation_weight, dim=0).detach().cpu().numpy()}"

        # 训练后的结果softmax
        aggregation_softmax = F.softmax(aggregation_weight, dim=0).detach().cpu().numpy()
        return aggregation_softmax
