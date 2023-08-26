import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from Methods.utils.meta_methods import FederatedMethod

from utils.utils import cal_client_weight


# https://github.com/katsura-jp/fedavg.pytorch
# https://github.com/vaseline555/Federated-Averaging-PyTorch

class FedDyn(FederatedMethod):
    NAME = 'FedDyn'

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(FedDyn, self).__init__(nets_list, client_domain_list, args, cfg)
        self.client_grads = {}
        self.reg_lamb = cfg[self.NAME].reg_lamb

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

        for i in range(len(self.nets_list)):
            self.client_grads[i] = self.build_grad_dict(self.global_net)

    def build_grad_dict(self, model):
        grad_dict = {}
        for key, params in model.state_dict().items():
            grad_dict[key] = torch.zeros_like(params)
        return grad_dict

    def update(self, priloader_list):
        total_clients = list(range(self.cfg.DATASET.parti_num))
        online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()  # 随机选取online的参与者

        for i in online_clients_list:
            self._train_net(i, self.nets_list[i], priloader_list[i])

        # 获取参与者的聚合权重
        freq = self.fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        client_weight = cal_client_weight(online_clients_list=online_clients_list, client_domain_list=self.client_domain_list, freq=freq)
        print(client_weight)

        # 聚合除了Lora的部分 即 Bone + cls
        self.fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=self.nets_list,
                                       global_net=self.global_net, freq=freq, except_part=['lora_'], global_only=False)

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)

        local_grad = copy.deepcopy(self.client_grads[index])

        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)

                reg_loss = 0.0
                cnt = 0.0
                for name, param in self.global_net.named_parameters():
                    term1 = (param * (
                            local_grad[name] - self.global_net.state_dict()[name]
                    )).sum()
                    term2 = (param * param).sum()

                    reg_loss += self.reg_lamb * (term1 + term2)
                    cnt += 1.0

                loss_ce = criterion(outputs, labels)
                loss = loss_ce + reg_loss / cnt
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()

        for name, param in net.named_parameters():
            local_grad[name] += (
                    net.state_dict()[name] - self.global_net.state_dict()[name]
            )
        self.client_grads[index] = local_grad
