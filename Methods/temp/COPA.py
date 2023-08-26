
from Methods.utils.meta_methods import FederatedMethod
from utils.utils import cal_client_weight, set_requires_grad
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy


class COPA(FederatedMethod):
    NAME = 'COPA'

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(COPA, self).__init__(nets_list, client_domain_list, args, cfg)

        # self.pred_threshold = cfg[self.NAME].pred_threshold
        self.head_dict = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        self.global_net = self.global_net.to(self.device)

        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

        # 其他客户端的头
        for i in range(len(self.nets_list)):
            self.head_dict[i] = {}
            for j in range(len(self.nets_list)):
                # 加入其他头 并且不要梯度
                if i != j:
                    head = copy.deepcopy(self.nets_list[j].cls)
                    set_requires_grad(head, False)
                    self.head_dict[i][j] = head

    def update(self, priloader_list):
        total_clients = list(range(self.cfg.DATASET.parti_num))  # 获取所有参与者
        online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()  # 随机选取online的参与者

        for i in online_clients_list:  # 遍历循环当前的参与者
            self.train_net(i, self.nets_list[i], priloader_list[i])

        # 获取参与者的聚合权重
        freq = self.fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        client_weight = cal_client_weight(online_clients_list=online_clients_list, client_domain_list=self.client_domain_list, freq=freq)
        print(client_weight)

        # FedAVG 是聚合Bone
        self.fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=self.nets_list,
                                       global_net=self.global_net, freq=freq, except_part=['cls'], global_only=False)

        # 更新客户端的头
        self.head_dict = {}
        for i in range(len(self.nets_list)):
            self.head_dict[i] = {}
            for j in range(len(self.nets_list)):
                # 加入其他头 并且不要梯度
                if i != j:
                    head = copy.deepcopy(self.nets_list[j].cls)
                    set_requires_grad(head, False)
                    self.head_dict[i][j] = head
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
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                f = net.features(images)
                outputs = net.cls(f)
                loss = criterion(outputs, labels)

                # 其他头的loss
                loss_other = 0
                head_dict = self.head_dict[index]
                for k in head_dict:
                    head = head_dict[k]
                    out = head(f)
                    loss_other += criterion(out, labels)

                loss = loss + loss_other
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
