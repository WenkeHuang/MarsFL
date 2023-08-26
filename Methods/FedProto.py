import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from Methods.utils.meta_methods import FederatedMethod

from utils.utils import cal_client_weight
import torch

# https://github.com/yuetan031/fedproto


def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


class FedProto(FederatedMethod):
    NAME = 'FedProto'

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(FedProto, self).__init__(nets_list, client_domain_list, args, cfg)
        self.mu = cfg[self.NAME].mu
        self.global_protos = []
        self.local_protos = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def proto_aggregation(self,local_protos_list):
        agg_protos_label = dict()
        for idx in self.online_clients:
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos_label[label] = [proto / len(proto_list)]
            else:
                agg_protos_label[label] = [proto_list[0].data]

        return agg_protos_label


    def loc_update(self, priloader_list):
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
        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        for iter in iterator:
            agg_protos_label = {}
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                lossCE = criterion(outputs, labels)

                f = net.features(images)
                loss_mse = nn.MSELoss()
                if len(self.global_protos) == 0:
                    lossProto = 0*lossCE
                else:
                    f_new = copy.deepcopy(f.data)
                    i = 0
                    for label in labels:
                        if label.item() in self.global_protos.keys():
                            f_new[i, :] = self.global_protos[label.item()][0].data
                        i += 1
                    lossProto = loss_mse(f_new, f)

                lossProto = lossProto * self.mu

                loss = lossCE + lossProto
                loss.backward()
                iterator.desc = "Local Pariticipant %d CE = %0.3f,Proto = %0.3f" % (index, lossCE, lossProto)
                optimizer.step()

                if iter == self.cfg.OPTIMIZER.local_epoch - 1:
                    for i in range(len(labels)):
                        if labels[i].item() in agg_protos_label:
                            agg_protos_label[labels[i].item()].append(f[i,:])
                        else:
                            agg_protos_label[labels[i].item()] = [f[i,:]]

        agg_protos = agg_func(agg_protos_label)
        self.local_protos[index] = agg_protos
