import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch
import numpy as np


# https://github.com/lxcnju/FedRepo/blob/main/algorithms/fedrs.py
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via Fedavg.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FedRs(FederatedModel):
    NAME = 'fedrs'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedRs, self).__init__(nets_list, args, transform)

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
        self.aggregate_nets(None)

        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        # if self.args.optimizer == 'adam':
        #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.local_lr, weight_decay=self.args.reg)
        # elif self.args.optimizer == 'sgd':
        #     optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.local_lr, momentum=0.9,
        #                           weight_decay=self.args.reg)
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)

        cls_count_dict = self.net_cls_counts[index]

        cls_counts = torch.tensor(list(cls_count_dict.values()), device=self.device)
        cls_rate = cls_counts / torch.sum(cls_counts)

        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                features = net.features(images)

                ws = net.fc.weight

                cdist = cls_rate / cls_rate.max()
                cdist = cdist * (1.0 - self.args.alpha) + self.args.alpha
                cdist = cdist.reshape((1, -1))

                logits = cdist * features.mm(ws.transpose(0, 1))

                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
