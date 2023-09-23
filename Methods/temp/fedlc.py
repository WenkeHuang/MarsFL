import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch
import numpy as np


# https://github.com/KarhouTam/FL-bench/blob/007d88e4bb7000901b6ba35cd80189e0d6ad11ac/src/client/fedlc.py#L5
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedLc.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FedLc(FederatedModel):
    NAME = 'fedlc'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedLc, self).__init__(nets_list, args, transform)
        self.tau = self.args.tau

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
        # self._classifier_weight()
        self.aggregate_nets(None)
        return None

    def logit_calibrated_loss(self, logit, y, label_distrib):
        label_distrib[label_distrib == 0] = 1e-8
        cal_logit = torch.exp(
            logit
            - (
                    self.tau
                    * torch.pow(label_distrib, -1 / 4)
                    .unsqueeze(0)
                    .expand((logit.shape[0], -1))
            )
        )

        cal_logit = torch.clip(cal_logit, max=100)

        y_logit = torch.gather(cal_logit, dim=-1, index=y.unsqueeze(1))
        loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
        # a=torch.where(loss == torch.nan)
        # b=torch.where(loss == torch.inf)
        loss_out = loss.sum() / logit.shape[0]
        return loss_out

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.local_lr, weight_decay=self.args.reg)
        elif self.args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.local_lr, momentum=0.9,
                                  weight_decay=self.args.reg)

        criterion = self.logit_calibrated_loss
        label_distrib = torch.tensor(list(self.net_cls_counts[index].values()), device=self.device).float()
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                # if len(images)!=1:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels, label_distrib)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
