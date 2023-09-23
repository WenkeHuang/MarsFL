import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch
import numpy as np


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning FedNova FedNova.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FedNova(FederatedModel):
    NAME = 'fednova'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedNova, self).__init__(nets_list, args, transform)
        self.rho = args.rho

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        self.a_list = []
        self.d_list = []
        self.n_list = []

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
            n_i = len(priloader_list[i])
            self.n_list.append(n_i)

        self.update_global()

    def update_global(self):

        total_n = sum(self.n_list)
        # print("total_n:", total_n)
        d_total_round = copy.deepcopy(self.global_net.state_dict())
        for key in d_total_round:
            d_total_round[key] = 0.0

        for i in range(len(self.online_clients)):
            d_para = self.d_list[i]
            for key in d_para:
                # if d_total_round[key].type == 'torch.LongTensor':
                #    d_total_round[key] += (d_para[key] * n_list[i] / total_n).type(torch.LongTensor)
                # else:
                d_total_round[key] += d_para[key] * self.n_list[i] / total_n

        # for i in range(len(selected)):
        #     d_total_round = d_total_round + d_list[i] * n_list[i] / total_n

        # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

        # update global model
        coeff = 0.0
        for i in range(len(self.online_clients)):
            coeff = coeff + self.a_list[i] * self.n_list[i] / total_n

        updated_model = self.global_net.state_dict()
        for key in updated_model:
            # print(updated_model[key])
            if updated_model[key].type() == 'torch.LongTensor':
                updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
            elif updated_model[key].type() == 'torch.cuda.LongTensor':
                updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
            else:
                # print(updated_model[key].type())
                # print((coeff*d_total_round[key].type()))
                updated_model[key] -= coeff * d_total_round[key]
        self.global_net.load_state_dict(updated_model)

        for _, net in enumerate(self.nets_list):
            net.load_state_dict(self.global_net.state_dict())

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.local_lr, weight_decay=self.args.reg)
        elif self.args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.local_lr, momentum=self.rho,
                                  weight_decay=self.args.reg)

        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))

        tau = 0

        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                if len(images) != 1:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                    optimizer.step()

                    tau += 1

        a_i = (tau - self.rho * (1 - pow(self.rho, tau)) / (1 - self.rho)) / (1 - self.rho)
        global_model_para = self.global_net.state_dict()
        net_para = net.state_dict()
        norm_grad = copy.deepcopy(self.global_net.state_dict())
        for key in norm_grad:
            # norm_grad[key] = (global_model_para[key] - net_para[key]) / a_i
            norm_grad[key] = torch.true_divide(global_model_para[key] - net_para[key], a_i)

        self.a_list.append(a_i)
        self.d_list.append(norm_grad)
