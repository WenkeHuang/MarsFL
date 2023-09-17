import copy

import numpy as np
import torch

from Local.utils.local_methods import LocalMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


def norm_grad(grad_list):
    # input: nested gradients
    # output: square of the L-2 norm # output a flattened array

    return np.sum(np.square(grad_list))


class qffeAVGLocal(LocalMethod):
    NAME = 'qffeAVGLocal'

    def __init__(self, args, cfg):
        super(qffeAVGLocal, self).__init__(args, cfg)
        self.criterion = nn.CrossEntropyLoss()
        self.q = cfg.Local[self.NAME].q

    def get_train_loss(self, net, train_loader):
        net.eval()
        all_loss = []
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = self.criterion(outputs, labels).cpu().numpy()
                all_loss.append(loss)

        net.train()
        return np.mean(all_loss)

    def loc_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        learning_rate = self.cfg.OPTIMIZER.local_train_lr

        all_deltas = []
        hs = []
        for i in online_clients_list:
            loss_before_train = self.get_train_loss(nets_list[i], priloader_list[i])

            self.train_net(i, nets_list[i], priloader_list[i])

            net_all_grads = []
            for name, param0 in global_net.state_dict().items():
                param1 = nets_list[i].state_dict()[name]
                grads = (param0.detach() - param1.detach()) / learning_rate
                net_all_grads.append(copy.deepcopy(grads.view(-1)))
            net_all_grads = torch.cat(net_all_grads, dim=0).cpu().numpy()

            delta = [np.float_power(loss_before_train + 1e-10, self.q) * grad for grad in net_all_grads]
            all_deltas.append(delta)
            data = self.q * np.float_power(loss_before_train + 1e-10, (self.q - 1)) * norm_grad(net_all_grads) + \
                   (1.0 / learning_rate) * np.float_power(loss_before_train + 1e-10, self.q)
            hs.append(data)

        return all_deltas, hs

    def train_net(self, index, net, train_loader):
        net.train()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)

        self.criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = self.criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
