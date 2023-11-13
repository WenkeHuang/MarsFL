import numpy as np
import torch

from Local.utils.local_methods import LocalMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


class MOONLocal(LocalMethod):
    NAME = 'MOONLocal'

    def __init__(self, args, cfg):
        super(MOONLocal, self).__init__(args, cfg)
        self.mu = cfg.Local[self.NAME].mu
        self.temperature_moon = cfg.Local[self.NAME].temperature_moon
        # self.global_lr = cfg.Sever[self.NAME].global_lr

    def loc_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        prev_nets_list = kwargs['prev_nets_list']

        for i in online_clients_list:
            self.train_net(i, nets_list[i], global_net, prev_nets_list[i], priloader_list[i])

    def train_net(self, index, net, global_net, prev_net, train_loader):
        net = net.to(self.device)
        prev_net = prev_net.to(self.device)
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        global_net = global_net.to(self.device)
        cos = torch.nn.CosineSimilarity(dim=-1)
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                f = net.features(images)
                outputs = net.classifier(f)
                with torch.no_grad():
                    pre_f = prev_net.features(images)
                    g_f = global_net.features(images)
                posi = cos(f, g_f)
                temp = posi.reshape(-1, 1)
                nega = cos(f, pre_f)
                temp = torch.cat((temp, nega.reshape(-1, 1)), dim=1)
                temp /= self.temperature_moon
                temp = temp.to(self.device)
                targets = torch.zeros(labels.size(0)).to(self.device).long()
                lossCON = self.mu * criterion(temp, targets)

                lossCE = criterion(outputs, labels)
                loss = lossCE + lossCON
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d CE = %0.3f,CON = %0.3f" % (index, lossCE, lossCON)
                optimizer.step()
