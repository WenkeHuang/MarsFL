import numpy as np
import torch

from Local.utils.local_methods import LocalMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy

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


class FedProtoLocal(LocalMethod):
    NAME = 'FedProtoLocal'

    def __init__(self, args, cfg):
        super(FedProtoLocal, self).__init__(args, cfg)
        self.mu = cfg.Local[self.NAME].mu


    def loc_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        global_protos = kwargs['global_protos']
        local_protos=kwargs['local_protos']
        epoch_index=kwargs['epoch_index']

        for i in online_clients_list:  # 遍历循环当前的参与者
            self.train_net(i, nets_list[i], global_net, priloader_list[i], global_protos,local_protos,epoch_index)

    def train_net(self, index, net, global_net, train_loader, global_protos, local_protos, epoch_index):
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
                if len(global_protos) == 0:
                    lossProto = 0*lossCE
                else:
                    f_new = copy.deepcopy(f.data)
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            f_new[i, :] = global_protos[label.item()][0].data
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
        local_protos[index] = agg_protos
