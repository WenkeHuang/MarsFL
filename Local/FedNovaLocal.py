from Local.utils.local_methods import LocalMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
import copy

class FedNovaLocal(LocalMethod):
    NAME = 'FedNovaLocal'

    def __init__(self, args, cfg):
        super(FedNovaLocal, self).__init__(args, cfg)
        self.rho = cfg.Local[self.NAME].rho

    def loc_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        priloader_list = kwargs['priloader_list']
        n_list = kwargs['n_list']
        global_net = kwargs['global_net']
        a_list= kwargs['a_list']
        d_list = kwargs['d_list']

        for i in online_clients_list:  # 遍历循环当前的参与者
            self.train_net(i, nets_list[i], priloader_list[i],global_net,a_list,d_list)
            n_i = len(priloader_list[i])
            n_list.append(n_i)

    def train_net(self, index, net, train_loader,global_net,a_list,d_list):
        net = net.to(self.device)
        net.train()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)

        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
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
        global_model_para = global_net.state_dict()
        net_para = net.state_dict()
        norm_grad = copy.deepcopy(global_net.state_dict())
        for key in norm_grad:
            # norm_grad[key] = (global_model_para[key] - net_para[key]) / a_i
            norm_grad[key] = torch.true_divide(global_model_para[key] - net_para[key], a_i)

        a_list.append(a_i)
        d_list.append(norm_grad)
