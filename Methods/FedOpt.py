import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from Methods.utils.meta_methods import FederatedMethod

from utils.utils import cal_client_weight


class FedOpt(FederatedMethod):
    NAME = 'FedOpt'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(FedOpt, self).__init__(nets_list, client_domain_list, args, cfg)

        self.global_lr = cfg[self.NAME].global_lr  # 0.5 0.25 0.1

        self.global_optimizer = None

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        self.global_optimizer = torch.optim.SGD(
            self.global_net.parameters(),
            lr=self.global_lr,
            momentum=0.9,
            weight_decay=0.0
        )
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def update_global(self):
        mean_state_dict = {}

        for name, param in self.global_net.state_dict().items():
            vs = []
            for client in self.nets_list:
                vs.append(client.state_dict()[name])
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()
            mean_state_dict[name] = mean_value

        # zero_grad
        self.global_optimizer.zero_grad()
        global_optimizer_state = self.global_optimizer.state_dict()

        # new_model
        new_model = copy.deepcopy(self.global_net)
        new_model.load_state_dict(mean_state_dict, strict=True)

        # set global_model gradient
        with torch.no_grad():
            for param, new_param in zip(
                    self.global_net.parameters(), new_model.parameters()
            ):
                param.grad = param.data - new_param.data

        # replace some non-parameters's state dict
        state_dict = self.global_net.state_dict()
        for name in dict(self.global_net.named_parameters()).keys():
            mean_state_dict[name] = state_dict[name]
        self.global_net.load_state_dict(mean_state_dict, strict=True)

        # optimization
        self.global_optimizer = torch.optim.SGD(
            self.global_net.parameters(),
            lr=self.global_lr,
            momentum=0.9,
            weight_decay=0.0
        )
        self.global_optimizer.load_state_dict(global_optimizer_state)
        self.global_optimizer.step()

        for _, net in enumerate(self.nets_list):
            net.load_state_dict(self.global_net.state_dict())

    def loc_update(self, priloader_list):
        total_clients = list(range(self.cfg.DATASET.parti_num))
        online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()  # 随机选取online的参与者

        # self.aggregate_nets(None)
        for i in online_clients_list:
            self._train_net(i, self.nets_list[i], priloader_list[i])

        self.update_global()
        return None

    def _train_net(self, index, net, train_loader):
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
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
