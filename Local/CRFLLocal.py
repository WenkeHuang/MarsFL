import torch

from Local.utils.local_methods import LocalMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


class CRFLLocal(LocalMethod):
    NAME = 'CRFLLocal'

    def __init__(self, args, cfg):
        super(CRFLLocal, self).__init__(args, cfg)
        self.scale_factor = cfg.Local[self.NAME].scale_factor

    def loc_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']

        submit_params_update_dict = {}

        target_params = dict()
        for name, param in global_net.named_parameters():
            target_params[name] = global_net.state_dict()[name].clone().detach().requires_grad_(False)

        for i in online_clients_list:
            self.train_net(i, nets_list[i], priloader_list[i])

            for name, data in nets_list[i].state_dict().items():
                new_value = target_params[name] + (data - target_params[name]) * self.scale_factor
                nets_list[i].state_dict()[name].copy_(new_value)

            client_pramas_update = dict()
            for name, data in nets_list[i].state_dict().items():
                client_pramas_update[name] = torch.zeros_like(data)
                client_pramas_update[name] = (data - target_params[name])

            submit_params_update_dict[i] = client_pramas_update

        return submit_params_update_dict

    def train_net(self, index, net, train_loader):
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
