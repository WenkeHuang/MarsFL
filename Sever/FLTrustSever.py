import copy

import numpy as np
import torch
from torch import optim, nn
from tqdm import tqdm

from Datasets.public_dataset import get_public_dataset
from Sever.utils.sever_methods import SeverMethod

from utils.utils import row_into_parameters


class FLTrustSever(SeverMethod):
    NAME = 'FLTrustSever'

    def __init__(self, args, cfg):
        super(FLTrustSever, self).__init__(args, cfg)

        public_dataset_name = cfg.Sever[self.NAME].public_dataset_name
        pub_len = cfg.Sever[self.NAME].pub_len
        pub_aug = cfg.Sever[self.NAME].pub_aug
        public_batch_size = cfg.Sever[self.NAME].public_batch_size
        self.public_epoch = cfg.Sever[self.NAME].public_epoch
        self.public_dataset = get_public_dataset(args, cfg, public_dataset_name=public_dataset_name,
                                                 pub_len=pub_len, pub_aug=pub_aug, public_batch_size=public_batch_size)
        self.public_dataset.get_data_loaders()
        self.public_loader = self.public_dataset.traindl

    def sever_update(self, **kwargs):

        online_clients_list = kwargs['online_clients_list']

        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        temp_net = copy.deepcopy(global_net)

        with torch.no_grad():
            all_delta = []
            global_net_para = []
            add_global = True
            for i in online_clients_list:

                net_all_delta = []
                for name, param0 in temp_net.state_dict().items():
                    param1 = nets_list[i].state_dict()[name]
                    delta = (param1.detach() - param0.detach())

                    net_all_delta.append(copy.deepcopy(delta.view(-1)))
                    if add_global:
                        weights = copy.deepcopy(param0.detach().view(-1))
                        global_net_para.append(weights)

                add_global = False
                net_all_delta = torch.cat(net_all_delta, dim=0).cpu().numpy()
                all_delta.append(net_all_delta)

            all_delta = np.array(all_delta)
            global_net_para = np.array(torch.cat(global_net_para, dim=0).cpu().numpy())

        criterion = nn.CrossEntropyLoss()
        iterator = tqdm(range(self.public_epoch))
        optimizer = optim.SGD(temp_net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                              momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(self.public_loader):
                images = images
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = temp_net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            global_delta = []
            for name, param0 in temp_net.state_dict().items():
                param1 = global_net.state_dict()[name]
                delta = (param0.detach() - param1.detach())
                global_delta.append(copy.deepcopy(delta.view(-1)))

            global_delta = torch.cat(global_delta, dim=0).cpu().numpy()
            global_delta = np.array(global_delta)

        total_TS = 0
        TSnorm = []
        for d in all_delta:
            tmp_weight = copy.deepcopy(d)

            TS = np.dot(tmp_weight, global_delta) / (np.linalg.norm(tmp_weight) * np.linalg.norm(global_delta) + 1e-5)
            # print(TS)
            if TS < 0:
                TS = 0
            total_TS += TS

            norm = np.linalg.norm(global_delta) / (np.linalg.norm(tmp_weight) + 1e-5)
            TSnorm.append(TS * norm)

        delta_weight = np.sum(np.array(TSnorm).reshape(-1, 1) * all_delta, axis=0) / (total_TS + 1e-5)
        new_global_net_para = global_net_para + delta_weight
        row_into_parameters(new_global_net_para, global_net.parameters())
        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())

