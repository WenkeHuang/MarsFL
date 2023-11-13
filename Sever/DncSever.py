import copy

import numpy as np
import torch

from Sever.utils.sever_methods import SeverMethod

from utils.utils import row_into_parameters


class DncSever(SeverMethod):
    NAME = 'DncSever'

    def __init__(self, args, cfg):
        super(DncSever, self).__init__(args, cfg)
        self.sub_dim = 10000
        self.num_iters = 1
        self.filter_frac = 1.0

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
                net_all_delta /= (np.linalg.norm(net_all_delta) + 1e-5)
                all_delta.append(net_all_delta)

            all_delta = np.array(all_delta)
            global_net_para = np.array(torch.cat(global_net_para, dim=0).cpu().numpy())

        updates = all_delta
        d = len(updates[0])

        bad_client_num = int(self.cfg.attack.bad_client_rate * len(online_clients_list))
        benign_ids = []
        for i in range(self.num_iters):
            indices = torch.randperm(d)[: self.sub_dim]
            sub_updates = updates[:, indices]
            mu = sub_updates.mean(axis=0)
            centered_update = sub_updates - mu
            v = np.linalg.svd(centered_update, full_matrices=False)[2][0, :]
            s = np.array(
                [(np.dot(update - mu, v) ** 2).item() for update in sub_updates]
            )

            good = s.argsort()[: len(updates) - int(self.filter_frac * bad_client_num)]

            benign_ids.extend(good)

        benign_ids = list(set(benign_ids))
        benign_updates = updates[benign_ids, :].mean(axis=0)

        new_global_net_para = global_net_para + benign_updates
        row_into_parameters(new_global_net_para, global_net.parameters())
        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())

