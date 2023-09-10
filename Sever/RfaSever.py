import copy

import numpy as np
import torch

from Backbones import get_private_backbones
from Sever.utils.sever_methods import SeverMethod
from Sever.utils.utils import trimmed_mean, fools_gold, geometric_median_update
from utils.utils import row_into_parameters


class RfaSever(SeverMethod):
    NAME = 'RfaSever'

    def __init__(self, args, cfg):
        super(RfaSever, self).__init__(args, cfg)

        self.max_iter = 3

    def sever_update(self, **kwargs):

        online_clients_list = kwargs['online_clients_list']
        priloader_list=kwargs['priloader_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        temp_net = copy.deepcopy(global_net)

        # 算模型差 全局模型参数拉平
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
                # net_all_delta /= np.linalg.norm(net_all_delta)
                all_delta.append(net_all_delta)

            all_delta = np.array(all_delta)
            global_net_para = np.array(torch.cat(global_net_para, dim=0).cpu().numpy())

        online_clients_dl = [priloader_list[online_clients_index] for online_clients_index in online_clients_list]
        online_clients_len = [len(dl.sampler.indices) for dl in online_clients_dl]
        weighted_updates, num_comm_rounds, _ = geometric_median_update(all_delta, online_clients_len,
                                                                       maxiter=self.max_iter, eps=1e-5,
                                                                       verbose=False, ftol=1e-6)
        # update_norm = np.linalg.norm(weighted_updates)
        new_global_net_para = global_net_para + weighted_updates

        row_into_parameters(new_global_net_para, global_net.parameters())