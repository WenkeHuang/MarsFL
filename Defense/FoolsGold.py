import copy

import numpy as np
import torch
from Defense.utils.defense_methods import DefenseMethod
from Defense.utils.utils import trimmed_mean, fools_gold
from utils.utils import row_into_parameters


class FoolsGold(DefenseMethod):
    NAME = 'FoolsGold'

    def __init__(self, args, cfg):
        super(FoolsGold, self).__init__(args, cfg)

    def defense_operation(self, **kwargs):

        online_clients_list = kwargs['online_clients_list']

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
                net_all_delta /= (np.linalg.norm(net_all_delta) + 1e-5)
                all_delta.append(net_all_delta)

            all_delta = np.array(all_delta)
            global_net_para = np.array(torch.cat(global_net_para, dim=0).cpu().numpy())

            if not hasattr(self, 'summed_deltas'):
                self.summed_deltas = all_delta
            else:
                self.summed_deltas += all_delta
            this_delta = fools_gold(all_delta, self.summed_deltas,
                                    np.arange(len(online_clients_list)), global_net_para, clip=0)
            new_global_net_para = global_net_para + this_delta
            row_into_parameters(new_global_net_para, global_net.parameters())
            for _, net in enumerate(nets_list):
                net.load_state_dict(global_net.state_dict())

