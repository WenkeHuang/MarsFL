import copy

import numpy as np
import torch

from Defense.utils.defense_methods import DefenseMethod
from Defense.utils.utils import trimmed_mean
from Backbones import get_private_backbones
from utils.utils import row_into_parameters

class TrimmedMean(DefenseMethod):
    NAME = 'TrimmedMean'

    def __init__(self, args, cfg):
        super(TrimmedMean, self).__init__(args, cfg)

        nets_list = get_private_backbones(cfg)

        self.momentum = 0.9
        self.learning_rate = self.cfg.OPTIMIZER.local_train_lr
        self.current_weights = []
        for name, param in copy.deepcopy(nets_list[0]).cpu().state_dict().items():
            param = nets_list[0].state_dict()[name].view(-1)
            self.current_weights.append(param)
        self.current_weights = torch.cat(self.current_weights, dim=0).cpu().numpy()
        self.velocity = np.zeros(self.current_weights.shape, self.current_weights.dtype)
        self.n = 5

    def defense_operation(self, **kwargs):

        online_clients_list = kwargs['online_clients_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        temp_net = copy.deepcopy(global_net)
        # 算模型梯度
        with torch.no_grad():
            all_grads = []
            for i in online_clients_list:
                grads = {}
                net_all_grads = []
                for name, param0 in temp_net.state_dict().items():
                    param1 = nets_list[i].state_dict()[name]
                    grads[name] = (param0.detach() - param1.detach()) / self.learning_rate
                    net_all_grads.append(copy.deepcopy(grads[name].view(-1)))

                net_all_grads = torch.cat(net_all_grads, dim=0).cpu().numpy()
                all_grads.append(net_all_grads)
            all_grads = np.array(all_grads)

        # bad_client_num = int(self.args.bad_client_rate * len(self.online_clients))
        f = len(online_clients_list) // 2  # worse case 50% malicious points
        k = len(online_clients_list) - f - 1

        current_grads = trimmed_mean(all_grads, len(online_clients_list), k)

        self.velocity = self.momentum * self.velocity - self.learning_rate * current_grads
        self.current_weights += self.velocity

        row_into_parameters(self.current_weights, global_net.parameters())
        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())

