import copy

import numpy as np
import torch

from Sever.utils.sever_methods import SeverMethod
from utils.utils import row_into_parameters


class qffeAVGSever(SeverMethod):
    NAME = 'qffeAVGSever'

    def __init__(self, args, cfg):
        super(qffeAVGSever, self).__init__(args, cfg)

    def sever_update(self, **kwargs):
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']

        weights_before = []
        for name, param0 in global_net.state_dict().items():
            weights = copy.deepcopy(param0.detach().view(-1))
            weights_before.append(weights)
        weights_before = np.array(torch.cat(weights_before, dim=0).cpu().numpy())

        all_deltas = kwargs['all_deltas']
        hs = kwargs['hs']

        demominator = np.sum(np.asarray(hs))
        # num_clients = len(all_deltas)
        scaled_deltas = []
        for client_delta in all_deltas:
            scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])

        updates = []
        for i in range(len(all_deltas[0])):
            tmp = scaled_deltas[0][i]
            for j in range(1, len(all_deltas)):
                tmp += scaled_deltas[j][i]
            updates.append(tmp)

        new_solutions = [(u - v) * 1.0 for u, v in zip(weights_before, updates)]
        new_solutions = np.array(new_solutions)

        row_into_parameters(new_solutions, global_net.parameters())
        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())
