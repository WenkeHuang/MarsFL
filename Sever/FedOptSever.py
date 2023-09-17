import copy

import torch

from Sever.utils.sever_methods import SeverMethod


class FedOptSever(SeverMethod):
    NAME = 'FedOptSever'

    def __init__(self, args, cfg):
        super(FedOptSever, self).__init__(args, cfg)
        self.global_lr = cfg.Sever[self.NAME].global_lr

    def sever_update(self, **kwargs):

        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']

        mean_state_dict = {}
        if not hasattr(self, 'global_optimizer'):
            self.global_optimizer = torch.optim.SGD(
                global_net.parameters(),
                lr=self.global_lr,
                momentum=0.9,
                weight_decay=0.0
            )
        for name, param in global_net.state_dict().items():
            vs = []
            for client in nets_list:
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
        new_model = copy.deepcopy(global_net)
        new_model.load_state_dict(mean_state_dict, strict=True)

        # set global_model gradient
        with torch.no_grad():
            for param, new_param in zip(
                    global_net.parameters(), new_model.parameters()
            ):
                param.grad = param.data - new_param.data

        # replace some non-parameters's state dict
        state_dict = global_net.state_dict()
        for name in dict(global_net.named_parameters()).keys():
            mean_state_dict[name] = state_dict[name]
        global_net.load_state_dict(mean_state_dict, strict=True)

        # optimization
        self.global_optimizer = torch.optim.SGD(
            global_net.parameters(),
            lr=self.global_lr,
            momentum=0.9,
            weight_decay=0.0
        )
        self.global_optimizer.load_state_dict(global_optimizer_state)
        self.global_optimizer.step()

        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())
