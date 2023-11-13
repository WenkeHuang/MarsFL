from Local.utils.local_methods import LocalMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
import torch


class ScaffoldOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(
            lr=lr, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def step(self, server_control, client_control, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        ng = len(self.param_groups[0]["params"])
        names = list(server_control.keys())

        # BatchNorm: running_mean/std, num_batches_tracked
        names = [name for name in names if "running" not in name]
        names = [name for name in names if "num_batch" not in name]

        t = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                c = server_control[names[t]]
                ci = client_control[names[t]]

                # print(names[t], p.shape, c.shape, ci.shape)
                d_p = p.grad.data + c.data - ci.data
                p.data = p.data - d_p.data * group["lr"]
                t += 1
        # assert t == ng
        return loss


class ScaffoldLocal(LocalMethod):
    NAME = 'ScaffoldLocal'

    def __init__(self, args, cfg):
        super(ScaffoldLocal, self).__init__(args, cfg)
        self.max_grad_norm = cfg.Local[self.NAME].max_grad_norm
        # self.infoNCET = cfg.Local[self.NAME].infoNCET

    def loc_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        local_controls = kwargs['local_controls']
        global_control = kwargs['global_control']
        delta_models = kwargs['delta_models']
        delta_controls = kwargs['delta_controls']
        for i in online_clients_list:
            self.train_net(i, nets_list[i], priloader_list[i], global_net, local_controls, global_control, delta_models)
            self.update_local_control(i, local_controls, global_control, delta_models, delta_controls)

    def train_net(self, index, net, train_loader, global_net, local_controls, global_control, delta_models):
        net = net.to(self.device)
        net.train()
        # optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        optimizer = ScaffoldOptimizer(
            net.parameters(),
            lr=self.cfg.OPTIMIZER.local_train_lr,
            weight_decay=self.cfg.OPTIMIZER.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        self.cnt = 0
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    net.parameters(), self.max_grad_norm
                )

                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step(server_control=global_control, client_control=local_controls[index])
                self.cnt += 1

        delta_model = self.get_delta_model(global_net, net)
        delta_models[index] = copy.deepcopy(delta_model)

    def update_local_control(self, index, local_controls, global_control, delta_models, delta_controls):
        client_control = local_controls[index]
        delta_model = delta_models[index]

        new_control = copy.deepcopy(client_control)
        delta_control = copy.deepcopy(client_control)

        for name in delta_models[index].keys():
            c = global_control[name]
            ci = client_control[name]
            delta = delta_model[name]

            new_ci = ci.data - c.data + delta.data / (self.cnt * self.cfg.OPTIMIZER.local_train_lr)
            new_control[name].data = new_ci
            delta_control[name].data = ci.data - new_ci

        local_controls[index] = copy.deepcopy(new_control)
        delta_controls[index] = copy.deepcopy(delta_control)

    def get_delta_model(self, model_0, model_1):
        """ return a dict: {name: params}
        """
        state_dict = {}
        for name, param0 in model_0.state_dict().items():
            param1 = model_1.state_dict()[name]
            state_dict[name] = param0.detach() - param1.detach()
        return state_dict
