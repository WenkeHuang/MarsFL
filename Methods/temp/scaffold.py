import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via SCAFFOLD.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


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


class Scaffold(FederatedModel):
    NAME = 'scaffold'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(Scaffold, self).__init__(nets_list, args, transform)
        self.global_control = {}
        self.local_controls = {}
        self.delta_models = {}
        self.delta_controls = {}
        self.max_grad_norm = 100
        self.global_lr = args.global_lr  # 0.5 0.25 0.1

    def init_control(self, model):
        """ a dict type: {name: params}
        """
        control = {
            name: torch.zeros_like(
                p.data
            ).to(self.device) for name, p in model.state_dict().items()
        }
        return control

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        self.global_control = self.init_control(self.global_net)
        self.local_controls = {
            i: self.init_control(self.nets_list[i]) for i in range(len(self.nets_list))
        }
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def get_delta_model(self, model_0, model_1):
        """ return a dict: {name: params}
        """
        state_dict = {}
        for name, param0 in model_0.state_dict().items():
            param1 = model_1.state_dict()[name]
            state_dict[name] = param0.detach() - param1.detach()
        return state_dict

    def loc_update(self, priloader_list):
        self.delta_models = {}
        self.delta_controls = {}
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        # self.aggregate_nets(None)
        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
            self.update_local_control(i)

        # self.update_global()

        new_control = self.update_global_control()

        self.global_control = copy.deepcopy(new_control)
        self.aggregate_nets(None)
        return None

    def update_local_control(self, index):
        client_control = self.local_controls[index]
        delta_model = self.delta_models[index]

        new_control = copy.deepcopy(client_control)
        delta_control = copy.deepcopy(client_control)

        for name in self.delta_models[index].keys():
            # c = client_control[name]
            c = self.global_control[name]
            ci = client_control[name]
            delta = delta_model[name]

            new_ci = ci.data - c.data + delta.data / (self.cnt * self.local_lr)
            new_control[name].data = new_ci
            delta_control[name].data = ci.data - new_ci

        self.local_controls[index] = copy.deepcopy(new_control)
        self.delta_controls[index] = copy.deepcopy(delta_control)

    def update_global_control(self):
        new_control = copy.deepcopy(self.global_control)
        for name, c in self.global_control.items():
            mean_ci = []
            for _, delta_control in self.delta_controls.items():
                mean_ci.append(delta_control[name])
            ci = torch.stack(mean_ci).mean(dim=0)
            new_control[name] = c - ci
        return new_control

    def update_global(self):
        state_dict = {}

        for name, param in self.global_net.state_dict().items():
            vs = []
            for client in self.delta_models.keys():
                vs.append(self.delta_models[client][name])
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)
                vs = param - self.global_lr * mean_value
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()
                vs = param - self.global_lr * mean_value
                vs = vs.long()

            state_dict[name] = vs

        self.global_net.load_state_dict(state_dict, strict=True)
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(self.global_net.state_dict())

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        # optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        optimizer = ScaffoldOptimizer(
            net.parameters(),
            lr=self.local_lr,
            weight_decay=1e-5
        )
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        self.cnt=0
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
                optimizer.step(server_control=self.global_control, client_control=self.local_controls[index])
                self.cnt+=1

        delta_model = self.get_delta_model(self.global_net, net)
        self.delta_models[index] = copy.deepcopy(delta_model)
