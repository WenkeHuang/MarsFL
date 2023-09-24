import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch
import numpy as np


# https://github.com/gaoliang13/FedDC/blob/main/utils_methods_FedDC.py
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via Fedavg.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


def set_client_from_params(mdl, params,device):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
        idx += length

    mdl.load_state_dict(dict_param)
    return mdl


class FedDc(FederatedModel):
    NAME = 'feddc'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedDc, self).__init__(nets_list, args, transform)
        self.alpha_coef = args.alpha_coef

        n_clnt = len(nets_list)
        self.n_par = len(get_mdl_params([nets_list[0]])[0])
        self.max_norm = 10.0

        self.parameter_drifts = np.zeros((n_clnt, self.n_par)).astype('float32')
        init_par_list = get_mdl_params([self.nets_list[0]], self.n_par)[0]
        self.clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
        self.state_gadient_diffs = np.zeros((n_clnt + 1, self.n_par)).astype('float32')

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

        # n_clnt=len(self.nets_list)
        # weight_list = np.asarray([len(self.trainloaders[i]) for i in range(n_clnt)])

        clients_dl = [self.trainloaders[i] for i in range(self.args.parti_num)]
        clients_len = [len(dl.sampler.indices) for dl in clients_dl]
        clients_all = np.sum(clients_len)
        freq = clients_len / clients_all

        self.weight_list = freq * len(self.nets_list)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        self.delta_g_sum = np.zeros(self.n_par)

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
        self.aggregate_nets(None)

        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.local_lr, weight_decay=self.args.reg)
        elif self.args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.local_lr, momentum=0.9,
                                  weight_decay=self.args.reg)

        ###
        cld_mdl_param = get_mdl_params([self.global_net], self.n_par)[0]
        global_model_param = torch.tensor(cld_mdl_param, dtype=torch.float32, device=self.device)

        ###
        local_update_last = self.state_gadient_diffs[index]  # delta theta_i
        global_update_last = self.state_gadient_diffs[-1] / self.weight_list[index]  # delta theta
        alpha = self.alpha_coef / self.weight_list[index]
        hist_i = torch.tensor(self.parameter_drifts[index], dtype=torch.float32, device=self.device)  # h_i

        state_update_diff = torch.tensor(-local_update_last + global_update_last, dtype=torch.float32, device=self.device)

        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss_ce = criterion(outputs, labels)

                local_parameter = None
                for param in net.parameters():
                    if not isinstance(local_parameter, torch.Tensor):
                        # Initially nothing to concatenate
                        local_parameter = param.reshape(-1)
                    else:
                        local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

                loss_cp = alpha / 2 * torch.sum((local_parameter - (global_model_param - hist_i)) *
                                                (local_parameter - (global_model_param - hist_i)))
                loss_cg = torch.sum(local_parameter * state_update_diff)

                loss = loss_ce + loss_cp + loss_cg
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()

        ####
        curr_model_par = get_mdl_params([net], self.n_par)[0]
        delta_param_curr = curr_model_par - cld_mdl_param
        self.parameter_drifts[index] += delta_param_curr
        beta = 1 / self.local_batch_size / self.local_lr

        state_g = local_update_last - global_update_last + beta * (-delta_param_curr)
        delta_g_cur = (state_g - self.state_gadient_diffs[index]) * self.weight_list[index]
        self.delta_g_sum += delta_g_cur
        self.state_gadient_diffs[index] = state_g
        self.clnt_params_list[index] = curr_model_par

    def aggregate_nets(self, freq=None):
        nets_list = self.nets_list
        global_net = self.global_net
        online_clients = self.online_clients

        avg_mdl_param_sel = np.mean(self.clnt_params_list[online_clients], axis=0)
        delta_g_cur = 1 / len(nets_list) * self.delta_g_sum
        self.state_gadient_diffs[-1] += delta_g_cur

        cld_mdl_param = avg_mdl_param_sel + np.mean(self.parameter_drifts, axis=0)

        all_model = set_client_from_params(global_net, avg_mdl_param_sel,self.device)
        self.global_net = set_client_from_params(global_net, cld_mdl_param,self.device)

        for _, net in enumerate(nets_list):
            net.load_state_dict(all_model.state_dict())
