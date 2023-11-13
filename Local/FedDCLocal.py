from Local.utils.local_methods import LocalMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
import numpy as np


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


class FedDCLocal(LocalMethod):
    NAME = 'FedDCLocal'

    def __init__(self, args, cfg):
        super(FedDCLocal, self).__init__(args, cfg)
        self.alpha_coef = cfg.Local[self.NAME].alpha_coef
        # self.mu = cfg.Local[self.NAME].mu
        self.max_norm = cfg.Local[self.NAME].max_norm

    def loc_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        n_par = kwargs['n_par']
        state_gadient_diffs = kwargs['state_gadient_diffs']
        weight_list = kwargs['weight_list']
        parameter_drifts = kwargs['parameter_drifts']
        delta_g_sum = kwargs['delta_g_sum']
        clnt_params_list = kwargs['clnt_params_list']
        for i in online_clients_list:
            self.train_net(i, nets_list[i], priloader_list[i], global_net, n_par, state_gadient_diffs, weight_list, parameter_drifts, delta_g_sum, clnt_params_list)

    def train_net(self, index, net, train_loader, global_net, n_par, state_gadient_diffs, weight_list, parameter_drifts, delta_g_sum, clnt_params_list):
        net = net.to(self.device)
        net.train()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)

        ###
        cld_mdl_param = get_mdl_params([global_net], n_par)[0]
        global_model_param = torch.tensor(cld_mdl_param, dtype=torch.float32, device=self.device)

        ###
        local_update_last = state_gadient_diffs[index]  # delta theta_i
        global_update_last = state_gadient_diffs[-1] / weight_list[index]  # delta theta
        alpha = self.alpha_coef / weight_list[index]
        hist_i = torch.tensor(parameter_drifts[index], dtype=torch.float32, device=self.device)  # h_i

        state_update_diff = torch.tensor(-local_update_last + global_update_last, dtype=torch.float32, device=self.device)

        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
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
        curr_model_par = get_mdl_params([net], n_par)[0]
        delta_param_curr = curr_model_par - cld_mdl_param
        parameter_drifts[index] += delta_param_curr
        beta = 1 / self.cfg.OPTIMIZER.local_train_batch / self.cfg.OPTIMIZER.local_train_lr

        state_g = local_update_last - global_update_last + beta * (-delta_param_curr)
        delta_g_cur = (state_g - state_gadient_diffs[index]) * weight_list[index]
        delta_g_sum += delta_g_cur
        state_gadient_diffs[index] = state_g
        clnt_params_list[index] = curr_model_par
