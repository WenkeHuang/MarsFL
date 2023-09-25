from Aggregations import get_fed_aggregation
from Methods.utils.meta_methods import FederatedMethod
import numpy as np
import copy
import torch
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


class FedDC(FederatedMethod):
    NAME = 'FedDC'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(FedDC, self).__init__(nets_list, client_domain_list, args, cfg)
        n_clnt = len(nets_list)
        self.n_par = len(get_mdl_params([nets_list[0]])[0])
        self.max_norm = 10.0
        self.first = True

        self.parameter_drifts = np.zeros((n_clnt, self.n_par)).astype('float32')
        init_par_list = get_mdl_params([self.nets_list[0]], self.n_par)[0]
        self.clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
        self.state_gadient_diffs = np.zeros((n_clnt + 1, self.n_par)).astype('float32')
    def ini(self):
        super().ini()

    def local_update(self, priloader_list):
        total_clients = list(range(self.cfg.DATASET.parti_num))  # 获取所有参与者
        if self.first:
            clients_dl = [priloader_list[i] for i in range(self.cfg.DATASET.parti_num)]
            clients_len = [len(dl.sampler.indices) for dl in clients_dl]
            clients_all = np.sum(clients_len)
            freq = clients_len / clients_all
            self.weight_list = freq * len(self.nets_list)
            self.first = False
        self.online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()  # 随机选取online的参与者
        self.delta_g_sum = np.zeros(self.n_par)
        self.local_model.loc_update(online_clients_list=self.online_clients_list, nets_list=self.nets_list, global_net=self.global_net,
                                    priloader_list=priloader_list,
                                    n_par=self.n_par,
                                    state_gadient_diffs=self.state_gadient_diffs,
                                    weight_list=self.weight_list,
                                    parameter_drifts=self.parameter_drifts,
                                    delta_g_sum = self.delta_g_sum,
                                    clnt_params_list = self.clnt_params_list
                                    )

    def sever_update(self, priloader_list):
        self.aggregation_weight_list = self.sever_model.sever_update(fed_aggregation=self.fed_aggregation,
                                                                     online_clients_list=self.online_clients_list,
                                                                     priloader_list=priloader_list,
                                                                     client_domain_list=self.client_domain_list,
                                                                     global_net=self.global_net, nets_list=self.nets_list,
                                                                     n_par=self.n_par,
                                                                     state_gadient_diffs=self.state_gadient_diffs,
                                                                     weight_list=self.weight_list,
                                                                     parameter_drifts=self.parameter_drifts,
                                                                     delta_g_sum=self.delta_g_sum,
                                                                     clnt_params_list=self.clnt_params_list
                                                                     )
