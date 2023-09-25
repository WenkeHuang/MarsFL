from Sever.utils.sever_methods import SeverMethod
import numpy as np
import copy
import torch

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

class FedDCSever(SeverMethod):
    NAME = 'FedDCSever'

    def __init__(self, args, cfg):
        super(FedDCSever, self).__init__(args, cfg)

    def sever_update(self, **kwargs):
        fed_aggregation = kwargs['fed_aggregation']
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        # client_domain_list = kwargs['client_domain_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        clnt_params_list = kwargs['clnt_params_list']
        delta_g_sum = kwargs['delta_g_sum']
        state_gadient_diffs = kwargs['state_gadient_diffs']
        parameter_drifts = kwargs['parameter_drifts']



        avg_mdl_param_sel = np.mean(clnt_params_list[online_clients_list], axis=0)
        delta_g_cur = 1 / len(nets_list) * delta_g_sum
        state_gadient_diffs[-1] += delta_g_cur

        cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0)

        all_model = set_client_from_params(global_net, avg_mdl_param_sel,self.device)
        global_net = set_client_from_params(global_net, cld_mdl_param,self.device)

        for _, net in enumerate(nets_list):
            net.load_state_dict(all_model.state_dict())

        # 获取参与者的聚合权重
        freq = fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        return freq
