from Sever.utils.sever_methods import SeverMethod
import copy
import torch

class FedNovaSever(SeverMethod):
    NAME = 'FedNovaSever'

    def __init__(self, args, cfg):
        super(FedNovaSever, self).__init__(args, cfg)

    def sever_update(self, **kwargs):
        fed_aggregation = kwargs['fed_aggregation']
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        # client_domain_list = kwargs['client_domain_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        n_list = kwargs['n_list']
        a_list= kwargs['a_list']
        d_list = kwargs['d_list']

        total_n = sum(n_list)
        # print("total_n:", total_n)
        d_total_round = copy.deepcopy(global_net.state_dict())
        for key in d_total_round:
            d_total_round[key] = 0.0

        for i in range(len(online_clients_list)):
            d_para = d_list[i]
            for key in d_para:
                d_total_round[key] += d_para[key] * n_list[i] / total_n

        coeff = 0.0
        for i in range(len(online_clients_list)):
            coeff = coeff + a_list[i] * n_list[i] / total_n

        updated_model = global_net.state_dict()

        for key in updated_model:
            # print(updated_model[key])
            if updated_model[key].type() == 'torch.LongTensor':
                updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
            elif updated_model[key].type() == 'torch.cuda.LongTensor':
                updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
            else:
                # print(updated_model[key].type())
                # print((coeff*d_total_round[key].type()))
                updated_model[key] -= coeff * d_total_round[key]
        global_net.load_state_dict(updated_model)

        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())

        # freq = fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        # return freq
