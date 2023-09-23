from Sever.utils.sever_methods import SeverMethod
import copy
import torch
class ScaffoldSever(SeverMethod):
    NAME = 'ScaffoldSever'

    def __init__(self, args, cfg):
        super(ScaffoldSever, self).__init__(args, cfg)

    def sever_update(self, **kwargs):
        fed_aggregation = kwargs['fed_aggregation']
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        # client_domain_list = kwargs['client_domain_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        local_controls = kwargs['local_controls']
        global_control = kwargs['global_control']
        delta_models = kwargs['delta_models']
        delta_controls = kwargs['delta_controls']
        new_control = self.update_global_control(global_control,delta_controls)
        global_control = copy.deepcopy(new_control)


        freq = fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        # FedAVG 是聚合Bone + cls
        fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                                  global_net=global_net, freq=freq, except_part=[], global_only=False)
        return freq


    def update_global_control(self,global_control,delta_controls):
        new_control = copy.deepcopy(global_control)
        for name, c in global_control.items():
            mean_ci = []
            for _, delta_control in delta_controls.items():
                mean_ci.append(delta_control[name])
            ci = torch.stack(mean_ci).mean(dim=0)
            new_control[name] = c - ci
        return new_control