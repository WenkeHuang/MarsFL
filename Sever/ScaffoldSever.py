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



        freq = fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        # FedAVG 是聚合Bone + cls
        fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                                  global_net=global_net, freq=freq, except_part=[], global_only=False)
        return freq


