from Sever.utils.sever_methods import SeverMethod

from utils.utils import cal_client_weight


class BaseGlobal(SeverMethod):
    NAME = 'BaseGlobal'

    def __init__(self, args, cfg):
        super(BaseGlobal, self).__init__(args, cfg)

    def global_update(self, **kwargs):
        fed_aggregation = kwargs['fed_aggregation']
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        client_domain_list = kwargs['client_domain_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']

        # 获取参与者的聚合权重
        freq = fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        client_weight = cal_client_weight(online_clients_list=online_clients_list, client_domain_list=client_domain_list, freq=freq)
        print(client_weight)

        # FedAVG 是聚合Bone + cls
        fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                                  global_net=global_net, freq=freq, except_part=[], global_only=False)
