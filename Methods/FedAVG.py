from Aggregations import get_fed_aggregation
from Methods.utils.meta_methods import FederatedMethod

import copy


class FedAVG(FederatedMethod):
    NAME = 'FedAVG'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(FedAVG, self).__init__(nets_list, client_domain_list, args, cfg)

    def ini(self):
        super.ini()

    def update(self, priloader_list):
        total_clients = list(range(self.cfg.DATASET.parti_num))  # 获取所有参与者
        online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()  # 随机选取online的参与者

        self.local_model.loc_update(online_clients_list=online_clients_list, nets_list=self.nets_list,
                                    priloader_list=priloader_list)
        self.sever_model.sever_update(fed_aggregation=self.fed_aggregation, online_clients_list=online_clients_list,
                                        priloader_list=priloader_list, client_domain_list=self.client_domain_list,
                                        global_net=self.global_net, nets_list=self.nets_list)

        return None
