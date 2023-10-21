from Aggregations import get_fed_aggregation
from Methods.utils.meta_methods import FederatedMethod

import copy


class FcclPlus(FederatedMethod):
    NAME = 'FcclPlus'
    COMPATIBILITY = ['homogeneity','heterogeneity']

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(FcclPlus, self).__init__(nets_list, client_domain_list, args, cfg)
        self.prev_nets_list = []

    def ini(self):
        super().ini()
        for _, net in enumerate(self.nets_list):
            self.prev_nets_list.append(copy.deepcopy(net)) # 初始化在loading global之前

    def local_update(self, priloader_list):
        total_clients = list(range(self.cfg.DATASET.parti_num))  # 获取所有参与者
        self.online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()  # 随机选取online的参与者
        self.local_model.loc_update(online_clients_list=self.online_clients_list, nets_list=self.nets_list, global_net=self.global_net,
                                    prev_nets_list=self.prev_nets_list, priloader_list=priloader_list)
        self.copy_nets2_prevnets()

    def sever_update(self, priloader_list):
        self.aggregation_weight_list = self.sever_model.sever_update(fed_aggregation=self.fed_aggregation,
                                                                     online_clients_list=self.online_clients_list,
                                                                     priloader_list=priloader_list,
                                                                     client_domain_list=self.client_domain_list,
                                                                     global_net=self.global_net, nets_list=self.nets_list)
