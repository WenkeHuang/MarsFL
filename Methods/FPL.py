from Methods.utils.meta_methods import FederatedMethod


class FPL(FederatedMethod):
    NAME = 'FPL'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(FPL, self).__init__(nets_list, client_domain_list, args, cfg)
        self.global_protos = []
        self.local_protos = {}

    def ini(self):
        super().ini()

    def local_update(self, priloader_list):
        total_clients = list(range(self.cfg.DATASET.parti_num))
        self.online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()

        self.local_model.loc_update(online_clients_list=self.online_clients_list,
                                    nets_list=self.nets_list, global_net=self.global_net,
                                    priloader_list=priloader_list,
                                    global_protos=self.global_protos,
                                    local_protos=self.local_protos, epoch_index=self.epoch_index)

    def sever_update(self, priloader_list):
        self.aggregation_weight_list, self.global_protos = self.sever_model.sever_update(fed_aggregation=self.fed_aggregation,
                                                                                         online_clients_list=self.online_clients_list,
                                                                                         priloader_list=priloader_list,
                                                                                         client_domain_list=self.client_domain_list,
                                                                                         global_net=self.global_net, nets_list=self.nets_list,
                                                                                         local_protos=self.local_protos)
