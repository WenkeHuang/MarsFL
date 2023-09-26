from Aggregations import get_fed_aggregation
from Methods.utils.meta_methods import FederatedMethod
import torch
import copy


class Scaffold(FederatedMethod):
    NAME = 'Scaffold'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(Scaffold, self).__init__(nets_list, client_domain_list, args, cfg)
        self.local_controls = {}
        self.global_control = {}
        self.delta_models = {}
        self.delta_controls = {}
    def init_control(self, model):
        """ a dict type: {name: params}
        """
        control = {
            name: torch.zeros_like(
                p.data
            ).to(self.device) for name, p in model.state_dict().items()
        }
        return control

    def ini(self):
        super().ini()
        self.global_control = self.init_control(self.global_net)
        self.local_controls = {
            i: self.init_control(self.nets_list[i]) for i in range(len(self.nets_list))
        }

    def local_update(self, priloader_list):
        total_clients = list(range(self.cfg.DATASET.parti_num))  # 获取所有参与者
        self.online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()  # 随机选取online的参与者

        self.local_model.loc_update(online_clients_list=self.online_clients_list, nets_list=self.nets_list, global_net=self.global_net,
                                    priloader_list=priloader_list,
                                    local_controls=self.local_controls,
                                    global_control=self.global_control,
                                    delta_models=self.delta_models,
                                    delta_controls=self.delta_controls
                                    )

    def sever_update(self, priloader_list):
        new_control = self.update_global_control(self.global_control,self.delta_controls)
        self.global_control = copy.deepcopy(new_control)
        self.aggregation_weight_list = self.sever_model.sever_update(fed_aggregation=self.fed_aggregation,
                                                                     online_clients_list=self.online_clients_list,
                                                                     priloader_list=priloader_list,
                                                                     client_domain_list=self.client_domain_list,
                                                                     global_net=self.global_net, nets_list=self.nets_list,
                                                                     local_controls=self.local_controls,
                                                                     global_control=self.global_control,
                                                                     delta_models=self.delta_models,
                                                                     delta_controls=self.delta_controls
                                                                     )
    def update_global_control(self,global_control,delta_controls):
        new_control = copy.deepcopy(global_control)
        for name, c in global_control.items():
            mean_ci = []
            for _, delta_control in delta_controls.items():
                mean_ci.append(delta_control[name])
            ci = torch.stack(mean_ci).mean(dim=0)
            new_control[name] = c - ci
        return new_control