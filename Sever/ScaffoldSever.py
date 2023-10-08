from Sever.utils.sever_methods import SeverMethod
import copy
import torch
class ScaffoldSever(SeverMethod):
    NAME = 'ScaffoldSever'

    def __init__(self, args, cfg):
        super(ScaffoldSever, self).__init__(args, cfg)
        self.global_lr =cfg.Sever[self.NAME].global_lr

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

        self.update_global(global_net,delta_models,nets_list)

        # freq = fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)
        #
        # # FedAVG 是聚合Bone + cls
        # fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
        #                           global_net=global_net, freq=freq, except_part=[], global_only=False)
        # return freq

    def update_global(self,global_net,delta_models,nets_list):
        state_dict = {}

        for name, param in global_net.state_dict().items():
            vs = []
            for client in delta_models.keys():
                vs.append(delta_models[client][name])
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)
                vs = param - self.global_lr * mean_value
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()
                vs = param - self.global_lr * mean_value
                vs = vs.long()

            state_dict[name] = vs

        global_net.load_state_dict(state_dict, strict=True)
        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())
