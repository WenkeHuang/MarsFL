import torch
import math

from Sever.utils.sever_methods import SeverMethod


def model_global_norm(model):
    squared_sum = 0
    for name, layer in model.named_parameters():
        squared_sum += torch.sum(torch.pow(layer.data, 2))
    return math.sqrt(squared_sum)


def clip_weight_norm(model, clip):
    total_norm = model_global_norm(model)

    max_norm = clip
    clip_coef = max_norm / (total_norm + 1e-6)
    current_norm = total_norm
    if total_norm > max_norm:
        for name, layer in model.named_parameters():
            layer.data.mul_(clip_coef)
        current_norm = model_global_norm(model)

    return current_norm


def dp_noise(param, sigma):
    noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
    return noised_layer


def smooth_model(target_model, sigma):
    for name, param in target_model.state_dict().items():
        param.add_(dp_noise(param, sigma))


class CRFLSever(SeverMethod):
    NAME = 'CRFLSever'

    def __init__(self, args, cfg):
        super(CRFLSever, self).__init__(args, cfg)
        self.param_clip_thres = cfg.Sever[self.NAME].param_clip_thres
        self.epoch_index_weight = cfg.Sever[self.NAME].epoch_index_weight
        self.epoch_index_bias = cfg.Sever[self.NAME].epoch_index_bias
        self.sigma = cfg.Sever[self.NAME].sigma

    def sever_update(self, **kwargs):
        fed_aggregation = kwargs['fed_aggregation']
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        # client_domain_list = kwargs['client_domain_list']
        global_net = kwargs['global_net']
        # nets_list = kwargs['nets_list']
        submit_params_update_dict = kwargs['submit_params_update_dict']
        epoch_index = kwargs['epoch_index']

        # 获取参与者的聚合权重
        freq = fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        # local更新加权聚合
        agg_params_update = dict()
        for name, data in global_net.state_dict().items():
            agg_params_update[name] = torch.zeros_like(data)

        for index, net_id in enumerate(online_clients_list):
            client_params_update = submit_params_update_dict[net_id]
            for name, data in client_params_update.items():
                agg_params_update[name].add_(client_params_update[name] * freq[index])

        for name, data in global_net.state_dict().items():
            update_per_layer = agg_params_update[name]

            data.add_(update_per_layer)

        # clip global_net
        dynamic_thres = epoch_index * self.epoch_index_weight + self.epoch_index_bias
        if dynamic_thres < self.param_clip_thres:
            param_clip_thres = dynamic_thres
        else:
            param_clip_thres = self.param_clip_thres
        clip_weight_norm(global_net, param_clip_thres)

        # smooth_model
        if epoch_index < self.cfg.DATASET.communication_epoch - 1:
            smooth_model(global_net, self.sigma)

        return freq
