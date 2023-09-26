from Sever.utils.sever_methods import SeverMethod
import torch

def euclidean_proj_simplex(v, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and (v >= 0).all():
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = torch.flip(torch.sort(v)[0],dims=(0,))
    cssv = torch.cumsum(u,dim=0)
    # get the number of > 0 components of the optimal solution
    non_zero_vector = torch.nonzero(u * torch.arange(1, n+1) > (cssv - s), as_tuple=False)
    if len(non_zero_vector) == 0:
        rho=0.0
    else:
        rho = non_zero_vector[-1].squeeze()
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w


class AFLSever(SeverMethod):
    NAME = 'AFLSever'

    def __init__(self, args, cfg):
        super(AFLSever, self).__init__(args, cfg)
        self.drfa_gamma = cfg.Sever[self.NAME].drfa_gamma
        # self.alpha_coef = cfg.Local[self.NAME].alpha_coef

    def sever_update(self, **kwargs):
        fed_aggregation = kwargs['fed_aggregation']
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        # client_domain_list = kwargs['client_domain_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        loss_dict = kwargs['loss_dict']
        loss_tensor = torch.zeros(len(online_clients_list))
        for i in online_clients_list:
            loss_tensor[i]=torch.Tensor([loss_dict[i]])


        self.lambda_vector= torch.Tensor([1/len(online_clients_list) for _ in range(len(online_clients_list))])
        lambda_vector = self.lambda_vector
        lambda_vector += self.drfa_gamma * loss_tensor
        lambda_vector=euclidean_proj_simplex(lambda_vector)
        lambda_zeros = lambda_vector <= 1e-3
        if lambda_zeros.sum() > 0:
            lambda_vector[lambda_zeros] = 1e-3
            lambda_vector /= lambda_vector.sum()
        self.lambda_vector = lambda_vector
        freq = lambda_vector.cpu().numpy()
        # print("lambda:",lambda_vector)
        # w_avg=weighted_average_weights(local_weights,global_weight.state_dict(),lambda_vector.to(self.device))
        # self.model.load_state_dict(w_avg)

        # 获取参与者的聚合权重
        # freq = fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        # FedAVG 是聚合Bone + cls
        fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                                  global_net=global_net, freq=freq, except_part=[], global_only=False)
        return freq
