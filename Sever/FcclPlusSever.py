
import torch
from torch import optim, nn
import torch.nn.functional as F

from Datasets.public_dataset import get_public_dataset
from Sever.utils.sever_methods import SeverMethod


class FcclPlusSever(SeverMethod):
    NAME = 'FcclPlusSever'

    def __init__(self, args, cfg):
        super(FcclPlusSever, self).__init__(args, cfg)

        public_dataset_name = cfg.Sever[self.NAME].public_dataset_name
        pub_len = cfg.Sever[self.NAME].pub_len
        pub_aug = cfg.Sever[self.NAME].pub_aug

        public_batch_size = cfg.Sever[self.NAME].public_batch_size
        self.public_epoch = cfg.Sever[self.NAME].public_epoch
        self.public_dataset = get_public_dataset(args, cfg, public_dataset_name=public_dataset_name,
                                                 pub_len=pub_len, pub_aug=pub_aug, public_batch_size=public_batch_size)
        self.public_dataset.get_data_loaders()
        self.public_loader = self.public_dataset.traindl
        self.public_lr = cfg.Sever[self.NAME].public_lr

        self.dis_power = cfg.Sever[self.NAME].dis_power
        self.temp = cfg.Sever[self.NAME].temp

    def _calculate_isd_sim(self, features):
        sim_q = torch.mm(features, features.T)
        logits_mask = torch.scatter(
            torch.ones_like(sim_q),
            1,
            torch.arange(sim_q.size(0)).view(-1, 1).to(self.device),
            0
        )
        row_size = sim_q.size(0)
        sim_q = sim_q[logits_mask.bool()].view(row_size, -1)
        return sim_q / self.temp

    def _off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def sever_update(self, **kwargs):
        fed_aggregation = kwargs['fed_aggregation']
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        # client_domain_list = kwargs['client_domain_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']

        criterion = torch.nn.L1Loss(reduction='mean')
        criterion.to(self.device)
        for _ in range(self.public_epoch):
            for _, (images, _) in enumerate(self.public_loader):
                '''
                Aggregate the output from participants
                '''

                linear_output_list = []
                linear_output_target_list = []
                logitis_sim_list = []
                logits_sim_target_list = []
                images = images.to(self.device)

                for _, net in enumerate(nets_list):
                    net = net.to(self.device)
                    net.train()
                    linear_output = net(images)
                    linear_output_target_list.append(linear_output.clone().detach())
                    linear_output_list.append(linear_output)
                    features = net.features(images)
                    features = F.normalize(features, dim=1)
                    logits_sim = self._calculate_isd_sim(features)
                    logits_sim_target_list.append(logits_sim.clone().detach())
                    logitis_sim_list.append(logits_sim)

                for net_idx, net in enumerate(nets_list):
                    '''
                    FCCL Loss for overall Network
                    '''
                    optimizer = optim.Adam(net.parameters(), lr=self.public_lr)

                    linear_output = linear_output_list[net_idx]
                    linear_output_target_avg_list = []
                    for k in range(len(online_clients_list)):
                        linear_output_target_avg_list.append(linear_output_target_list[k])
                    linear_output_target_avg = torch.mean(torch.stack(linear_output_target_avg_list), 0)

                    z_1_bn = (linear_output - linear_output.mean(0)) / linear_output.std(0)
                    z_2_bn = (linear_output_target_avg - linear_output_target_avg.mean(0)) / linear_output_target_avg.std(0)
                    c = z_1_bn.T @ z_2_bn
                    c.div_(len(images))

                    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                    off_diag = self._off_diagonal(c).add_(1).pow_(2).sum()
                    fccl_loss = on_diag + 0.0051 * off_diag

                    logits_sim = logitis_sim_list[net_idx]
                    logits_sim_target_avg_list = []
                    for k in range(len(online_clients_list)):
                        logits_sim_target_avg_list.append(logits_sim_target_list[k])
                    logits_sim_target_avg = torch.mean(torch.stack(logits_sim_target_avg_list), 0)

                    inputs = F.log_softmax(logits_sim, dim=1)
                    targets = F.softmax(logits_sim_target_avg, dim=1)
                    loss_distill = F.kl_div(inputs, targets, reduction='batchmean')
                    loss_distill = self.dis_power * loss_distill

                    optimizer.zero_grad()
                    col_loss = fccl_loss + loss_distill

                    col_loss.backward()
                    optimizer.step()

        if self.args.structure == 'homogeneity':

            freq = fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

            fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                                      global_net=global_net, freq=freq, except_part=[], global_only=False)
            return freq
