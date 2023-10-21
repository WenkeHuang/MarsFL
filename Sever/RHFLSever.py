import copy

import numpy as np
import torch
from torch import optim, nn
import torch.nn.functional as F

from Datasets.public_dataset import get_public_dataset
from Methods.utils.sce_loss import SCELoss
from Sever.utils.sever_methods import SeverMethod


class RHFLSever(SeverMethod):
    NAME = 'RHFLSever'

    def __init__(self, args, cfg):
        super(RHFLSever, self).__init__(args, cfg)

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

        self.alpha = cfg.Sever[self.NAME].alpha
        self.beta = cfg.Sever[self.NAME].beta


    def sever_update(self, **kwargs):
        fed_aggregation = kwargs['fed_aggregation']
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        epoch_index = kwargs['epoch_index']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']

        '''
               Calculate Client Confidence with label quality and model performance
               '''
        beta = 0.5
        N_Participants = len(self.nets_list)
        amount_with_quality = [1 / (N_Participants - 1) for i in range(N_Participants)]
        weight_with_quality = []
        quality_list = []
        amount_with_quality_exp = []
        last_mean_loss_list = self.current_mean_loss_list
        self.current_mean_loss_list = []
        for participant_index in range(N_Participants):
            train_loader = self.trainloaders[participant_index]
            participant_loss_list = []
            net = self.nets_list[participant_index]
            net = net.to(self.device)
            criterion = SCELoss(alpha=self.alpha, beta=self.beta, device=self.device)
            criterion.to(self.device)
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                private_loss = criterion(outputs, labels)
                participant_loss_list.append(private_loss.item())
            mean_participant_loss = np.mean(participant_loss_list)
            self.current_mean_loss_list.append(mean_participant_loss)

        if epoch_index > 0:
            for participant_index in range(N_Participants):
                delta_loss = last_mean_loss_list[participant_index] - self.current_mean_loss_list[participant_index]
                quality_list.append(delta_loss / self.current_mean_loss_list[participant_index])
            quality_sum = sum(quality_list)
            for participant_index in range(N_Participants):
                amount_with_quality[participant_index] += beta * quality_list[participant_index] / quality_sum
                amount_with_quality_exp.append(np.exp(amount_with_quality[participant_index]))
            amount_with_quality_sum = sum(amount_with_quality_exp)
            for participant_index in range(N_Participants):
                weight_with_quality.append(amount_with_quality_exp[participant_index] / amount_with_quality_sum)
        else:
            weight_with_quality = [1 / (N_Participants - 1) for i in range(N_Participants)]
        weight_with_quality = torch.tensor(weight_with_quality)

        '''
        Collaborative learning
        '''
        for _, (images, _) in enumerate(self.public_loader):
            '''
            Aggregate the output from participants
            '''
            outputs_list = []
            targets_list = []
            images = images.to(self.device)
            for _, net in enumerate(self.nets_list):
                net = net.to(self.device)
                net.train()
                outputs = net(images)
                target = outputs.clone().detach()
                outputs_list.append(outputs)
                targets_list.append(target)
            criterion = nn.KLDivLoss(reduction='batchmean')
            criterion.to(self.device)
            for net_idx, net in enumerate(self.nets_list):
                optimizer = optim.Adam(net.parameters(), lr=self.public_lr, weight_decay=1e-3)
                optimizer.zero_grad()
                loss = torch.tensor(0)
                for i, net in enumerate(self.nets_list):
                    if i != net_idx:
                        weight_index = weight_with_quality[i]
                        loss_batch_sample = criterion(outputs_list[net_idx], targets_list[i])
                        temp = weight_index * loss_batch_sample
                        loss = loss + temp
                loss.backward()
                optimizer.step()

        if self.args.structure == 'homogeneity':
            # 获取参与者的聚合权重
            freq = fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

            # FedAVG聚合
            fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                                      global_net=global_net, freq=freq, except_part=[], global_only=False)
            return freq
