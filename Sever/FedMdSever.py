
import torch
from torch import optim

from Datasets.public_dataset import get_public_dataset
from Sever.utils.sever_methods import SeverMethod


class FedMdSever(SeverMethod):
    NAME = 'FedMdSever'

    def __init__(self, args, cfg):
        super(FedMdSever, self).__init__(args, cfg)

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
                # outputs_list = []
                targets_list = []
                images = images.to(self.device)
                with torch.no_grad():
                    for _, net in enumerate(nets_list):
                        net = net.to(self.device)
                        net.train()
                        outputs = net(images)
                        target = outputs.clone().detach()
                        # outputs_list.append(outputs)
                        targets_list.append(target)

                target = torch.mean(torch.stack(targets_list), 0)
                for net_idx, net in enumerate(nets_list):
                    optimizer = optim.SGD(net.parameters(), lr=self.public_lr, weight_decay=1e-5)

                    optimizer.zero_grad()
                    outputs = net(images)
                    loss = criterion(outputs, target)
                    loss.backward()
                    optimizer.step()

        if self.args.structure == 'homogeneity':
            freq = fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

            fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                                      global_net=global_net, freq=freq, except_part=[], global_only=False)
            return freq

