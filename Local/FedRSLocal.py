from Local.utils.local_methods import LocalMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch


class FedRSLocal(LocalMethod):
    NAME = 'FedRSLocal'

    def __init__(self, args, cfg):
        super(FedRSLocal, self).__init__(args, cfg)
        self.alpha = cfg.Local[self.NAME].alpha

    def loc_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        priloader_list = kwargs['priloader_list']
        net_cls_counts = kwargs['net_cls_counts']
        for i in online_clients_list:
            self.train_net(i, nets_list[i], priloader_list[i], net_cls_counts)

    def train_net(self, index, net, train_loader, net_cls_counts):
        net.train()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        criterion = nn.CrossEntropyLoss()
        cls_count_dict = net_cls_counts[index]

        cls_counts = torch.tensor(list(cls_count_dict.values()), device=self.device)
        cls_rate = cls_counts / torch.sum(cls_counts)

        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                features = net.features(images)

                ws = net.cls.weight

                cdist = cls_rate / cls_rate.max()
                cdist = cdist * (1.0 - self.alpha) + self.alpha
                cdist = cdist.reshape((1, -1))

                logits = cdist * features.mm(ws.transpose(0, 1))

                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
