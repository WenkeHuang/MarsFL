from Local.utils.local_methods import LocalMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
class FedLCLocal(LocalMethod):
    NAME = 'FedLCLocal'

    def __init__(self, args, cfg):
        super(FedLCLocal, self).__init__(args, cfg)
        self.tau = cfg.Local[self.NAME].tau
        # self.mu = cfg.Local[self.NAME].mu

    def loc_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        priloader_list = kwargs['priloader_list']
        net_cls_counts = kwargs['net_cls_counts']
        for i in online_clients_list:  # 遍历循环当前的参与者
            self.train_net(i, nets_list[i], priloader_list[i],net_cls_counts)


    def train_net(self, index, net, train_loader,net_cls_counts):
        net = net.to(self.device)
        net.train()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        criterion = self.logit_calibrated_loss
        label_distrib = torch.tensor(list(net_cls_counts[index].values()), device=self.device).float()
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                # if len(images)!=1:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels, label_distrib)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()

    def logit_calibrated_loss(self, logit, y, label_distrib):
        label_distrib[label_distrib == 0] = 1e-5
        cal_logit = torch.exp(
            logit
            - (
                    self.tau
                    * torch.pow(label_distrib, -1 / 4)
                    .unsqueeze(0)
                    .expand((logit.shape[0], -1))
            )
        )

        cal_logit = torch.clip(cal_logit, max=100)

        y_logit = torch.gather(cal_logit, dim=-1, index=y.unsqueeze(1))
        loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
        # a=torch.where(loss == torch.nan)
        # b=torch.where(loss == torch.inf)
        loss_out = loss.sum() / logit.shape[0]
        return loss_out
