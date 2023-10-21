from Local.utils.local_methods import LocalMethod
import torch.optim as optim
from tqdm import tqdm

from Methods.utils.sce_loss import SCELoss


class RHFLLocal(LocalMethod):
    NAME = 'RHFLLocal'

    def __init__(self, args, cfg):
        super(RHFLLocal, self).__init__(args, cfg)
        self.alpha = cfg.Local[self.NAME].alpha
        self.beta = cfg.Local[self.NAME].beta

    def loc_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        priloader_list = kwargs['priloader_list']

        for i in online_clients_list:  # 遍历循环当前的参与者
            self.train_net(i, nets_list[i], priloader_list[i])

    def train_net(self, index, net, train_loader):
        net.train()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        criterion = SCELoss(alpha=self.alpha, beta=self.beta, device=self.device)
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
