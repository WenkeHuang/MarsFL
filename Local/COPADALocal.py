from Local.utils.local_methods import LocalMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


class COPADALocal(LocalMethod):
    NAME = 'COPADALocal'

    def __init__(self, args, cfg):
        super(COPADALocal, self).__init__(args, cfg)

    def loc_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        priloader_list = kwargs['priloader_list']
        head_dict = kwargs['head_dict']

        for i in online_clients_list:  # 遍历循环当前的参与者
            self.train_net(i, nets_list[i], priloader_list[i],head_dict)

    def train_net(self, index, net, train_loader,head_dict):
        net.train()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                f = net.features(images)
                outputs = net.cls(f)
                loss = criterion(outputs, labels)

                # 其他头的loss
                loss_other = 0
                head_ = head_dict[index]
                for k in head_:
                    head = head_[k]
                    out = head(f)
                    loss_other += criterion(out, labels)

                loss = loss + loss_other
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
