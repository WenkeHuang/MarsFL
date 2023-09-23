from Local.utils.local_methods import LocalMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


class FedDynLocal(LocalMethod):
    NAME = 'FedDynLocal'

    def __init__(self, args, cfg):
        super(FedDynLocal, self).__init__(args, cfg)
        self.reg_lamb = cfg.Local[self.NAME].reg_lamb
        # self.temperature_moon = cfg.Local[self.NAME].temperature_moon
    def loc_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        priloader_list = kwargs['priloader_list']
        client_grads = kwargs['client_grads']
        global_net = kwargs['global_net']
        for i in online_clients_list:  # 遍历循环当前的参与者
            self.train_net(i, nets_list[i], priloader_list[i],client_grads,global_net)

    def train_net(self, index, net, train_loader,client_grads,global_net):
        net = net.to(self.device)
        net.train()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)

        local_grad = copy.deepcopy(client_grads[index])

        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)

                reg_loss = 0.0
                cnt = 0.0
                for name, param in global_net.named_parameters():
                    term1 = (param * (
                            local_grad[name] - global_net.state_dict()[name]
                    )).sum()
                    term2 = (param * param).sum()

                    reg_loss += self.reg_lamb * (term1 + term2)
                    cnt += 1.0

                loss_ce = criterion(outputs, labels)
                loss = loss_ce + reg_loss / cnt
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()

        for name, param in net.named_parameters():
            local_grad[name] += (
                    net.state_dict()[name] - global_net.state_dict()[name]
            )
        client_grads[index] = local_grad
