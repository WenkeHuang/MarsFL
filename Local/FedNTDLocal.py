import torch

from Local.utils.local_methods import LocalMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits

class NTD_Loss(nn.Module):
    """Not-true Distillation Loss"""

    def __init__(self, num_classes=10, tau=3, beta=1):
        super(NTD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta

    def forward(self, logits, targets, dg_logits):
        ce_loss = self.CE(logits, targets)
        ntd_loss = self._ntd_loss(logits, dg_logits, targets)

        loss = ce_loss + self.beta * ntd_loss

        return loss

    def _ntd_loss(self, logits, dg_logits, targets):
        """Not-tue Distillation Loss"""

        # Get smoothed local model prediction
        logits = refine_as_not_true(logits, targets, self.num_classes)
        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        # Get smoothed global model prediction
        with torch.no_grad():
            dg_logits = refine_as_not_true(dg_logits, targets, self.num_classes)
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = (self.tau ** 2) * self.KLDiv(pred_probs, dg_probs)

        return loss
class FedNTDLocal(LocalMethod):
    NAME = 'FedNTDLocal'
    def __init__(self, args, cfg):
        super(FedNTDLocal, self).__init__(args, cfg)
        # self.mu = cfg.Local[self.NAME].mu
        self.tau = cfg.Local[self.NAME].tau
        self.beta = cfg.Local[self.NAME].beta
        self.criterion = NTD_Loss(num_classes=self.cfg.DATASET.n_classes, tau=self.tau, beta=self.beta)

    def loc_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        for i in online_clients_list:  # 遍历循环当前的参与者
            i = 1
            self.train_net(i, nets_list[i], global_net, priloader_list[i])

    def train_net(self, index, net, global_net, train_loader):
        net = net.to(self.device)
        net.train()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        self.criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        global_net = global_net.to(self.device)

        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = net(images)
                with torch.no_grad():
                    global_logits = global_net(images)
                loss = self.criterion(logits, labels, global_logits)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss)
                optimizer.step()

        # import datetime
        #
        # begin_time = datetime.datetime.now()
        # # print(begin_time)
        # net = net.to(self.device)
        # net.train()
        # if self.cfg.OPTIMIZER.type == 'SGD':
        #     optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
        #                           momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        # criterion = nn.CrossEntropyLoss()
        # criterion.to(self.device)
        # iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        # global_net = global_net.to(self.device)
        # global_weight_collector = list(global_net.parameters())
        #
        # for _ in iterator:
        #     for batch_idx, (images, labels) in enumerate(train_loader):
        #         images = images.to(self.device)
        #         labels = labels.to(self.device)
        #         outputs = net(images)
        #         loss = criterion(outputs, labels)
        #         fed_prox_reg = 0.0
        #         for param_index, param in enumerate(net.parameters()):
        #             fed_prox_reg += ((0.01 / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
        #         loss += 0.01 * fed_prox_reg
        #         optimizer.zero_grad()
        #         loss.backward()
        #         iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss)
        #         optimizer.step()
        # end_time = datetime.datetime.now()
        # # print(end_time)
        # print((end_time - begin_time))
        #
        #
        # begin_time = datetime.datetime.now()
        # # print(begin_time)
        # net = net.to(self.device)
        # net.train()
        # if self.cfg.OPTIMIZER.type == 'SGD':
        #     optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
        #                           momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        # for batch_idx, (images, _) in enumerate(train_loader):
        #     images = images.to(self.device)
        #     with torch.no_grad():
        #         global_logits = global_net(images)
        # iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        # for _ in iterator:
        #     for batch_idx, (images, labels) in enumerate(train_loader):
        #         images = images.to(self.device)
        #         labels = labels.to(self.device)
        #         logits = net(images)
        #         loss = criterion(logits, labels) + criterion(logits, labels)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss)
        #         optimizer.step()
        # end_time = datetime.datetime.now()
        # # print(end_time)
        # print((end_time - begin_time))