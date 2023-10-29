import torch
import torch.nn.functional as F
from Local.utils.local_methods import LocalMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


class FcclPlusLocal(LocalMethod):
    NAME = 'FcclPlusLocal'

    def __init__(self, args, cfg):
        super(FcclPlusLocal, self).__init__(args, cfg)

        self.local_dis_power = cfg.Local[self.NAME].local_dis_power

    def loc_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        priloader_list = kwargs['priloader_list']
        # prev_nets_list = kwargs['prev_nets_list']
        global_net = kwargs['global_net']

        for i in online_clients_list:  # 遍历循环当前的参与者
            self.train_net(i, nets_list[i], global_net, priloader_list[i])

    def train_net(self, index, net, teacher_net, train_loader):
        T = self.local_dis_power

        net = net.to(self.device)
        teacher_net = teacher_net.to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr, weight_decay=1e-5)

        criterionCE = nn.CrossEntropyLoss()
        criterionCE.to(self.device)
        criterionKL = nn.KLDivLoss(reduction='batchmean')
        criterionKL.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                bs, class_num = outputs.shape
                soft_outputs = F.softmax(outputs / T, dim=1)
                non_targets_mask = torch.ones(bs, class_num).to(self.device).scatter_(1, labels.view(-1, 1), 0)
                non_target_soft_outputs = soft_outputs[non_targets_mask.bool()].view(bs, class_num - 1)

                non_target_logsoft_outputs = torch.log(non_target_soft_outputs)

                with torch.no_grad():
                    inter_outputs = teacher_net(images)
                    soft_inter_outpus = F.softmax(inter_outputs / T, dim=1)
                    non_target_soft_inter_outputs = soft_inter_outpus[non_targets_mask.bool()].view(bs, class_num - 1)

                inter_loss = criterionKL(non_target_logsoft_outputs, non_target_soft_inter_outputs)
                loss_hard = criterionCE(outputs, labels)
                inter_loss = inter_loss * (T ** 2)
                loss = loss_hard + inter_loss
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d lossCE = %0.3f lossKD = %0.3f" % (index, loss_hard.item(), inter_loss.item())
                optimizer.step()
