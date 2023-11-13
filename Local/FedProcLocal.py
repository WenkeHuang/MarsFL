import numpy as np
import torch

from Local.utils.local_methods import LocalMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


class FedProcLocal(LocalMethod):
    NAME = 'FedProcLocal'

    def __init__(self, args, cfg):
        super(FedProcLocal, self).__init__(args, cfg)



    def loc_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        nets_list = kwargs['nets_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        global_protos = kwargs['global_protos']
        local_protos=kwargs['local_protos']
        epoch_index=kwargs['epoch_index']

        for i in online_clients_list:
            self.train_net(i, nets_list[i], global_net, priloader_list[i], global_protos,local_protos,epoch_index)

    def train_net(self, index, net, global_net, train_loader,global_protos,local_protos,epoch_index):

        total_epoch = self.cfg.DATASET.communication_epoch - 1
        alpha = 1 - epoch_index / total_epoch

        net = net.to(self.device)
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)

        if len(global_protos) != 0:
            all_global_protos_keys = np.array(list(global_protos.keys()))
            all_f = []
            for protos_key in all_global_protos_keys:
                temp_f = global_protos[protos_key]
                temp_f = torch.cat(temp_f, dim=0).to(self.device)
                all_f.append(temp_f.cpu())

        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        for iter in iterator:
            agg_protos_label = {}
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)
                f = net.features(images)
                outputs = net.classifier(f)
                lossCE = criterion(outputs, labels)

                if len(global_protos) == 0:
                    loss_InfoNCE = 0 * lossCE
                else:
                    i = 0
                    loss_InfoNCE = None
                    for label in labels:
                        if label.item() in global_protos.keys():

                            f_pos = np.array(all_f)[all_global_protos_keys == label.item()][0].to(self.device)

                            f_neg = torch.cat(list(np.array(all_f)[all_global_protos_keys != label.item()])).to(
                                self.device)

                            f_now = f[i].unsqueeze(0)

                            embedding_len = f_pos.shape
                            f_neg = f_neg.unsqueeze(1).view(-1, embedding_len[0])
                            f_pos = f_pos.view(-1, embedding_len[0])
                            f_proto = torch.cat((f_pos, f_neg), dim=0)
                            l = torch.cosine_similarity(f_now, f_proto, dim=1)
                            l = l

                            exp_l = torch.exp(l)
                            exp_l = exp_l.view(1, -1)

                            pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
                            pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
                            pos_mask = pos_mask.view(1, -1)

                            pos_l = exp_l * pos_mask
                            sum_pos_l = pos_l.sum(1)
                            sum_exp_l = exp_l.sum(1)
                            loss_instance = -torch.log(sum_pos_l / sum_exp_l)
                            if loss_InfoNCE is None:
                                loss_InfoNCE = loss_instance
                            else:
                                loss_InfoNCE += loss_instance
                        i += 1
                    loss_InfoNCE = loss_InfoNCE / i
                loss_InfoNCE = loss_InfoNCE

                loss = alpha * loss_InfoNCE + (1 - alpha) * lossCE
                loss.backward()
                iterator.desc = "Local Pariticipant %d CE = %0.3f,InfoNCE = %0.3f" % (index, lossCE, loss_InfoNCE)
                optimizer.step()

                if iter == self.cfg.OPTIMIZER.local_epoch - 1:
                    for i in range(len(labels)):
                        if labels[i].item() in agg_protos_label:
                            agg_protos_label[labels[i].item()].append(f[i, :])
                        else:
                            agg_protos_label[labels[i].item()] = [f[i, :]]

        agg_protos = agg_func(agg_protos_label)
        local_protos[index] = agg_protos
