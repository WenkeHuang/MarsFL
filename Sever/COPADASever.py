from Sever.utils.sever_methods import SeverMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
import numpy as np
from itertools import permutations, combinations

class COPADASever(SeverMethod):
    NAME = 'COPADASever'
    def __init__(self, args, cfg):
        super(COPADASever, self).__init__(args, cfg)
        # self.mu = cfg.Local[self.NAME].mu
        # self.temperature_moon = cfg.Local[self.NAME].temperature_moon


    def sever_update(self, **kwargs):
        fed_aggregation = kwargs['fed_aggregation']
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        # client_domain_list = kwargs['client_domain_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        epoch_index = kwargs['epoch_index']
        total_epoch = self.cfg.DATASET.communication_epoch - 1
        out_train_loader = kwargs['out_train_loader']
        head_dict = kwargs['head_dict']
        # acquire Z_list
        Z_list = [0 for _ in range(len(nets_list))]
        for batch_idx, (images, labels) in enumerate(out_train_loader):
            images = images.to(self.device)
            confidences_list = list()
            labels_index_list = list()
            for index, net in enumerate(nets_list):
                p_out = list()
                f = net.features(images)
                outputs = net.cls(f)
                p_out.append(outputs)
                head_ = head_dict[index]
                for k in head_:
                    head = head_[k]
                    out = head(f)
                    p_out.append(out)
                out_final = torch.softmax(torch.mean(torch.stack(p_out), 0), 1)
                confidences, index_1 = torch.max(out_final, 1)
                confidences_list.append(confidences)
                labels_index_list.append(index_1)
            confices, index_1 = torch.max(torch.stack(confidences_list, 1), 1)
            for index in index_1:
                Z_list[index.item()] += 1
        # freq = fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)
        sum_arr = np.sum(Z_list)
        Z_divide = np.divide(Z_list, sum_arr)
        def softmax(x):
            f_x = np.exp(x) / np.sum(np.exp(x))
            return f_x
        freq = softmax(Z_divide)
        # FedAVG 是聚合Bone + cls
        fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                                  global_net=global_net, freq=freq, except_part=['cls'], global_only=True)

        #global update
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(global_net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum,
                                  weight_decay=self.cfg.OPTIMIZER.weight_decay)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))
        global_net.train()

        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(out_train_loader):
                images = images.to(self.device)
                confidences_list = list()
                labels_index_list = list()
                for index,net in enumerate(nets_list):
                    p_out = list()
                    f = net.features(images)
                    outputs = net.cls(f)
                    p_out.append(outputs)
                    head_ = head_dict[index]
                    for k in head_:
                        head = head_[k]
                        out = head(f)
                        p_out.append(out)
                    out_final = torch.softmax(torch.mean(torch.stack(p_out), 0), 1)
                    confidences, index_1 = torch.max(out_final, 1)
                    confidences_list.append(confidences)
                    labels_index_list.append(index_1)
                confices, index_1 = torch.max(torch.stack(confidences_list, 1), 1)
                pseudo_labels = torch.stack(labels_index_list,1)[torch.arange(confidences.size(0)),index_1]
                pseudo_labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, pseudo_labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Global Adaptive %d loss = %0.3f" % (index, loss)
                optimizer.step()

        # 获取参与者的聚合权重
        # freq = fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        # FedAVG 是聚合Bone + cls
        # fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
        #                           global_net=global_net, freq=freq, except_part=[], global_only=False)
        for net in nets_list:
            net.load_state_dict(global_net.state_dict(), strict=False)
        # federated_average(nets_list,self.domain_weight, global_net)


        return freq

