import numpy as np
import torch

from Sever.utils.sever_methods import SeverMethod


class FedProxGASever(SeverMethod):
    NAME = 'FedProxGASever'

    def __init__(self, args, cfg):
        super(FedProxGASever, self).__init__(args, cfg)
        self.agg_weight = np.ones(self.cfg.DATASET.parti_num) / self.cfg.DATASET.parti_num

        self.base_step_size = cfg.Sever[self.NAME].base_step_size
        self.step_size_decay = self.base_step_size / self.cfg.DATASET.communication_epoch

    def get_local_test_acc(self, net, test_loader):
        total = 0
        top1 = 0
        net.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                if isinstance(images,list):
                    images, labels = images[0].to(self.device), labels.to(self.device)
                else:
                    images, labels = images.to(self.device), labels.to(self.device)
                outputs = net(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                # print(batch_idx,images.size(),labels.size(), max5.size())
                top1 += (labels == max5[:, 0:1]).sum().item()

                total += labels.size(0)

        top1acc = round(100 * top1 / total, 2)
        net.train()
        return top1acc

    def update_weight_by_GA(self, accs_before_agg, accs_after_agg, online_num, epoch_index):
        accs_before_agg = np.array(accs_before_agg)
        accs_after_agg = np.array(accs_after_agg)

        # 准确度差值
        accs_diff = accs_after_agg - accs_before_agg

        # 根据等权重修为基础做修正
        step_size = self.base_step_size - (epoch_index - 1) * self.step_size_decay
        step_size *= np.ones(online_num) / online_num

        norm_gap_array = accs_diff / np.max(np.abs(accs_diff))

        # 更新
        self.agg_weight += norm_gap_array * step_size
        self.agg_weight = np.clip(self.agg_weight, 0, 1)

        self.agg_weight = self.agg_weight / np.sum(self.agg_weight)

        return

    def sever_update(self, **kwargs):
        test_loaders = kwargs['test_loaders']
        train_eval_loaders = kwargs['train_eval_loaders']
        fed_aggregation = kwargs['fed_aggregation']
        online_clients_list = kwargs['online_clients_list']
        # priloader_list = kwargs['priloader_list']
        client_domain_list = kwargs['client_domain_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        epoch_index = kwargs['epoch_index']

        accs_before_agg = []
        for i in online_clients_list:  # 遍历循环当前的参与者
            acc_before_agg = self.get_local_test_acc(nets_list[i], train_eval_loaders[client_domain_list[i]])
            accs_before_agg.append(acc_before_agg)

        # FedAVG 是聚合Bone + cls
        fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                                  global_net=global_net, freq=self.agg_weight, except_part=[], global_only=False)
        accs_after_agg = []
        for i in online_clients_list:
            acc_after_agg = self.get_local_test_acc(global_net, test_loaders[client_domain_list[i]])
            accs_after_agg.append(acc_after_agg)

        self.update_weight_by_GA(accs_before_agg, accs_after_agg, len(online_clients_list), epoch_index)
        return self.agg_weight
