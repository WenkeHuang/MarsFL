from Aggregations import get_fed_aggregation
from Methods.utils.meta_methods import FederatedMethod
from utils.utils import cal_client_weight, set_requires_grad
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy


class COPADA(FederatedMethod):
    NAME = 'COPADA'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(COPAdA, self).__init__(nets_list, client_domain_list, args, cfg)
        self.head_dict = {}
    def ini(self):
        super().ini()
        # 其他客户端的头
        for i in range(len(self.nets_list)):
            self.head_dict[i] = {}
            for j in range(len(self.nets_list)):
                # 加入其他头 并且不要梯度
                if i != j:
                    head = copy.deepcopy(self.nets_list[j].cls)
                    set_requires_grad(head, False)
                    self.head_dict[i][j] = head

    def local_update(self, priloader_list):
        total_clients = list(range(self.cfg.DATASET.parti_num))  # 获取所有参与者
        self.online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()  # 随机选取online的参与者

        self.local_model.loc_update(online_clients_list=self.online_clients_list, nets_list=self.nets_list, global_net=self.global_net,
                                    priloader_list=priloader_list,head_dict=self.head_dict)

    def sever_update(self, priloader_list):
        # 更新客户端的头
        self.head_dict = {}
        for i in range(len(self.nets_list)):
            self.head_dict[i] = {}
            for j in range(len(self.nets_list)):
                # 加入其他头 并且不要梯度
                if i != j:
                    head = copy.deepcopy(self.nets_list[j].cls)
                    set_requires_grad(head, False)
                    self.head_dict[i][j] = head

        self.aggregation_weight_list = self.sever_model.sever_update(fed_aggregation=self.fed_aggregation,
                                                                     online_clients_list=self.online_clients_list,
                                                                     priloader_list=priloader_list,
                                                                     client_domain_list=self.client_domain_list,
                                                                     global_net=self.global_net,
                                                                     epoch_index=self.epoch_index,
                                                                     nets_list=self.nets_list,
                                                                     out_train_loader = self.out_train_loader,
                                                                     head_dict = self.head_dict
                                                                     )
