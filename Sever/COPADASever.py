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
        global_net.train()
        if self.cfg.OPTIMIZER.type == 'SGD':
            optimizer = optim.SGD(global_net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
                                  momentum=self.cfg.OPTIMIZER.momentum,
                                  weight_decay=self.cfg.OPTIMIZER.weight_decay)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.cfg.OPTIMIZER.local_epoch))

        for _ in iterator:
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
                torch.stack(confidences_list, 1)

        freq = fed_aggregation.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        fed_aggregation.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                                  global_net=global_net, freq=freq, except_part=[], global_only=False)

        # federated_average(nets_list,self.domain_weight, global_net)

        return freq


def knowledge_vote(knowledge_list, confidence_gate, num_classes):
    """
    :param torch.tensor knowledge_list : recording the knowledge from each source domain model
    :param float confidence_gate: the confidence gate to judge which sample to use
    :return: consensus_confidence,consensus_knowledge,consensus_knowledge_weight
    """
    max_p, max_p_class = knowledge_list.max(2)
    max_conf, _ = max_p.max(1)
    max_p_mask = (max_p > confidence_gate).float()
    consensus_knowledge = torch.zeros(knowledge_list.size(0), knowledge_list.size(2)).to(knowledge_list.device)
    for batch_idx, (p, p_class, p_mask) in enumerate(zip(max_p, max_p_class, max_p_mask)):
        # to solve the [0,0,0] situation
        if torch.sum(p_mask) > 0:
            p = p * p_mask
        for source_idx, source_class in enumerate(p_class):
            consensus_knowledge[batch_idx, source_class] += p[source_idx]
    consensus_knowledge_conf, consensus_knowledge = consensus_knowledge.max(1)
    consensus_knowledge_mask = (max_conf > confidence_gate).float().to(knowledge_list.device)
    consensus_knowledge = torch.zeros(consensus_knowledge.size(0), num_classes).to(knowledge_list.device).scatter_(1,
                                                                                                                   consensus_knowledge.view(
                                                                                                                       -1, 1), 1)
    return consensus_knowledge_conf, consensus_knowledge, consensus_knowledge_mask


def calculate_consensus_focus(consensus_focus_dict, knowledge_list, confidence_gate, source_domain_numbers,
                              num_classes):
    """
    :param consensus_focus_dict: record consensus_focus for each domain
    :param torch.tensor knowledge_list : recording the knowledge from each source domain model
    :param float confidence_gate: the confidence gate to judge which sample to use
    :param source_domain_numbers: the numbers of source domains
    """
    domain_contribution = {frozenset(): 0}
    for combination_num in range(1, source_domain_numbers + 1):
        combination_list = list(combinations(range(source_domain_numbers), combination_num))
        for combination in combination_list:
            consensus_knowledge_conf, consensus_knowledge, consensus_knowledge_mask = knowledge_vote(
                knowledge_list[:, combination, :], confidence_gate, num_classes)
            domain_contribution[frozenset(combination)] = torch.sum(
                consensus_knowledge_conf * consensus_knowledge_mask).item()
    permutation_list = list(permutations(range(source_domain_numbers), source_domain_numbers))
    permutation_num = len(permutation_list)
    for permutation in permutation_list:
        permutation = list(permutation)
        for source_idx in range(source_domain_numbers):
            consensus_focus_dict[source_idx + 1] += (
                                                            domain_contribution[frozenset(
                                                                permutation[:permutation.index(source_idx) + 1])]
                                                            - domain_contribution[
                                                                frozenset(permutation[:permutation.index(source_idx)])]
                                                    ) / permutation_num
    return consensus_focus_dict


def create_domain_weight(source_domain_num):
    global_federated_matrix = [1 / (source_domain_num + 1)] * (source_domain_num + 1)
    return global_federated_matrix


def update_domain_weight(global_domain_weight, epoch_domain_weight, momentum=0.9):
    global_domain_weight = [round(global_domain_weight[i] * momentum + epoch_domain_weight[i] * (1 - momentum), 4)
                            for i in range(len(epoch_domain_weight))]
    return global_domain_weight


def federated_average(model_list, coefficient_matrix, global_net, batchnorm_mmd=True):
    """
    :param model_list: a list of all models needed in federated average. [0]: model for target domain,
    [1:-1] model for source domains
    :param coefficient_matrix: the coefficient for each model in federate average, list or 1-d np.array
    :param batchnorm_mmd: bool, if true, we use the batchnorm mmd
    :return model list after federated average
    """

    model_list.insert(0, global_net)
    if batchnorm_mmd:
        dict_list = [it.state_dict() for it in model_list]
        dict_item_list = [dic.items() for dic in dict_list]
        for key_data_pair_list in zip(*dict_item_list):
            source_data_list = [pair[1] * coefficient_matrix[idx] for idx, pair in
                                enumerate(key_data_pair_list)]
            dict_list[0][key_data_pair_list[0][0]] = sum(source_data_list)
        for model in model_list:
            model.load_state_dict(dict_list[0])
        model_list.pop(0)
    else:
        named_parameter_list = [model.named_parameters() for model in model_list]
        for parameter_list in zip(*named_parameter_list):
            source_parameters = [parameter[1].data.clone() * coefficient_matrix[idx] for idx, parameter in
                                 enumerate(parameter_list)]
            parameter_list[0][1].data = sum(source_parameters)
            for parameter in parameter_list[1:]:
                parameter[1].data = parameter_list[0][1].data.clone()
