from Sever.utils.sever_methods import SeverMethod
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
import numpy as np
from itertools import permutations, combinations

class KD3ASever(SeverMethod):
    NAME = 'KD3ASever'
    def __init__(self, args, cfg):
        super(KD3ASever, self).__init__(args, cfg)
        # self.mu = cfg.Local[self.NAME].mu
        # self.temperature_moon = cfg.Local[self.NAME].temperature_moon
        self.confidence_gate_begin = cfg.Sever[self.NAME].confidence_gate_begin
        self.confidence_gate_end = cfg.Sever[self.NAME].confidence_gate_end
        self.target_weight = [0, 0]
        self.consensus_focus_dict = {}
        self.domain_weight = list()


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
        confidence_gate = (self.confidence_gate_end - self.confidence_gate_begin) * (
                    epoch_index / total_epoch) + self.confidence_gate_begin
        for i in range(len(priloader_list)):
            self.consensus_focus_dict[i+1] = 0

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
                # labels = labels.to(self.device)
                with torch.no_grad():
                    knowledge_list = [torch.softmax(nets_list[i](images), dim=1).unsqueeze(1) for
                                      i in range(len(nets_list))]
                    knowledge_list = torch.cat(knowledge_list, 1)
                knowledge_list.to(self.device)
                _, consensus_knowledge, consensus_weight = knowledge_vote(knowledge_list, confidence_gate,
                                                                          num_classes=self.cfg.DATASET.n_classes)

                self.target_weight[0] += torch.sum(consensus_weight).item()
                self.target_weight[1] += consensus_weight.size(0)


                lam = np.random.beta(2, 2)
                batch_size = images.size(0)
                index = torch.randperm(batch_size).to(self.device)
                mixed_image = lam * images + (1 - lam) * images[index, :]
                mixed_consensus = lam * consensus_knowledge + (1 - lam) * consensus_knowledge[index, :]
                output_t = global_net(mixed_image)
                output_t = torch.log_softmax(output_t, dim=1)
                task_loss_t = torch.mean(consensus_weight * torch.sum(-1 * mixed_consensus * output_t, dim=1))
                optimizer.zero_grad()
                task_loss_t.backward()
                optimizer.step()
                # Calculate consensus focus
                self.consensus_focus_dict = calculate_consensus_focus(self.consensus_focus_dict, knowledge_list, confidence_gate,
                                                                 len(nets_list), self.cfg.DATASET.n_classes)
                # loss.backward()
                # iterator.desc = "Global  %d loss = %0.1f" % (index, task_loss_t)
        # Consensus Focus Re-weighting
        target_parameter_alpha = self.target_weight[0] / self.target_weight[1]
        target_weight = round(target_parameter_alpha / len(nets_list), 4)
        epoch_domain_weight = []
        source_total_weight = 1 - target_weight
        for i in range(1, len(nets_list)+1):
            epoch_domain_weight.append(self.consensus_focus_dict[i])
        if sum(epoch_domain_weight) == 0:
            epoch_domain_weight = [v + 1e-3 for v in epoch_domain_weight]
        epoch_domain_weight = [round(source_total_weight * v / sum(epoch_domain_weight), 4) for v in
                               epoch_domain_weight]
        epoch_domain_weight.insert(0, target_weight)
        # Update domain weight with moving average
        if epoch_index == 0:
            self.domain_weight = epoch_domain_weight
        else:
            self.domain_weight = update_domain_weight(self.domain_weight, epoch_domain_weight)

        federated_average(nets_list,self.domain_weight, global_net)
        freq = self.domain_weight

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



def federated_average(model_list, coefficient_matrix, global_net,batchnorm_mmd=True):
    """
    :param model_list: a list of all models needed in federated average. [0]: model for target domain,
    [1:-1] model for source domains
    :param coefficient_matrix: the coefficient for each model in federate average, list or 1-d np.array
    :param batchnorm_mmd: bool, if true, we use the batchnorm mmd
    :return model list after federated average
    """

    model_list.insert(0,global_net)
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