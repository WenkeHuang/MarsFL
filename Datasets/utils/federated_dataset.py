import copy

from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import transforms
from yacs.config import CfgNode as CN
from torchvision import datasets
from abc import abstractmethod
from argparse import Namespace
from typing import Tuple
import numpy as np

dataloader_kwargs = {'num_workers': 2, 'pin_memory': True}


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform_left, base_transform_right):
        self.base_transform_left = base_transform_left
        self.base_transform_right = base_transform_right

    def __call__(self, x):
        q = self.base_transform_left(x)
        k = self.base_transform_right(x)
        return [q, k]


class FederatedDataset:
    """
    Federated Learning Setting.
    """
    NAME = None
    SETTING = None  # Label & Domain
    N_CLASS = None

    def __init__(self, args: Namespace, cfg: CN) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loaders = []
        self.train_eval_loaders = {}
        self.test_loader = {}
        self.args = args
        self.cfg = cfg

        # if self.args.OOD != "NONE":
        # self.ood_ratio = cfg.DATASET.ood_ratio

    @abstractmethod
    def get_data_loaders(self, selected_domain_list=[]) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass

    def partition_domain_loaders(self, client_domain_name_list: list, domain_training_dataset_dict: dict,
                                 domain_testing_dataset_dict: dict, domain_train_eval_dataset_dict):

        '''
        Initialize Each Domain Index
        '''
        ini_len_dict = {}
        not_used_index_dict = {}
        for key, value in domain_training_dataset_dict.items():
            if key in ['SVHN']:
                y_train = value.dataset.labels
            elif key in ['SYN']:
                y_train = value.imagefolder_obj.targets
            elif key in ['MNIST', 'USPS']:
                y_train = value.dataset.targets
            elif key in ['photo', 'art_painting', 'cartoon', 'sketch']:
                y_train = value.labels
            elif key in ['caltech', 'amazon', 'webcam', 'dslr']:
                y_train = value.imagefolder_obj.targets
            elif key in ['Art', 'Clipart', 'Product', 'Real World']:
                y_train = value.imagefolder_obj.targets

            not_used_index_dict[key] = np.arange(len(y_train))
            ini_len_dict[key] = len(y_train)

        # train
        for client_domain_name in client_domain_name_list:
            if client_domain_name in 'SYN':
                train_dataset = domain_training_dataset_dict[client_domain_name].imagefolder_obj
            elif client_domain_name in ['MNIST', 'USPS', 'SVHN']:
                train_dataset = domain_training_dataset_dict[client_domain_name].dataset
            elif client_domain_name in ['photo', 'art_painting', 'cartoon', 'sketch']:
                train_dataset = domain_training_dataset_dict[client_domain_name]
            elif client_domain_name in ['caltech', 'amazon', 'webcam', 'dslr']:
                train_dataset = domain_training_dataset_dict[client_domain_name].imagefolder_obj
            elif client_domain_name in ['Art', 'Clipart', 'Product', 'Real World']:
                train_dataset = domain_training_dataset_dict[client_domain_name].imagefolder_obj

            idxs = np.random.permutation(not_used_index_dict[client_domain_name])
            percent = self.domain_ratio[client_domain_name]
            selected_idx = idxs[0:int(percent * ini_len_dict[client_domain_name])]
            not_used_index_dict[client_domain_name] = idxs[int(percent * ini_len_dict[client_domain_name]):]
            train_sampler = SubsetRandomSampler(selected_idx)
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.cfg.OPTIMIZER.local_train_batch, sampler=train_sampler, **dataloader_kwargs)
            self.train_loaders.append(train_loader)

        # train eval
        ini_len_dict = {}
        not_used_index_dict = {}
        for key, value in domain_train_eval_dataset_dict.items():
            if key in ['SVHN']:
                y_train_eval = value.dataset.labels
            elif key in ['SYN']:
                y_train_eval = value.imagefolder_obj.targets
            elif key in ['MNIST', 'USPS']:
                y_train_eval = value.dataset.targets
            elif key in ['photo', 'art_painting', 'cartoon', 'sketch']:
                y_train_eval = value.labels
            elif key in ['caltech', 'amazon', 'webcam', 'dslr']:
                y_train_eval = value.imagefolder_obj.targets
            elif key in ['Art', 'Clipart', 'Product', 'Real World']:
                y_train_eval = value.imagefolder_obj.targets

            ini_len_dict[key] = len(y_train_eval)
            not_used_index_dict[key] = np.arange(len(y_train_eval))
        '''
        调用Validation的Dataloader
        '''
        for domain_name, value in domain_train_eval_dataset_dict.items():
            if domain_name in 'SYN':
                train_eval_dataset = domain_train_eval_dataset_dict[domain_name].imagefolder_obj
            elif domain_name in ['MNIST', 'USPS', 'SVHN']:
                train_eval_dataset = domain_train_eval_dataset_dict[domain_name].dataset
            elif domain_name in ['photo', 'art_painting', 'cartoon', 'sketch']:
                train_eval_dataset = domain_train_eval_dataset_dict[domain_name]
            elif domain_name in ['caltech', 'amazon', 'webcam', 'dslr']:
                train_eval_dataset = domain_train_eval_dataset_dict[domain_name].imagefolder_obj
            elif domain_name in ['Art', 'Clipart', 'Product', 'Real World']:
                train_eval_dataset = domain_train_eval_dataset_dict[domain_name].imagefolder_obj

            idxs = np.random.permutation(not_used_index_dict[domain_name])
            percent = self.train_eval_domain_ratio[domain_name]
            selected_idx = idxs[0:int(percent * ini_len_dict[domain_name])]
            not_used_index_dict[domain_name] = idxs[int(percent * ini_len_dict[domain_name]):]
            train_eval_sampler = SubsetRandomSampler(selected_idx)
            train_eval_loader = DataLoader(train_eval_dataset,
                                           batch_size=self.cfg.OPTIMIZER.val_batch, sampler=train_eval_sampler, **dataloader_kwargs)
            self.train_eval_loaders[domain_name] = train_eval_loader

        '''
        调用Testing的Dataloader
        '''
        for key, value in domain_testing_dataset_dict.items():
            if key in ['SYN']:
                test_dataset = value.imagefolder_obj
            elif key in ['MNIST', 'USPS', 'SVHN']:
                test_dataset = value.dataset
            elif key in ['photo', 'art_painting', 'cartoon', 'sketch']:
                test_dataset = value
            elif key in ['caltech', 'amazon', 'webcam', 'dslr']:
                test_dataset = value.imagefolder_obj
            elif key in ['Art', 'Clipart', 'Product', 'Real World']:
                test_dataset = value.imagefolder_obj

            test_loader = DataLoader(test_dataset,
                                     batch_size=self.cfg.OPTIMIZER.local_test_batch, shuffle=False, **dataloader_kwargs)
            self.test_loader[key] = test_loader

def partition_label_skew_loaders(train_dataset: datasets, test_dataset: datasets,
                                 setting: FederatedDataset) -> Tuple[list, DataLoader, dict]:
    n_class = setting.N_CLASS
    n_participants = setting.args.parti_num
    n_class_sample = setting.N_SAMPLES_PER_Class
    min_size = 0
    min_require_size = 10
    y_train = train_dataset.targets
    N = len(y_train)
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_participants)]
        for k in range(n_class):
            idx_k = [i for i, j in enumerate(y_train) if j == k]
            np.random.shuffle(idx_k)
            if n_class_sample != None:
                idx_k = idx_k[0:n_class_sample * n_participants]
            beta = setting.args.beta
            if beta == 0:  # beta为0，不能用狄利克雷，均分？
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.array_split(idx_k, n_participants))]
            else:
                proportions = np.random.dirichlet(np.repeat(a=beta, repeats=n_participants))
                proportions = np.array([p * (len(idx_j) < N / n_participants) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for j in range(n_participants):
        np.random.shuffle(idx_batch[j])
        if n_class_sample != None:
            idx_batch[j] = idx_batch[j][0:n_class_sample * n_class]
        net_dataidx_map[j] = idx_batch[j]
    net_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    for j in range(n_participants):
        train_sampler = SubsetRandomSampler(net_dataidx_map[j])
        train_loader = DataLoader(train_dataset,
                                  batch_size=setting.args.local_batch_size, sampler=train_sampler, num_workers=4, drop_last=True)
        setting.train_loaders.append(train_loader)

    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.local_batch_size, shuffle=False, num_workers=4)
    setting.test_loader = test_loader

    return setting.train_loaders, setting.test_loader, net_cls_counts


# home数据集
def partition_office_domain_skew_loaders_new(train_datasets: list, test_datasets: list,
                                             setting: FederatedDataset) -> Tuple[list, list]:
    ini_len_dict = {}
    not_used_index_dict = {}
    all_labels_list = []
    for i in range(len(train_datasets)):
        name = train_datasets[i].data_name
        all_train_index = np.array(train_datasets[i].train_index_list)
        if name not in not_used_index_dict:
            not_used_index_dict[name] = np.arange(len(all_train_index))
            ini_len_dict[name] = len(all_train_index)

        all_labels_list.append(np.unique(np.array(train_datasets[i].imagefolder_obj.targets)[all_train_index]))

    all_labels_array = np.array(all_labels_list)

    for index in range(len(test_datasets)):
        test_dataset = test_datasets[index]
        test_loader = DataLoader(test_dataset, batch_size=setting.args.local_batch_size)
        setting.test_loader.append(test_loader)

    for index in range(len(train_datasets)):
        name = train_datasets[index].data_name
        # train_dataset = train_datasets[index].imagefolder_obj

        train_dataset = train_datasets[index]

        idxs = np.random.permutation(not_used_index_dict[name])
        percent = setting.percent_dict[name]
        selected_idx = idxs[0:int(percent * ini_len_dict[name])]

        # 找到使用的label
        all_train_index = np.array(train_datasets[index].train_index_list)
        train_labels = np.array(train_datasets[index].imagefolder_obj.targets)[all_train_index]
        selected_labels = train_labels[selected_idx]

        # 对应的数量
        show_up_num = np.zeros(len(all_labels_array[index]))
        for i in range(len(selected_labels)):
            label = selected_labels[i]
            show_up_num[label] += 1

        # 未使用的label
        not_used_labels = np.where(show_up_num == 0)[0]
        # 更改对应index
        for i in range(len(not_used_labels)):
            not_used_label = not_used_labels[i]
            not_used_label_idx = np.where(train_labels == not_used_label)[0]
            add_index = not_used_label_idx[np.random.randint(len(not_used_label_idx))]

            used_label = train_labels[selected_idx]
            # 随机在出现数>=2的删一个
            prob_del_place = np.where(show_up_num >= 2)[0]
            del_index = np.random.randint(len(prob_del_place))
            del_label = prob_del_place[del_index]

            prob_del_selected = np.where(used_label == del_label)[0]
            del_index_selected = prob_del_selected[np.random.randint(len(prob_del_selected))]
            # 修改index
            selected_idx = selected_idx[selected_idx != selected_idx[del_index_selected]]
            selected_idx = np.append(selected_idx, add_index)

            # 更改数量
            show_up_num[del_label] -= 1
            show_up_num[not_used_label] += 1

        # 未使用部分
        not_select_index = np.array(idxs)
        for i in range(len(selected_idx)):
            not_select_index = not_select_index[not_select_index != selected_idx[i]]

        not_used_index_dict[name] = not_select_index

        train_sampler = SubsetRandomSampler(selected_idx)

        train_loader = DataLoader(train_dataset,
                                  batch_size=setting.args.local_batch_size, sampler=train_sampler)
        setting.train_loaders.append(train_loader)

    return setting.train_loaders, setting.test_loader


def partition_office_domain_skew_loaders(train_datasets: list, test_datasets: list,
                                         setting: FederatedDataset) -> Tuple[list, list]:
    ini_len_dict = {}
    not_used_index_dict = {}
    # all_labels = []
    for i in range(len(train_datasets)):
        name = train_datasets[i].data_name
        if name not in not_used_index_dict:
            all_train_index = np.array(train_datasets[i].train_index_list)
            not_used_index_dict[name] = np.arange(len(all_train_index))
            ini_len_dict[name] = len(all_train_index)

            # all_labels.append(np.unique(np.array(train_datasets[i].imagefolder_obj.targets)[all_train_index]))

    for index in range(len(test_datasets)):
        test_dataset = test_datasets[index]
        test_loader = DataLoader(test_dataset, batch_size=setting.args.local_batch_size)
        setting.test_loader.append(test_loader)

    for index in range(len(train_datasets)):
        name = train_datasets[index].data_name
        train_dataset = train_datasets[index]

        idxs = np.random.permutation(not_used_index_dict[name])

        # use_labels = np.unique(train_dataset.labels[selected_idx])

        percent = setting.percent_dict[name]
        selected_idx = idxs[0:int(percent * ini_len_dict[name])]
        not_used_index_dict[name] = idxs[int(percent * ini_len_dict[name]):]

        train_sampler = SubsetRandomSampler(selected_idx)

        train_loader = DataLoader(train_dataset,
                                  batch_size=setting.args.local_batch_size, sampler=train_sampler)
        setting.train_loaders.append(train_loader)

    return setting.train_loaders, setting.test_loader


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    y_train = np.array(y_train)
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list = []
    for net_id, data in net_cls_counts.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    print('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts
