import copy

from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import transforms
from yacs.config import CfgNode as CN
from torchvision import datasets
from abc import abstractmethod
from argparse import Namespace
from typing import Tuple
import numpy as np

from Datasets.utils.utils import record_net_data_stats

dataloader_kwargs = {'num_workers': 2, 'pin_memory': True}


class MultiDomainDataset:
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

    def partition_domain_loaders(self, client_domain_name_list, domain_training_dataset_dict,
                                 domain_testing_dataset_dict, domain_train_eval_dataset_dict):

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
                y_train = value.train_data_list[:, 1]
            elif key in ['Art', 'Clipart', 'Product', 'Real_World']:
                y_train = value.train_data_list[:, 1]
            elif key in ['caltech', 'labelme', 'pascal', 'sun']:
                y_train = value.train_data_list[:, 1]
            not_used_index_dict[key] = np.arange(len(y_train))
            ini_len_dict[key] = len(y_train)

        # train
        for client_domain_name in client_domain_name_list:

            train_dataset = domain_training_dataset_dict[client_domain_name]

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
                y_train_eval = value.test_data_list[:, 1]
            elif key in ['Art', 'Clipart', 'Product', 'Real_World']:
                y_train_eval = value.test_data_list[:, 1]
            elif key in ['caltech', 'labelme', 'pascal', 'sun']:
                y_train_eval = value.test_data_list[:, 1]
            ini_len_dict[key] = len(y_train_eval)
            not_used_index_dict[key] = np.arange(len(y_train_eval))
        '''
        调用Validation的Dataloader
        '''
        for domain_name, value in domain_train_eval_dataset_dict.items():

            train_eval_dataset = domain_train_eval_dataset_dict[domain_name]

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

            test_dataset = domain_testing_dataset_dict[key]

            test_loader = DataLoader(test_dataset,
                                     batch_size=self.cfg.OPTIMIZER.local_test_batch, shuffle=False, **dataloader_kwargs)
            self.test_loader[key] = test_loader
