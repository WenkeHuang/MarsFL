
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import transforms
from yacs.config import CfgNode as CN
from abc import abstractmethod
from argparse import Namespace
from typing import Tuple
import numpy as np

from Datasets.utils.utils import record_net_data_stats

dataloader_kwargs = {'num_workers': 2, 'pin_memory': True}


class SingleDomainDataset:
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
        # self.train_eval_loaders = {}
        # self.test_loader = {}
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

    def partition_label_skew_loaders(self, train_dataset, test_dataset) -> Tuple[list, DataLoader, dict]:
        n_class = self.N_CLASS
        n_participants = self.cfg.DATASET.parti_num
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

                beta = self.cfg.DATASET.beta
                if beta == 0:
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

            net_dataidx_map[j] = idx_batch[j]
        self.net_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        for j in range(n_participants):
            train_sampler = SubsetRandomSampler(net_dataidx_map[j])
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.cfg.OPTIMIZER.local_train_batch, sampler=train_sampler, num_workers=4, drop_last=True)
            self.train_loaders.append(train_loader)

        test_loader = DataLoader(test_dataset,
                                 batch_size=self.cfg.OPTIMIZER.local_test_batch, shuffle=False, num_workers=4)
        self.test_loader = test_loader