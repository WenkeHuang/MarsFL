from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms

from Datasets.federated_dataset.single_domain.utils.single_domain_dataset import SingleDomainDataset
from Datasets.utils.transforms import DeNormalize
from utils.conf import multi_domain_data_path, single_domain_data_path
from PIL import Image

from typing import Tuple

import torchvision.transforms as T

from torch.autograd import Variable
import torch.nn.functional as F


class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class FedLeaCIFAR100(SingleDomainDataset):
    NAME = 'fl_cifar100'
    SETTING = 'label_skew'
    N_CLASS = 100

    def __init__(self, args, cfg) -> None:
        super().__init__(args, cfg)
        normalization = self.get_normalization_transform()

        self.weak_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalization])

        self.strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalization])

    def get_data_loaders(self):
        pri_aug = self.cfg.DATASET.aug
        if pri_aug == 'weak':
            train_transform = self.weak_transform
        elif pri_aug == 'strong':
            train_transform = self.strong_transform

        train_dataset = MyCIFAR100(root=single_domain_data_path(), train=True,
                                  download=False, transform=train_transform)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])
        test_dataset = CIFAR100(single_domain_data_path(), train=False,
                               download=False, transform=test_transform)
        self.partition_label_skew_loaders(train_dataset, test_dataset)

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), FedLeaCIFAR100.Nor_TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                              (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                              (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        return transform

