from torchvision.datasets import CIFAR10, CIFAR100

from Datasets.federated_dataset.single_domain.utils.single_domain_dataset import SingleDomainDataset
from Datasets.utils.transforms import DeNormalize
from utils.conf import single_domain_data_path
import torchvision.transforms as transforms
from PIL import Image

from torch.utils.data import Dataset
import numpy as np
import os

class TinyImagenet(Dataset):
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.targets = np.concatenate(np.array(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(np.uint8(255 * img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)


        return img, target

class MyTinyImagenet(TinyImagenet):

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyTinyImagenet, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]


        img = Image.fromarray(np.uint8(255 * img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class FedLeaTinyImagenet(SingleDomainDataset):
    NAME = 'fl_tinyimagenet'
    SETTING = 'label_skew'
    N_CLASS = 200

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

        train_dataset = MyTinyImagenet(single_domain_data_path()+'TINYIMG', train=True,
                                  download=False, transform=train_transform)

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        test_dataset = MyTinyImagenet(single_domain_data_path()+'TINYIMG', train=False,
                                  download=False, transform=test_transform)

        self.partition_label_skew_loaders(train_dataset, test_dataset)

    # @staticmethod
    # def get_transform():
    #     transform = transforms.Compose(
    #         [transforms.ToPILImage(), FedLeaTinyImagenet.Nor_TRANSFORM])
    #     return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))
        return transform

