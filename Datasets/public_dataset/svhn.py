import torch
import torchvision.transforms as transforms
from torchvision.datasets import SVHN

import torchvision.transforms as T

from Datasets.public_dataset.utils.public_dataset import PublicDataset, GaussianBlur
from Datasets.utils.transforms import DeNormalize, TwoCropsTransform
from utils.conf import single_domain_data_path


class MySVHN(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False, data_name=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        dataobj = SVHN(self.root, 'train', self.transform, self.target_transform, self.download)

        return dataobj

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        img, target = self.dataset[index]

        return img, target


class PublicSVHN(PublicDataset):
    NAME = 'pub_svhn'

    torchvision_normalization = T.Normalize(mean=(0.485, 0.456, 0.406),
                                            std=(0.229, 0.224, 0.225))
    torchvision_denormalization = DeNormalize(mean=(0.485, 0.456, 0.406),
                                              std=(0.229, 0.224, 0.225))

    strong_aug = transforms.Compose(
        [
            transforms.RandomResizedCrop(32),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            torchvision_normalization
        ]
    )

    weak_aug = transforms.Compose(
        [
            # transforms.ToPILImage(),
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            torchvision_normalization
        ]
    )

    def __init__(self, args, cfg, **kwargs) -> None:
        super().__init__(args, cfg, **kwargs)

        self.pub_len = kwargs['pub_len']
        self.public_batch_size = kwargs['public_batch_size']
        self.aug = kwargs['pub_aug']

    def get_data_loaders(self):

        if self.aug == 'two_weak':

            train_transform = TwoCropsTransform(self.weak_aug, self.weak_aug)

        elif self.aug == 'two_strong':

            train_transform = TwoCropsTransform(self.strong_aug, self.strong_aug)

        else:
            train_transform = self.weak_aug

        train_dataset = MySVHN(data_name='syn', root=single_domain_data_path(),
                               transform=train_transform)

        self.traindl = self.random_loaders(train_dataset, self.pub_len, self.public_batch_size)

    @staticmethod
    def get_normalization_transform():
        transform = PublicSVHN.torchvision_normalization
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = PublicSVHN.torchvision_denormalization
        return transform
