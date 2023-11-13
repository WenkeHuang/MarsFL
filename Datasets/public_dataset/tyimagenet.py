import os

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

from Datasets.public_dataset.utils.public_dataset import PublicDataset
from Datasets.utils.transforms import DeNormalize, TwoCropsTransform
from utils.conf import single_domain_data_path


class TinyImagenet(Dataset):
    def __init__(self, root: str, train: bool = True, transform: transforms = None,
                 target_transform: transforms = None, download: bool = False) -> None:
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
                      ('train' if self.train else 'val', num + 1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num + 1))))
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

        all_type = np.unique(self.targets)
        used_types = all_type[0:10]
        used_data = []
        used_targets = []
        for i in range(len(used_types)):
            used_type = used_types[i]
            target_place = (np.array(self.targets) == used_type)
            data = self.data[target_place]
            targets = np.array(self.targets)[target_place]

            used_data.append(data)
            used_targets.append(targets)

        self.data = np.concatenate(used_data)
        self.targets = np.concatenate(used_targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(np.uint8(255 * img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class PublicTyimagenet(PublicDataset):
    NAME = 'pub_tyimagenet'

    strong_aug = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.RandomApply([
             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
         ], p=0.8),
         transforms.RandomGrayscale(p=0.2),
         transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4865, 0.4409),
                              (0.2673, 0.2564, 0.2762))])

    weak_aug = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4865, 0.4409),
                              (0.2673, 0.2564, 0.2762))])

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

        train_dataset = MyTinyImagenet(single_domain_data_path() + 'TINYIMG', train=True,
                                       download=False, transform=train_transform)

        self.traindl = self.random_loaders(train_dataset, self.pub_len, self.public_batch_size)
