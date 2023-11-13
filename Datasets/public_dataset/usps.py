import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import USPS
import torchvision.transforms as T

from Datasets.public_dataset.utils.public_dataset import PublicDataset, GaussianBlur
from Datasets.utils.transforms import DeNormalize, TwoCropsTransform
from utils.conf import single_domain_data_path


class MyUSPS(torch.utils.data.Dataset):
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
        self.data = self.dataset.data

        if hasattr(self.dataset, 'labels'):
            self.targets = self.dataset.labels

        elif hasattr(self.dataset, 'targets'):
            self.targets = self.dataset.targets

        if isinstance(self.targets, torch.Tensor):
            self.targets = self.targets.numpy()
        if isinstance(self.data, torch.Tensor):
            self.data = self.data.numpy()

    def __build_truncated_dataset__(self):
        dataobj = USPS(self.root, self.train, self.transform, self.target_transform, self.download)

        return dataobj

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        img = self.data[index]
        target = self.targets[index]
        if len(self.data.shape) == 4:
            img = Image.fromarray(img, mode='RGB')
        else:
            img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class PublicUSPS(PublicDataset):
    NAME = 'pub_usps'


    def __init__(self, args, cfg, **kwargs) -> None:
        super().__init__(args, cfg,**kwargs)

        self.strong_aug = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.1307, 0.1307, 0.1307),
                                     (0.3081, 0.3081, 0.3081))])
        self.weak_aug = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
             transforms.Normalize((0.1307, 0.1307, 0.1307),
                                  (0.3081, 0.3081, 0.3081))])

        self.pub_len=kwargs['pub_len']
        self.public_batch_size=kwargs['public_batch_size']
        self.aug=kwargs['pub_aug']


    def get_data_loaders(self):

        if self.aug == 'two_weak':
            train_transform = TwoCropsTransform(self.weak_aug, self.weak_aug)

        elif self.aug == 'two_strong':
            train_transform = TwoCropsTransform(self.strong_aug, self.strong_aug)

        else:
            train_transform = self.weak_aug

        train_dataset = MyUSPS(data_name='usps', root=single_domain_data_path(),
                               transform=train_transform)

        self.traindl = self.random_loaders(train_dataset, self.pub_len, self.public_batch_size)
