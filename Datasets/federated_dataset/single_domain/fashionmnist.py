import torch
from PIL import Image
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms

from Datasets.federated_dataset.single_domain.utils.single_domain_dataset import SingleDomainDataset

from utils.conf import single_domain_data_path


class MyFashionMNIST(torch.utils.data.Dataset):
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
        dataobj = FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)

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


class FLFASHIONMNIST(SingleDomainDataset):
    NAME = 'fl_fashionmnist'
    SETTING = 'label_skew'
    N_CLASS = 10

    def __init__(self, args, cfg) -> None:
        super().__init__(args, cfg)

        self.one_channel_train_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.1307, 0.1307, 0.1307),
                                     (0.3081, 0.3081, 0.3081))])
        self.one_channel_test_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
             transforms.Normalize((0.1307, 0.1307, 0.1307),
                                  (0.3081, 0.3081, 0.3081))])

    def get_data_loaders(self):
        pri_aug = self.cfg.DATASET.aug
        if pri_aug == 'weak':
            train_transform = self.one_channel_train_transform
        elif pri_aug == 'strong':
            train_transform = self.one_channel_train_transform

        train_dataset = MyFashionMNIST(root=single_domain_data_path(), train=True,
                                       download=False, transform=train_transform)
        test_dataset = MyFashionMNIST(root=single_domain_data_path(), train=False,
                                      download=False, transform=self.one_channel_test_transform)
        self.partition_label_skew_loaders(train_dataset, test_dataset)
