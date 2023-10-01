import torch
from Datasets.federated_dataset.multi_domain.utils.multi_domain_dataset import MultiDomainDataset
from Datasets.utils.transforms import TwoCropsTransform
from torchvision.datasets import MNIST, SVHN, ImageFolder, DatasetFolder, USPS
from Datasets.utils.transforms import DeNormalize
import torchvision.transforms as transforms
from utils.conf import multi_domain_data_path
import torch.utils.data as data
from typing import Tuple
from PIL import Image
class MyDigits(data.Dataset):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False, data_name=None) -> None:
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
        if self.data_name == 'MNIST':
            dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        elif self.data_name == 'USPS':
            dataobj = USPS(self.root, self.train, self.transform, self.target_transform, self.download)
        elif self.data_name == 'SVHN':
            if self.train:
                dataobj = SVHN(self.root, 'train', self.transform, self.target_transform, self.download)
            else:
                dataobj = SVHN(self.root, 'test', self.transform, self.target_transform, self.download)
        return dataobj

    def __getitem__(self, index: int) -> Tuple[type(Image), int]:
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
    def __len__(self):
        return len(self.dataset)


class ImageFolder_Custom(DatasetFolder):
    def __init__(self, data_name, root, train=True, transform=None, target_transform=None):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.imagefolder_obj = ImageFolder(self.root + self.data_name + '/train/', self.transform, self.target_transform)
        else:
            self.imagefolder_obj = ImageFolder(self.root + self.data_name + '/val/', self.transform, self.target_transform)

        self.paths = [self.imagefolder_obj.samples[index][0] for index in range(len(self.imagefolder_obj.samples))]
        self.targets = [self.imagefolder_obj.samples[index][1] for index in range(len(self.imagefolder_obj.samples))]

    def __getitem__(self, index):
        path = self.paths[index]
        target = self.targets[index]
        target = int(target)
        img = self.imagefolder_obj.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imagefolder_obj.samples)


class FLDigits(MultiDomainDataset):
    NAME = 'Digits'
    SETTING = 'Domain'
    N_CLASS = 10

    def __init__(self, args, cfg) -> None:
        super().__init__(args, cfg)
        self.domain_list = ['MNIST', 'USPS', 'SVHN', 'SYN']
        self.domain_ratio = {'MNIST': cfg.DATASET.domain_ratio, 'USPS': cfg.DATASET.domain_ratio,
                             'SVHN': cfg.DATASET.domain_ratio, 'SYN': cfg.DATASET.domain_ratio}

        self.train_eval_domain_ratio = {'MNIST': cfg.DATASET.train_eval_domain_ratio, 'USPS': cfg.DATASET.train_eval_domain_ratio,
                                        'SVHN': cfg.DATASET.train_eval_domain_ratio, 'SYN': cfg.DATASET.train_eval_domain_ratio}
        # 56 32
        # self.train_transform = utils.Compose(
        #     [utils.Resize((56, 56)),
        #      utils.RandomCrop(56, padding=4),
        #      utils.ToTensor(),
        #      self.get_normalization_transform()
        #      ]
        # )
        #
        # self.one_channel_train_transform = utils.Compose(
        #     [utils.Resize((56, 56)),
        #      utils.RandomCrop(56, padding=4),
        #      utils.ToTensor(),
        #      utils.Lambda(lambda x: x.repeat(3, 1, 1)),
        #      self.get_normalization_transform()
        #      ]
        # )
        #
        # self.strong_transform = utils.Compose(
        #     [utils.Resize((56, 56)),
        #      utils.RandomCrop(56, padding=4),
        #      utils.RandAugment(num_ops=1,magnitude=1),
        #      utils.ToTensor(),
        #      self.get_normalization_transform()
        #      ]
        # )
        #
        # self.one_channel_strong_transform = utils.Compose(
        #     [utils.Resize((56, 56)),
        #      utils.RandomCrop(56, padding=4),
        #      utils.RandAugment(num_ops=1,magnitude=1),
        #      utils.ToTensor(),
        #      utils.Lambda(lambda x: x.repeat(3, 1, 1)),
        #      self.get_normalization_transform()
        #      ]
        # )
        #
        # self.test_transform = utils.Compose(
        #     [utils.Resize((56, 56)),
        #      utils.ToTensor(),
        #      self.get_normalization_transform()])
        #
        # self.one_channel_test_transform = utils.Compose(
        #     [utils.Resize((56, 56)),
        #      utils.ToTensor(),
        #      utils.Lambda(lambda x: x.repeat(3, 1, 1)),
        #      self.get_normalization_transform()])
        self.train_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),
             self.get_normalization_transform()
             ]
        )

        self.one_channel_train_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
             self.get_normalization_transform()
             ]
        )

        self.strong_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.RandomCrop(32, padding=4),
             transforms.RandAugment(num_ops=1, magnitude=1),
             transforms.ToTensor(),
             self.get_normalization_transform()
             ]
        )

        self.one_channel_strong_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.RandomCrop(32, padding=4),
             transforms.RandAugment(num_ops=1, magnitude=1),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
             self.get_normalization_transform()
             ]
        )

        self.test_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             self.get_normalization_transform()])

        self.one_channel_test_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
             self.get_normalization_transform()])

    def get_data_loaders(self, selected_domain_list=[]):

        client_domain_name_list = self.domain_list if selected_domain_list == [] else selected_domain_list

        '''
        Loading the default four domains datasets
        '''
        domain_training_dataset_dict = {}

        domain_testing_dataset_dict = {}
        domain_train_eval_dataset_dict = {}

        train_transform = self.train_transform
        one_channel_train_transform = self.one_channel_train_transform

        # if self.cfg.DATASET.aug == 'ASY':
        #     # 构造非对称aug
        #     train_val_transform = TwoCropsTransform(self.strong_transform, self.train_transform)
        #     one_channel_train_val_transform = TwoCropsTransform(self.one_channel_strong_transform,
        #                                                            self.one_channel_train_transform)
        if self.cfg.DATASET.aug == 'two_weak':
            # 构造双弱aug
            train_val_transform = TwoCropsTransform(self.train_transform, self.train_transform)
            one_channel_train_val_transform = TwoCropsTransform(self.one_channel_train_transform,
                                                                self.one_channel_train_transform)
        elif self.cfg.DATASET.aug == 'two_strong':
            # 构造双强aug
            train_val_transform = TwoCropsTransform(self.strong_transform, self.strong_transform)
            one_channel_train_val_transform = TwoCropsTransform(self.one_channel_strong_transform,
                                                                self.one_channel_strong_transform)
        elif self.cfg.DATASET.aug == 'weak':
            train_val_transform = self.train_transform
            one_channel_train_val_transform = self.one_channel_train_transform

        for _, domain in enumerate(self.domain_list):
            if domain == 'SYN':
                domain_training_dataset_dict[domain] = ImageFolder_Custom(data_name=domain, root=multi_domain_data_path(), train=True,
                                                                          transform=train_transform)
                domain_testing_dataset_dict[domain] = ImageFolder_Custom(data_name=domain, root=multi_domain_data_path(), train=False,
                                                                         transform=self.test_transform)

                domain_train_eval_dataset_dict[domain] = ImageFolder_Custom(data_name=domain, root=multi_domain_data_path(), train=False,
                                                                            transform=train_val_transform)

            elif domain in ['MNIST', 'USPS']:
                domain_training_dataset_dict[domain] = MyDigits(multi_domain_data_path(), train=True, download=True,
                                                                transform=one_channel_train_transform, data_name=domain)

                domain_testing_dataset_dict[domain] = MyDigits(multi_domain_data_path(), train=False, download=True,
                                                               transform=self.one_channel_test_transform, data_name=domain)

                domain_train_eval_dataset_dict[domain] = MyDigits(multi_domain_data_path(), train=False, download=True,
                                                                  transform=one_channel_train_val_transform, data_name=domain)

            elif domain == 'SVHN':
                domain_training_dataset_dict[domain] = MyDigits(multi_domain_data_path(), train=True,
                                                                download=True, transform=train_transform, data_name=domain)
                domain_testing_dataset_dict[domain] = MyDigits(multi_domain_data_path(), train=False,
                                                               download=True, transform=self.test_transform, data_name=domain)

                domain_train_eval_dataset_dict[domain] = MyDigits(multi_domain_data_path(), train=False, download=True,
                                                                  transform=train_val_transform, data_name=domain)

        self.partition_domain_loaders(client_domain_name_list, domain_training_dataset_dict, domain_testing_dataset_dict, domain_train_eval_dataset_dict)

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
