from Datasets.utils.federated_dataset import FederatedDataset, TwoCropsTransform
from torchvision.datasets import ImageFolder, DatasetFolder
from Datasets.transforms.denormalization import DeNormalize
import torchvision.transforms as transforms
from utils.conf import data_path
import torch.utils.data as data
from typing import Tuple
from PIL import Image


class ImageFolder_Custom(DatasetFolder):
    def __init__(self, data_name, root, train=True, transform=None, target_transform=None, subset_train_num=7, subset_capacity=10):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.imagefolder_obj = ImageFolder(self.root + 'Office_Caltech_10/' + self.data_name + '/', self.transform, self.target_transform)
        else:
            self.imagefolder_obj = ImageFolder(self.root + 'Office_Caltech_10/' + self.data_name + '/', self.transform, self.target_transform)

        all_data = self.imagefolder_obj.samples
        self.train_index_list = []
        self.test_index_list = []
        for i in range(len(all_data)):
            if i % subset_capacity <= subset_train_num:
                self.train_index_list.append(i)
            else:
                self.test_index_list.append(i)

    def __len__(self):
        if self.train:
            return len(self.train_index_list)
        else:
            return len(self.test_index_list)

    def __getitem__(self, index):

        if self.train:
            used_index_list = self.train_index_list
        else:
            used_index_list = self.test_index_list

        path = self.imagefolder_obj.samples[used_index_list[index]][0]
        target = self.imagefolder_obj.samples[used_index_list[index]][1]
        target = int(target)
        img = self.imagefolder_obj.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class FLOfficeCaltech(FederatedDataset):
    NAME = 'OfficeCaltech'
    SETTING = 'Domain'

    # N_CLASS = 65
    N_CLASS = 10
    # Nor_TRANSFORM = transforms.Compose(
    #     [transforms.Resize((32, 32)),
    #      transforms.RandomCrop(32, padding=4),
    #      transforms.RandomHorizontalFlip(),
    #      transforms.ToTensor(),
    #      transforms.Normalize((0.485, 0.456, 0.406),
    #                           (0.229, 0.224, 0.225))])

    def __init__(self, args, cfg) -> None:
        super().__init__(args, cfg)
        # self.domain_list = ['caltech', 'amazon', 'webcam', 'dslr']
        self.domain_list = ['webcam', 'amazon', 'caltech', 'dslr']
        self.domain_ratio = {'caltech': cfg.DATASET.domain_ratio, 'amazon': cfg.DATASET.domain_ratio,
                             'webcam': cfg.DATASET.domain_ratio, 'dslr': cfg.DATASET.domain_ratio}

        self.train_eval_domain_ratio = {'caltech': cfg.DATASET.train_eval_domain_ratio, 'amazon': cfg.DATASET.train_eval_domain_ratio,
                                        'webcam': cfg.DATASET.train_eval_domain_ratio, 'dslr': cfg.DATASET.train_eval_domain_ratio}

        self.train_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             self.get_normalization_transform()])

        self.test_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
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
        if self.cfg.DATASET.use_two_crop == 'WEAK':
            train_val_transform = TwoCropsTransform(self.train_transform, self.train_transform)
        else:
            train_val_transform = self.train_transform

        for _, domain in enumerate(self.domain_list):
            train_dataset = ImageFolder_Custom(data_name=domain, root=data_path(), train=True,
                                               transform=train_transform)
            test_dataset = ImageFolder_Custom(data_name=domain, root=data_path(), train=False,
                                              transform=self.test_transform)
            train_eval_dataset = ImageFolder_Custom(data_name=domain, root=data_path(), train=False,
                                                    transform=train_val_transform)

            domain_training_dataset_dict[domain] = train_dataset
            domain_testing_dataset_dict[domain] = test_dataset
            domain_train_eval_dataset_dict[domain] = train_eval_dataset

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
