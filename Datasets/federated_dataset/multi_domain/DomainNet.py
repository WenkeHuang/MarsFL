from Datasets.federated_dataset.multi_domain.utils.multi_domain_dataset import MultiDomainDataset
from Datasets.utils.transforms import TwoCropsTransform
from Datasets.utils.transforms import DeNormalize
import torchvision.transforms as transforms
from utils.conf import multi_domain_data_path
import torch.utils.data as data
from PIL import Image
import os

split_dict = {
    'train': 'train',
    'val': 'crossval',
    'test': 'test',
}


class MyDomainNet(data.Dataset):
    def __init__(self, root, train='train', transform=None,
                 target_transform=None, data_name=None) -> None:
        # self.not_aug_transform = utils.Compose([utils.ToTensor()])
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.dataset = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        self.split_file = os.path.join(self.root,  f'{self.data_name}_{split_dict[self.train]}.txt')
        self.imgs, self.labels = MyDomainNet.read_txt(self.split_file, self.root)

    @staticmethod
    def read_txt(txt_path, root_path):
        imgs = []
        labels = []
        with open(txt_path, 'r') as f:
            txt_component = f.readlines()
        for line_txt in txt_component:
            line_txt = line_txt.replace('\n', '')
            line_txt = line_txt.split(' ')
            imgs.append(os.path.join(root_path, line_txt[0]))
            labels.append(int(line_txt[1]))
        return imgs, labels

    def __getitem__(self, index):
        img_path = self.imgs[index]
        target = self.labels[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


class FLDomainNet(MultiDomainDataset):
    NAME = 'DomainNet'
    SETTING = 'Domain'
    N_CLASS = 345

    def __init__(self, args, cfg) -> None:
        super().__init__(args, cfg)
        self.domain_list = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        self.domain_ratio = {'clipart': cfg.DATASET.domain_ratio, 'painting': cfg.DATASET.domain_ratio, 'infograph': cfg.DATASET.domain_ratio,
                             'quickdraw': cfg.DATASET.domain_ratio, 'real': cfg.DATASET.domain_ratio, 'sketch': cfg.DATASET.domain_ratio}

        self.train_eval_domain_ratio = {'clipart': cfg.DATASET.train_eval_domain_ratio, 'infograph': cfg.DATASET.train_eval_domain_ratio,
                                        'painting': cfg.DATASET.train_eval_domain_ratio, 'quickdraw': cfg.DATASET.train_eval_domain_ratio,
                                        'real': cfg.DATASET.train_eval_domain_ratio, 'sketch': cfg.DATASET.train_eval_domain_ratio}

        self.train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
             transforms.RandomHorizontalFlip(),
             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
             transforms.RandomGrayscale(),
             transforms.ToTensor(),
             self.get_normalization_transform()])

        self.test_transform = transforms.Compose(
            [transforms.Resize([224, 224]),
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
        # ood_dataset = None
        # 是否使用两个数据相同的数据增强
        if self.cfg.DATASET.use_two_crop:
            train_transform = TwoCropsTransform(self.train_transform, self.train_transform)

            # 构造非对称aug
            train_val_transform = TwoCropsTransform(self.test_transform, self.train_transform)
        else:
            train_transform = self.train_transform
            train_val_transform = self.train_transform

        for _, domain in enumerate(self.domain_list):
            domain_training_dataset_dict[domain] = MyDomainNet(multi_domain_data_path() + 'DomainNet/', train='train', transform=train_transform, data_name=domain)

            domain_testing_dataset_dict[domain] = MyDomainNet(multi_domain_data_path() + 'DomainNet/', train='test', transform=self.test_transform, data_name=domain)

            domain_train_eval_dataset_dict[domain] = MyDomainNet(multi_domain_data_path() + 'DomainNet/', train='test', transform=train_val_transform, data_name=domain)

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
