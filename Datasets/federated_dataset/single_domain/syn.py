
from torchvision.datasets import ImageFolder, DatasetFolder
import torchvision.transforms as transforms

from Datasets.federated_dataset.single_domain.utils.single_domain_dataset import SingleDomainDataset
from Datasets.utils.transforms import DeNormalize

from utils.conf import single_domain_data_path


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


class FLSYN(SingleDomainDataset):
    NAME = 'fl_syn'
    SETTING = 'label_skew'
    N_CLASS = 10

    def __init__(self, args, cfg) -> None:
        super().__init__(args, cfg)

        self.train_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),
             self.get_normalization_transform()
             ]
        )

        self.test_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             self.get_normalization_transform()])

    def get_data_loaders(self):
        pri_aug = self.cfg.DATASET.aug
        if pri_aug == 'weak':
            train_transform = self.train_transform
        elif pri_aug == 'strong':
            train_transform = self.train_transform

        train_dataset = ImageFolder_Custom(data_name='syn', root=single_domain_data_path(), train=True,
                                           transform=train_transform)

        test_dataset = ImageFolder_Custom(data_name='syn', root=single_domain_data_path(), train=False,
                                          transform=self.test_transform)
        self.partition_label_skew_loaders(train_dataset, test_dataset)

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
