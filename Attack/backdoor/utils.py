import copy

import numpy as np
import torch
from torch.utils.data import DataLoader


# 基础的后门攻击
def base_backdoor(cfg, img, target, noise_data_rate):
    if torch.rand(1) < noise_data_rate:
        target = cfg.attack.backdoor.backdoor_label
        for pos_index in range(0, len(cfg.attack.backdoor.trigger_position)):
            pos = cfg.attack.backdoor.trigger_position[pos_index]
            img[pos[0]][pos[1]][pos[2]] = cfg.attack.backdoor.trigger_value[pos_index]
    return img, target


# 语义攻击
def semantic_backdoor(cfg, img, target, noise_data_rate):
    if torch.rand(1) < noise_data_rate:
        if target == cfg.attack.backdoor.semantic_backdoor_label:
            target = cfg.attack.backdoor.backdoor_label

            # img, _ = dataset.__getitem__(used_index)
            img = img + torch.randn(img.size()) * 0.05

    return img, target


# 从数据集后面攻击
def backdoor_attack(args, cfg, client_type, private_dataset, is_train):
    noise_data_rate = cfg.attack.noise_data_rate if is_train else 1.0
    if is_train:
        for client_index in range(cfg.DATASET.parti_num):
            if not client_type[client_index]:

                dataset = copy.deepcopy(private_dataset.train_loaders[client_index].dataset)

                all_targets = []
                all_imgs = []

                for i in range(len(dataset)):
                    img, target = dataset.__getitem__(i)
                    if cfg.attack.backdoor.evils == 'base_backdoor':
                        img, target = base_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)

                    if cfg.attack.backdoor.evils == 'semantic_backdoor':
                        img, target = semantic_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)

                    all_targets.append(target)
                    all_imgs.append(img.numpy())

                new_dataset = BackdoorDataset(all_imgs, all_targets)
                train_sampler = private_dataset.train_loaders[client_index].batch_sampler.sampler

                if args.task == 'label_skew':
                    private_dataset.train_loaders[client_index] = DataLoader(new_dataset, batch_size=cfg.OPTIMIZER.local_train_batch,
                                                                             sampler=train_sampler, num_workers=4, drop_last=True)

    else:

        if args.task == 'label_skew':
            dataset = copy.deepcopy(private_dataset.test_loader.dataset)

            all_targets = []
            all_imgs = []

            for i in range(len(dataset)):
                img, target = dataset.__getitem__(i)
                if cfg.attack.backdoor.evils == 'base_backdoor':
                    img, target = base_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)

                    all_targets.append(target)
                    all_imgs.append(img.numpy())
                elif cfg.attack.backdoor.evils == 'semantic_backdoor':
                    if target == cfg.attack.backdoor.semantic_backdoor_label:
                        img, target = semantic_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), noise_data_rate)
                        all_targets.append(target)
                        all_imgs.append(img.numpy())

                # all_targets.append(target)
                # all_imgs.append(img.numpy())
            new_dataset = BackdoorDataset(all_imgs, all_targets)
            private_dataset.backdoor_test_loader = DataLoader(new_dataset, batch_size=cfg.OPTIMIZER.local_train_batch, num_workers=4)


class BackdoorDataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        self.data = np.array(data)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
