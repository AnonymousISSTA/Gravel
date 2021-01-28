import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from settings import *

import os


class CustomTrainDataset(Dataset):
    """
    dataset with indexed data.
    """
    def __init__(self, dataset_name, indexed=False):
        self.data_root = os.path.join(data_dir, dataset_name)
        self.indexed = indexed
        if dataset_name == 'mnist':
            self.data = datasets.MNIST(
                                        root=self.data_root, train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.5,), (0.5,)),
                                        ]))

        elif dataset_name == 'cifar10':
            self.data = datasets.CIFAR10(
                                        root=self.data_root, train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.Pad(4),
                                            transforms.RandomCrop(32),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ]))

        elif dataset_name == 'cifar100':
            # CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            # CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

            transform_train = transforms.Compose([
                #transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                # transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
            ])

            self.data = torchvision.datasets.CIFAR100(root=self.data_root, train=True, download=True, transform=transform_train)

        elif dataset_name == 'svhn':
            self.data = datasets.SVHN(
                                    root=self.data_root, split='train', download=True,
                                    transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))



    def __getitem__(self, index):
        data, target = self.data[index]
        if self.indexed:
            return data, target, index
        else:
            return data, target
    
    def __len__(self):
        return len(self.data)


class CustomTestDataset(Dataset):
    def __init__(self, dataset_name, indexed=False):
        self.data_root = os.path.join(data_dir, dataset_name)
        self.indexed = indexed
        if dataset_name == 'mnist':
            self.data = datasets.MNIST(
                                        root=self.data_root, train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.5,), (0.5,)),
                                        ]))
        elif dataset_name == 'cifar10':
            self.data = datasets.CIFAR10(
                                        root=self.data_root, train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ]))
        elif dataset_name == 'cifar100':
            # CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            # CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
            ])

            self.data = torchvision.datasets.CIFAR100(root=self.data_root, train=False, download=True, transform=transform_test)

        elif dataset_name == 'svhn':
            self.data = datasets.SVHN(
                                        root=self.data_root, split='test', download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ]))


    def __getitem__(self, index):
        data, target = self.data[index]
        if self.indexed:
            return data, target, index
        else:
            return data, target
    
    def __len__(self):
        return len(self.data)


def get_dataloader(dataset_name, indexed=False):
    train_dataset = CustomTrainDataset(dataset_name, indexed)
    test_dataset = CustomTestDataset(dataset_name, False) # test dataset need not to be indexed.

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader