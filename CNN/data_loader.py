# import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

# define a function to load CIFAR batch generator
def get_data(cifar_dir, batch_size, augment, validation_split, shuffle=True, num_workers=0, random_seed=1234):
    
    # define a novel transform method
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # define transform method for training set
    # if augment, RandomCrop and RandomHorizontalFlip is added to transform methods
    # if not, the same transform as test set is applied
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # load data from CIFAR 10 according to the tutorial
    train_set = torchvision.datasets.CIFAR10(root=cifar_dir, train=True, download=False, transform=train_transform)
    val_set = torchvision.datasets.CIFAR10(root=cifar_dir, train=True, download=False, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=cifar_dir, train=False, download=False, transform=transform)

    # define validation and training set index for train/val split
    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(validation_split * num_train))
    
    # shuffle batches
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
    # split train and validation set
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    # get train, val, test set generator from Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=len(val_idx), sampler=val_sampler,
        num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=len(test_set), num_workers=num_workers)
    
    # return the generator
    return train_loader, val_loader, test_loader

