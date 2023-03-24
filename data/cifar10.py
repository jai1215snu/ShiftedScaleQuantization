import os
import torch
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def build_cifar10_data(data_path: str = '', input_size: int = 224, batch_size: int = 64, workers: int = 4,
                        dist_sample: bool = False):
    print('==> Loading Cifar10 Dataset', data_path)
    T_mean = (0.4914, 0.4822, 0.4465)
    T_std = (0.2471, 0.2435, 0.2616)
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=T_mean,
                                     std=T_std)

    transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
    transforms_test = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])
    
    train_dataset = CIFAR10(root=data_path, train=True, download=False, transform=transforms_train)
    val_dataset = CIFAR10(root=data_path, train=False, download=False, transform=transforms_test)

    if dist_sample:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)
    return train_loader, val_loader