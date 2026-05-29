"""CIFAR-100 dataset loader with train/val splits and 224×224 resize."""

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar100_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, val_transform) for CIFAR-100 at 224×224.

    Returns:
        Tuple of (train_transform, val_transform).
    """
    mean = (0.5071, 0.4865, 0.4409)
    std  = (0.2673, 0.2564, 0.2762)

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, val_transform


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build CIFAR-100 train and val DataLoaders.

    Args:
        data_dir: Directory containing the CIFAR-100 dataset.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        seed: Random seed for the training sampler.

    Returns:
        (train_loader, val_loader)
    """
    train_transform, val_transform = get_cifar100_transforms()

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=False, transform=train_transform
    )
    val_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=False, transform=val_transform
    )

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
