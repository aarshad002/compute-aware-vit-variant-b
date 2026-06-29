"""CIFAR-100 dataset loader with clean train/val/test protocol and 224×224 resize.

Protocol
--------
The official CIFAR-100 *train* split (50,000 images) is partitioned, with a
fixed ``split_seed``, into a training subset (default 45,000) and a held-out
validation subset (default 5,000). The official CIFAR-100 *test* split
(10,000 images) is returned as a separate loader and must be used **only** for
final evaluation — never for early stopping or model/threshold selection.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
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
    val_size: int = 5000,
    split_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build clean CIFAR-100 train / val / test DataLoaders.

    The validation subset is carved out of the official *train* split using
    ``split_seed`` (deterministic). The official *test* split is returned
    separately and must only be consumed for final evaluation.

    The training subset uses augmentation; the validation and test subsets use
    the deterministic evaluation transform (no augmentation).

    Args:
        data_dir: Directory containing the CIFAR-100 dataset.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        seed: Random seed for the training sampler/shuffling.
        val_size: Number of images held out from the train split for validation.
        split_seed: Seed governing the deterministic train/val partition.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_transform, val_transform = get_cifar100_transforms()

    # Two views over the official train split: augmented (train) and clean (val).
    train_full = datasets.CIFAR100(
        root=data_dir, train=True, download=False, transform=train_transform
    )
    val_full = datasets.CIFAR100(
        root=data_dir, train=True, download=False, transform=val_transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=False, transform=val_transform
    )

    # Deterministic train/val partition of the official train split.
    num_total = len(train_full)  # 50,000 for CIFAR-100
    if not 0 < val_size < num_total:
        raise ValueError(f"val_size must be in (0, {num_total}); got {val_size}")
    perm = torch.randperm(
        num_total, generator=torch.Generator().manual_seed(split_seed)
    ).tolist()
    val_indices   = perm[:val_size]
    train_indices = perm[val_size:]

    train_subset = Subset(train_full, train_indices)
    val_subset   = Subset(val_full,   val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
