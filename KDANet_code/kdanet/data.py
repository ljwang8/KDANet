from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_dataloaders(config: Dict):
    data_cfg = config["data"]
    root = Path(data_cfg["root"])
    image_size = int(data_cfg["image_size"])
    num_workers = int(data_cfg.get("num_workers", 4))
    batch_size = int(config["train"]["batch_size"])
    test_batch_size = int(config.get("test", {}).get("batch_size", batch_size))

    train_dataset = datasets.ImageFolder(
        root / data_cfg.get("train_dir", "train"),
        transform=build_transforms(image_size, train=True),
    )
    val_dataset = datasets.ImageFolder(
        root / data_cfg.get("val_dir", "val"),
        transform=build_transforms(image_size, train=False),
    )
    test_dataset = datasets.ImageFolder(
        root / data_cfg.get("test_dir", "test"),
        transform=build_transforms(image_size, train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
