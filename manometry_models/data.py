from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NORMALIZATION_MEAN = (0.5, 0.5, 0.5)
NORMALIZATION_STD = (0.5, 0.5, 0.5)


@dataclass(slots=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    class_counts: dict[str, int]


def build_transforms(image_size: int, augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    train_steps: list[transforms.Transform] = [transforms.Resize((image_size, image_size))]
    if augment:
        # Keep online augmentation conservative because the training split already
        # includes offline augmented examples.
        train_steps.append(transforms.RandomAffine(degrees=3, translate=(0.02, 0.02)))
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ]
    )
    return transforms.Compose(train_steps), eval_transform


def count_images_by_class(split_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        counts[class_dir.name] = sum(1 for path in class_dir.rglob("*") if path.is_file())
    return counts


def compute_class_weights(class_counts: dict[str, int], class_names: list[str]) -> torch.Tensor:
    total_samples = sum(class_counts.values())
    num_classes = len(class_names)
    weights = [total_samples / (num_classes * max(class_counts[name], 1)) for name in class_names]
    return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    data_dir: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    augment: bool = False,
) -> DataBundle:
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "Expected data/train, data/val and data/test directories under the selected data directory."
        )

    train_transform, eval_transform = build_transforms(image_size=image_size, augment=augment)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    if train_dataset.classes != val_dataset.classes or train_dataset.classes != test_dataset.classes:
        raise ValueError("Class folders differ across train/val/test splits.")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=train_dataset.classes,
        class_counts=count_images_by_class(train_dir),
    )

