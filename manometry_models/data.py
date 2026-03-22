from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from manometry_models.dataset_rules import is_offline_augmented_file, is_supported_image_file
from manometry_models.model_registry import DEFAULT_NORMALIZATION_MEAN, DEFAULT_NORMALIZATION_STD

NORMALIZATION_MEAN = DEFAULT_NORMALIZATION_MEAN
NORMALIZATION_STD = DEFAULT_NORMALIZATION_STD


@dataclass(slots=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    class_counts: dict[str, int]
    raw_class_counts: dict[str, int]
    excluded_train_class_counts: dict[str, int]


def build_transforms(
    image_size: int,
    augment: bool,
    normalization_mean: tuple[float, float, float],
    normalization_std: tuple[float, float, float],
) -> tuple[transforms.Compose, transforms.Compose]:
    train_steps: list[transforms.Transform] = [transforms.Resize((image_size, image_size))]
    if augment:
        # Keep online augmentation conservative to preserve the structure of
        # the manometry plots.
        train_steps.append(transforms.RandomAffine(degrees=3, translate=(0.02, 0.02)))
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std),
        ]
    )
    return transforms.Compose(train_steps), eval_transform


def is_allowed_image_file(path: str | Path, include_offline_augmented: bool) -> bool:
    if not is_supported_image_file(path):
        return False
    if include_offline_augmented:
        return True
    return not is_offline_augmented_file(path)


def build_image_validator(include_offline_augmented: bool):
    def validator(path: str) -> bool:
        return is_allowed_image_file(path, include_offline_augmented=include_offline_augmented)

    return validator


def count_images_by_class(split_dir: Path, include_offline_augmented: bool = True) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        counts[class_dir.name] = sum(
            1
            for path in class_dir.rglob("*")
            if path.is_file() and is_allowed_image_file(path, include_offline_augmented=include_offline_augmented)
        )
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
    include_offline_augmented_train: bool = False,
    normalization_mean: tuple[float, float, float] = NORMALIZATION_MEAN,
    normalization_std: tuple[float, float, float] = NORMALIZATION_STD,
) -> DataBundle:
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "Expected data/train, data/val and data/test directories under the selected data directory."
        )

    train_transform, eval_transform = build_transforms(
        image_size=image_size,
        augment=augment,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
    )

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform,
        is_valid_file=build_image_validator(include_offline_augmented=include_offline_augmented_train),
    )
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

    raw_class_counts = count_images_by_class(train_dir, include_offline_augmented=True)
    effective_class_counts = count_images_by_class(
        train_dir,
        include_offline_augmented=include_offline_augmented_train,
    )
    excluded_train_class_counts = {
        class_name: raw_class_counts[class_name] - effective_class_counts[class_name]
        for class_name in raw_class_counts
    }

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=train_dataset.classes,
        class_counts=effective_class_counts,
        raw_class_counts=raw_class_counts,
        excluded_train_class_counts=excluded_train_class_counts,
    )
