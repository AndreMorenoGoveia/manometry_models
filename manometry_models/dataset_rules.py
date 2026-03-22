from __future__ import annotations

from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

OFFLINE_AUGMENTATION_PREFIXES = (
    "rotateImage",
    "brightnessE",
    "addGaussianNoise",
    "addSaltAndPepperNoise",
    "resizeImage",
    "saturationE",
    "cesun",
)


def is_supported_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def is_offline_augmented_file(path: str | Path) -> bool:
    filename = Path(path).name
    return filename.startswith(OFFLINE_AUGMENTATION_PREFIXES)

