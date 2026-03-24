from __future__ import annotations

SUPPORTED_MODEL_NAMES = (
    "cnn",
    "wang_cvp_gat",
    "resnet18",
    "efficientnet_b0",
    "convnext_tiny",
    "densenet201",
    "inception_v3",
)

DEFAULT_IMAGE_SIZE_BY_MODEL = {
    "cnn": 224,
    "wang_cvp_gat": 224,
    "resnet18": 224,
    "efficientnet_b0": 224,
    "convnext_tiny": 224,
    "densenet201": 224,
    "inception_v3": 299,
}

DEFAULT_NORMALIZATION_MEAN = (0.5, 0.5, 0.5)
DEFAULT_NORMALIZATION_STD = (0.5, 0.5, 0.5)
IMAGENET_NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
IMAGENET_NORMALIZATION_STD = (0.229, 0.224, 0.225)
