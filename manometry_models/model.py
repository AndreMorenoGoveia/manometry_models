from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models

from manometry_models.model_registry import (
    DEFAULT_IMAGE_SIZE_BY_MODEL,
    DEFAULT_NORMALIZATION_MEAN,
    DEFAULT_NORMALIZATION_STD,
    IMAGENET_NORMALIZATION_MEAN,
    IMAGENET_NORMALIZATION_STD,
    SUPPORTED_MODEL_NAMES,
)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ManometryCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.35) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)


@dataclass(slots=True, frozen=True)
class ModelConfig:
    model_name: str
    pretrained: bool
    image_size: int
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    dropout: float
    aux_logits: bool


def get_default_image_size(model_name: str) -> int:
    normalized = model_name.lower()
    if normalized not in DEFAULT_IMAGE_SIZE_BY_MODEL:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {', '.join(SUPPORTED_MODEL_NAMES)}")
    return DEFAULT_IMAGE_SIZE_BY_MODEL[normalized]


def build_model_config(
    model_name: str,
    *,
    pretrained: bool,
    dropout: float,
    image_size: int | None = None,
    aux_logits: bool | None = None,
) -> ModelConfig:
    normalized = model_name.lower()
    if normalized not in SUPPORTED_MODEL_NAMES:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {', '.join(SUPPORTED_MODEL_NAMES)}")
    if normalized == "cnn" and pretrained:
        raise ValueError("The custom cnn baseline does not provide pretrained weights.")

    resolved_aux_logits = aux_logits if aux_logits is not None else normalized == "inception_v3"
    resolved_image_size = image_size if image_size is not None else get_default_image_size(normalized)
    normalization_mean = IMAGENET_NORMALIZATION_MEAN if pretrained else DEFAULT_NORMALIZATION_MEAN
    normalization_std = IMAGENET_NORMALIZATION_STD if pretrained else DEFAULT_NORMALIZATION_STD

    return ModelConfig(
        model_name=normalized,
        pretrained=pretrained,
        image_size=resolved_image_size,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        dropout=dropout,
        aux_logits=resolved_aux_logits,
    )


def create_model(config: ModelConfig, num_classes: int) -> nn.Module:
    model_name = config.model_name

    if model_name == "cnn":
        return ManometryCNN(num_classes=num_classes, dropout=config.dropout)

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if config.pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if config.pretrained else None
        model = models.efficientnet_b0(weights=weights)
        last_layer = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last_layer.in_features, num_classes)
        return model

    if model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if config.pretrained else None
        model = models.convnext_tiny(weights=weights)
        last_layer = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last_layer.in_features, num_classes)
        return model

    if model_name == "densenet201":
        weights = models.DenseNet201_Weights.DEFAULT if config.pretrained else None
        model = models.densenet201(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if model_name == "inception_v3":
        weights = models.Inception_V3_Weights.DEFAULT if config.pretrained else None
        model = models.inception_v3(weights=weights, aux_logits=config.aux_logits)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if config.aux_logits and getattr(model, "AuxLogits", None) is not None:
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")
