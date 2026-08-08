"""Recover the attention a trained WangCVPGAT used on a given image.

The graph head reduces the CNN feature map to ``num_graph_nodes`` horizontal
bands, so every attention weight maps back to a band of image rows. This module
runs the model, captures the per-layer attention matrices, and reports both the
raw readout weights and the rollout-corrected contribution of each node.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from manometry_models.data import NORMALIZATION_MEAN, NORMALIZATION_STD
from manometry_models.graph_model import WangCVPGAT
from manometry_models.model import ModelConfig, build_model_config, create_model


@dataclass(slots=True)
class GraphAttentionResult:
    """Everything needed to draw an attention overlay for one image."""

    image_path: Path
    display: np.ndarray
    probabilities: np.ndarray
    predicted_index: int
    predicted_class: str
    readout: np.ndarray
    contribution: np.ndarray
    per_layer: list[np.ndarray]
    layer_means: list[np.ndarray]
    rollout: np.ndarray
    band_starts: list[int]
    band_ends: list[int]
    cell: float
    image_size: int

    def band_profile(self, weights: np.ndarray) -> np.ndarray:
        """Spread node weights over their (overlapping) image rows."""
        acc = np.zeros(self.image_size)
        cover = np.zeros(self.image_size)
        for node, weight in enumerate(weights):
            lo = int(round(self.band_starts[node] * self.cell))
            hi = int(round(self.band_ends[node] * self.cell))
            acc[lo:hi] += weight
            cover[lo:hi] += 1.0
        return acc / np.maximum(cover, 1.0)

    def band_center(self, node: int) -> float:
        return (self.band_starts[node] + self.band_ends[node]) / 2 * self.cell


class GraphAttentionExplainer:
    """Loads a graph checkpoint once and explains any number of images."""

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.class_names: list[str] = list(checkpoint["class_names"])
        self.model_name: str = checkpoint.get("model_name", "wang_cvp_gat")
        self.image_size = int(checkpoint.get("image_size", 224))
        self.num_nodes = int(checkpoint.get("graph_num_nodes", 6))

        self.config: ModelConfig = build_model_config(
            self.model_name,
            pretrained=False,
            dropout=float(checkpoint.get("dropout", 0.35)),
            image_size=self.image_size,
            aux_logits=bool(checkpoint.get("aux_logits", False)),
            graph_num_nodes=self.num_nodes,
            graph_temporal_bins=int(checkpoint.get("graph_temporal_bins", 8)),
            graph_hidden_dim=int(checkpoint.get("graph_hidden_dim", 256)),
            graph_num_heads=int(checkpoint.get("graph_num_heads", 4)),
            graph_num_layers=int(checkpoint.get("graph_num_layers", 2)),
            graph_radius=int(checkpoint.get("graph_radius", 2)),
        )
        model = create_model(self.config, num_classes=len(self.class_names))
        if not isinstance(model, WangCVPGAT):
            raise ValueError(f"Checkpoint model '{self.model_name}' is not a graph model; nothing to visualize.")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        self.model = model
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=tuple(checkpoint.get("normalization_mean", NORMALIZATION_MEAN)),
                    std=tuple(checkpoint.get("normalization_std", NORMALIZATION_STD)),
                ),
            ]
        )

        # AdaptiveAvgPool2d maps the feature grid onto num_nodes rows with
        # overlapping windows; mirror that arithmetic to place the bands.
        grid = self.image_size // 32
        self.band_starts = [int(np.floor(i * grid / self.num_nodes)) for i in range(self.num_nodes)]
        self.band_ends = [int(np.ceil((i + 1) * grid / self.num_nodes)) for i in range(self.num_nodes)]
        self.cell = self.image_size / grid

        # The post-softmax attention matrix is exactly the input of
        # attn_dropout, so a plain forward hook recovers it without duplicating
        # the block math.
        self._attention: list[torch.Tensor] = []
        self._nodes: list[torch.Tensor] = []
        for block in self.model.graph_blocks:
            block.attn_dropout.register_forward_hook(self._capture_attention)
            block.register_forward_hook(self._capture_nodes)

    def _capture_attention(self, _module, inputs, _output) -> None:
        self._attention.append(inputs[0].detach().cpu())

    def _capture_nodes(self, _module, _inputs, output) -> None:
        self._nodes.append(output.detach().cpu())

    def explain(self, image_path: Path) -> GraphAttentionResult:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        self._attention.clear()
        self._nodes.clear()

        with Image.open(image_path) as raw:
            raw = raw.convert("RGB")
            display = np.asarray(raw.resize((self.image_size, self.image_size)))
            tensor = self.transform(raw).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        readout_scores = torch.matmul(self._nodes[-1], self.model.readout_query.detach().cpu())
        readout = torch.softmax(readout_scores, dim=1).squeeze(0).numpy()

        # Attention rollout: average over heads, add the residual path, then
        # chain the layers so weights refer back to the *input* nodes.
        rollout = np.eye(self.num_nodes)
        per_layer: list[np.ndarray] = []
        layer_means: list[np.ndarray] = []
        for attention in self._attention:
            heads = attention.squeeze(0).numpy()
            head_mean = heads.mean(axis=0)
            per_layer.append(heads)
            layer_means.append(head_mean)
            residual = 0.5 * head_mean + 0.5 * np.eye(self.num_nodes)
            residual = residual / residual.sum(axis=1, keepdims=True)
            rollout = residual @ rollout

        contribution = readout @ rollout
        contribution = contribution / contribution.sum()
        predicted_index = int(np.argmax(probabilities))

        return GraphAttentionResult(
            image_path=image_path,
            display=display,
            probabilities=probabilities,
            predicted_index=predicted_index,
            predicted_class=self.class_names[predicted_index],
            readout=readout,
            contribution=contribution,
            per_layer=per_layer,
            layer_means=layer_means,
            rollout=rollout,
            band_starts=self.band_starts,
            band_ends=self.band_ends,
            cell=self.cell,
            image_size=self.image_size,
        )
