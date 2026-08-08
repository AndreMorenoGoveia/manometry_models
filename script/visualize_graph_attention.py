"""Overlay WangCVPGAT node attention on top of the HRM image it was computed from.

The graph head reduces the CNN feature map to ``num_graph_nodes`` horizontal
bands and attends over them, so every attention weight maps back to a band of
image rows. This script recovers those weights and paints them on the input.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render graph-attention overlays for a trained wang_cvp_gat checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pt.")
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        required=True,
        help="Image to explain. Repeat the flag for several images.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write the figures.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device. 'auto' picks CUDA, then MPS, then CPU.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.gridspec import GridSpec
    from PIL import Image
    from torchvision import transforms

    from manometry_models.data import NORMALIZATION_MEAN, NORMALIZATION_STD
    from manometry_models.graph_model import WangCVPGAT
    from manometry_models.model import build_model_config, create_model
    from manometry_models.training import resolve_device

    for image_path in args.image:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    class_names = checkpoint["class_names"]
    model_name = checkpoint.get("model_name", "wang_cvp_gat")
    image_size = int(checkpoint.get("image_size", 224))
    num_nodes = int(checkpoint.get("graph_num_nodes", 6))
    model_config = build_model_config(
        model_name,
        pretrained=False,
        dropout=float(checkpoint.get("dropout", 0.35)),
        image_size=image_size,
        aux_logits=bool(checkpoint.get("aux_logits", False)),
        graph_num_nodes=num_nodes,
        graph_temporal_bins=int(checkpoint.get("graph_temporal_bins", 8)),
        graph_hidden_dim=int(checkpoint.get("graph_hidden_dim", 256)),
        graph_num_heads=int(checkpoint.get("graph_num_heads", 4)),
        graph_num_layers=int(checkpoint.get("graph_num_layers", 2)),
        graph_radius=int(checkpoint.get("graph_radius", 2)),
    )
    model = create_model(model_config, num_classes=len(class_names))
    if not isinstance(model, WangCVPGAT):
        raise ValueError(f"Checkpoint model '{model_name}' is not a graph model; nothing to visualize.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    normalization_mean = tuple(checkpoint.get("normalization_mean", NORMALIZATION_MEAN))
    normalization_std = tuple(checkpoint.get("normalization_std", NORMALIZATION_STD))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std),
        ]
    )

    # The post-softmax attention matrix is exactly the input of attn_dropout,
    # so a plain forward hook recovers it without duplicating the block math.
    captured: list[torch.Tensor] = []
    node_states: list[torch.Tensor] = []

    def capture_attention(_module, inputs, _output) -> None:
        captured.append(inputs[0].detach().cpu())

    def capture_nodes(_module, _inputs, output) -> None:
        node_states.append(output.detach().cpu())

    handles = []
    for block in model.graph_blocks:
        handles.append(block.attn_dropout.register_forward_hook(capture_attention))
        handles.append(block.register_forward_hook(capture_nodes))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Nominal image rows behind each node: AdaptiveAvgPool2d maps the 7-row
    # feature grid onto num_nodes rows with overlapping windows.
    grid = image_size // 32
    starts = [int(np.floor(i * grid / num_nodes)) for i in range(num_nodes)]
    ends = [int(np.ceil((i + 1) * grid / num_nodes)) for i in range(num_nodes)]
    cell = image_size / grid

    for image_path in args.image:
        captured.clear()
        node_states.clear()

        with Image.open(image_path) as raw:
            raw = raw.convert("RGB")
            display = np.asarray(raw.resize((image_size, image_size)))
            tensor = transform(raw).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

        predicted = int(torch.argmax(probabilities))
        # Readout weights: recomputed from the final node states, matching
        # WangCVPGAT.forward exactly.
        final_nodes = node_states[-1]
        readout_scores = torch.matmul(final_nodes, model.readout_query.detach().cpu())
        readout = torch.softmax(readout_scores, dim=1).squeeze(0).numpy()

        # Attention rollout: average over heads, add the residual path, then
        # chain the layers so weights refer back to the *input* nodes.
        rollout = np.eye(num_nodes)
        per_layer = []
        layer_means = []
        for attention in captured:
            head_mean = attention.squeeze(0).mean(dim=0).numpy()
            per_layer.append(attention.squeeze(0).numpy())
            layer_means.append(head_mean)
            residual = 0.5 * head_mean + 0.5 * np.eye(num_nodes)
            residual = residual / residual.sum(axis=1, keepdims=True)
            rollout = residual @ rollout
        contribution = readout @ rollout
        contribution = contribution / contribution.sum()

        def band_profile(weights: np.ndarray) -> np.ndarray:
            """Spread node weights over their (overlapping) image rows."""
            acc = np.zeros(image_size)
            cover = np.zeros(image_size)
            for node, weight in enumerate(weights):
                lo = int(round(starts[node] * cell))
                hi = int(round(ends[node] * cell))
                acc[lo:hi] += weight
                cover[lo:hi] += 1.0
            return acc / np.maximum(cover, 1.0)

        figure = plt.figure(figsize=(16.5, 9.0))
        spec = GridSpec(2, 4, figure=figure, height_ratios=[1.0, 0.78], hspace=0.32, wspace=0.45)

        ax = figure.add_subplot(spec[0, 0])
        ax.imshow(display)
        for node in range(num_nodes):
            edge = starts[node] * cell
            ax.axhline(edge, color="white", linewidth=0.8, alpha=0.75)
            ax.text(
                4,
                (starts[node] + ends[node]) / 2 * cell,
                f"n{node}",
                color="white",
                fontsize=9,
                va="center",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5, "edgecolor": "none"},
            )
        ax.set_title(f"entrada {image_size}x{image_size}\n{num_nodes} nos (bandas sobrepostas)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        overlays = (
            (contribution, "contribuicao efetiva\n(rollout x readout)"),
            (readout, "atencao de readout\n(so a selecao final)"),
        )
        for column, (weights, label) in enumerate(overlays, start=1):
            ax = figure.add_subplot(spec[0, column])
            ax.imshow(display)
            profile = band_profile(weights)
            heat = np.repeat(profile[:, None], image_size, axis=1)
            mesh = ax.imshow(heat, cmap="turbo", alpha=0.5, vmin=profile.min(), vmax=profile.max())
            for node, weight in enumerate(weights):
                ax.text(
                    image_size - 6,
                    (starts[node] + ends[node]) / 2 * cell,
                    f"{weight:.3f}",
                    color="white",
                    fontsize=9,
                    ha="right",
                    va="center",
                    bbox={"facecolor": "black", "alpha": 0.5, "pad": 1.5, "edgecolor": "none"},
                )
            figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
            ax.set_title(label, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        ax = figure.add_subplot(spec[0, 3])
        order = np.argsort(probabilities.numpy())[::-1]
        ax.barh(
            [class_names[i] for i in order][::-1],
            [probabilities[i].item() for i in order][::-1],
            color="#4C78A8",
        )
        ax.set_xlim(0, 1)
        ax.set_title(f"predicao: {class_names[predicted]}\n({probabilities[predicted]:.3f})", fontsize=10)
        ax.tick_params(labelsize=8)

        def draw_matrix(ax, matrix, title: str, show_y: bool) -> None:
            mesh = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=float(matrix.max()))
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if matrix[i, j] < 0.01:
                        continue
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="white" if matrix[i, j] < 0.6 * matrix.max() else "black",
                    )
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("no de origem (j)", fontsize=8)
            if show_y:
                ax.set_ylabel("no de destino (i)", fontsize=8)
            ax.set_xticks(range(num_nodes))
            ax.set_yticks(range(num_nodes))
            ax.tick_params(labelsize=7)
            figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)

        panels = [
            (mean, f"camada {index + 1} (media das {len(per_layer[index])} cabecas)")
            for index, mean in enumerate(layer_means)
        ]
        panels.append((rollout, "rollout acumulado\n(quem alimenta quem)"))
        for column, (matrix, title) in enumerate(panels[:3]):
            draw_matrix(figure.add_subplot(spec[1, column]), matrix, title, show_y=column == 0)

        ax = figure.add_subplot(spec[1, 3])
        offsets = np.arange(num_nodes)
        ax.barh(offsets + 0.2, readout, height=0.38, color="#B4B4B4", label="readout")
        ax.barh(offsets - 0.2, contribution, height=0.38, color="#E45756", label="contribuicao efetiva")
        ax.set_yticks(offsets)
        ax.set_yticklabels([f"n{i}" for i in range(num_nodes)])
        ax.invert_yaxis()
        ax.set_xlabel("peso", fontsize=8)
        ax.set_title("peso por no", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="lower right")

        figure.suptitle(
            f"{model_name} - {image_path.name} (raio do grafo = {model_config.graph_radius})",
            fontsize=12,
        )
        output_path = args.output_dir / f"{image_path.stem}_graph_attention.png"
        figure.savefig(output_path, dpi=140, bbox_inches="tight")
        plt.close(figure)

        num_heads = per_layer[0].shape[0]
        heads_figure, axes = plt.subplots(
            len(per_layer),
            num_heads,
            figsize=(2.5 * num_heads, 2.6 * len(per_layer)),
            squeeze=False,
        )
        for layer, attention in enumerate(per_layer):
            for head in range(num_heads):
                ax = axes[layer][head]
                ax.imshow(attention[head], cmap="magma", vmin=0.0, vmax=1.0)
                ax.set_title(f"camada {layer + 1} - cabeca {head + 1}", fontsize=9)
                ax.set_xticks(range(num_nodes))
                ax.set_yticks(range(num_nodes))
                ax.tick_params(labelsize=6)
        heads_figure.suptitle(f"{image_path.name} - atencao por cabeca (linha i le da coluna j)", fontsize=11)
        heads_path = args.output_dir / f"{image_path.stem}_graph_attention_heads.png"
        heads_figure.savefig(heads_path, dpi=140, bbox_inches="tight")
        plt.close(heads_figure)

        print(f"{image_path.name}: {class_names[predicted]} ({probabilities[predicted]:.3f}) -> {output_path}")


if __name__ == "__main__":
    main()
