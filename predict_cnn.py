from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained manometry model checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pt.")
    parser.add_argument("--image", type=Path, required=True, help="Path to a single JPG image.")
    parser.add_argument("--top-k", type=int, default=3, help="How many classes to display.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device. 'auto' picks CUDA, then MPS, then CPU.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the prediction as JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import torch
    from PIL import Image
    from torchvision import transforms

    from manometry_models.data import NORMALIZATION_MEAN, NORMALIZATION_STD
    from manometry_models.model import build_model_config, create_model
    from manometry_models.training import resolve_device

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    class_names = checkpoint["class_names"]
    model_name = checkpoint.get("model_name", "cnn")
    image_size = checkpoint.get("image_size", 224)
    dropout = float(checkpoint.get("dropout", 0.35))
    normalization_mean = tuple(checkpoint.get("normalization_mean", NORMALIZATION_MEAN))
    normalization_std = tuple(checkpoint.get("normalization_std", NORMALIZATION_STD))
    aux_logits = bool(checkpoint.get("aux_logits", model_name == "inception_v3"))
    model_config = build_model_config(
        model_name,
        pretrained=False,
        dropout=dropout,
        image_size=image_size,
        aux_logits=aux_logits,
        graph_num_nodes=int(checkpoint.get("graph_num_nodes", 6)),
        graph_temporal_bins=int(checkpoint.get("graph_temporal_bins", 8)),
        graph_hidden_dim=int(checkpoint.get("graph_hidden_dim", 256)),
        graph_num_heads=int(checkpoint.get("graph_num_heads", 4)),
        graph_num_layers=int(checkpoint.get("graph_num_layers", 2)),
        graph_radius=int(checkpoint.get("graph_radius", 2)),
    )
    model = create_model(model_config, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std),
        ]
    )

    with Image.open(args.image) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        top_k = min(args.top_k, len(class_names))
        top_probabilities, top_indices = torch.topk(probabilities, k=top_k)

    predictions = [
        {
            "class_name": class_names[index],
            "probability": float(probability),
        }
        for probability, index in zip(top_probabilities.cpu().tolist(), top_indices.cpu().tolist())
    ]
    result = {
        "image": str(args.image.resolve()),
        "predicted_class": predictions[0]["class_name"],
        "confidence": predictions[0]["probability"],
        "top_k": predictions,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"Image: {result['image']}")
    print(f"Model: {model_name}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Top-k predictions:")
    for rank, prediction in enumerate(predictions, start=1):
        print(f"  {rank}. {prediction['class_name']}: {prediction['probability']:.4f}")


if __name__ == "__main__":
    main()
