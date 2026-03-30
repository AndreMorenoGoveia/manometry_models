from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT_DIR = Path("ingestion/extracted_images")
DEFAULT_CHECKPOINT = Path("artifacts/densenet201_pretrained/best_model.pt")
DEFAULT_OUTPUT_DIR = Path("reports/densenet201_extracted_images")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DenseNet predictions for all extracted exam images and generate a report.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing extracted exam image folders. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint used for inference. Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where reports will be written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many classes to include in each prediction record.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device. 'auto' picks CUDA, then MPS, then CPU.",
    )
    return parser


def iter_images(input_dir: Path) -> list[Path]:
    supported_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_suffixes
    )


def load_model(checkpoint_path: Path, device_name: str):
    import torch
    from manometry_models.data import NORMALIZATION_MEAN, NORMALIZATION_STD
    from manometry_models.model import build_model_config, create_model
    from manometry_models.training import resolve_device
    from torchvision import transforms

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    model_name = checkpoint.get("model_name", "cnn")
    image_size = int(checkpoint.get("image_size", 224))
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
    return model, class_names, transform, device, checkpoint


def predict_image(image_path: Path, *, model, class_names: list[str], transform, device, top_k: int) -> dict[str, object]:
    import torch
    from PIL import Image

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        resolved_top_k = min(top_k, len(class_names))
        top_probabilities, top_indices = torch.topk(probabilities, k=resolved_top_k)

    ranked_predictions = [
        {
            "class_name": class_names[index],
            "probability": float(probability),
        }
        for probability, index in zip(top_probabilities.cpu().tolist(), top_indices.cpu().tolist())
    ]
    best_prediction = ranked_predictions[0]
    return {
        "image_path": str(image_path),
        "folder_name": image_path.parent.name,
        "image_name": image_path.name,
        "predicted_class": best_prediction["class_name"],
        "confidence": best_prediction["probability"],
        "top_k": ranked_predictions,
    }


def summarize_predictions(predictions: list[dict[str, object]], class_names: list[str]) -> dict[str, object]:
    predicted_class_counts = Counter(prediction["predicted_class"] for prediction in predictions)
    confidences_by_class: dict[str, list[float]] = defaultdict(list)
    folder_predictions: dict[str, list[dict[str, object]]] = defaultdict(list)

    for prediction in predictions:
        confidences_by_class[prediction["predicted_class"]].append(float(prediction["confidence"]))
        folder_predictions[prediction["folder_name"]].append(prediction)

    overall = {
        "total_images": len(predictions),
        "predicted_class_distribution": {
            class_name: predicted_class_counts.get(class_name, 0) for class_name in class_names
        },
        "mean_confidence_by_predicted_class": {
            class_name: (
                sum(confidences_by_class[class_name]) / len(confidences_by_class[class_name])
                if confidences_by_class[class_name]
                else None
            )
            for class_name in class_names
        },
    }

    by_folder = []
    normalized_class_names = {class_name.lower(): class_name for class_name in class_names}
    for folder_name in sorted(folder_predictions):
        folder_items = folder_predictions[folder_name]
        folder_counts = Counter(item["predicted_class"] for item in folder_items)
        dominant_class, dominant_count = folder_counts.most_common(1)[0]
        expected_class = normalized_class_names.get(folder_name.lower())
        matched_predictions = (
            sum(1 for item in folder_items if item["predicted_class"] == expected_class)
            if expected_class
            else None
        )
        by_folder.append(
            {
                "folder_name": folder_name,
                "total_images": len(folder_items),
                "dominant_prediction": dominant_class,
                "dominant_prediction_count": dominant_count,
                "dominant_prediction_ratio": dominant_count / len(folder_items),
                "mean_confidence": sum(float(item["confidence"]) for item in folder_items) / len(folder_items),
                "predicted_class_distribution": dict(sorted(folder_counts.items())),
                "expected_class_from_folder_name": expected_class,
                "expected_class_match_count": matched_predictions,
            }
        )

    return {"overall": overall, "by_folder": by_folder}


def write_csv(predictions: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "folder_name",
        "image_name",
        "image_path",
        "predicted_class",
        "confidence",
        "top_1",
        "top_1_probability",
        "top_2",
        "top_2_probability",
        "top_3",
        "top_3_probability",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in predictions:
            ranked = prediction["top_k"]
            row = {
                "folder_name": prediction["folder_name"],
                "image_name": prediction["image_name"],
                "image_path": prediction["image_path"],
                "predicted_class": prediction["predicted_class"],
                "confidence": f"{float(prediction['confidence']):.6f}",
            }
            for index in range(3):
                class_key = f"top_{index + 1}"
                prob_key = f"top_{index + 1}_probability"
                if index < len(ranked):
                    row[class_key] = ranked[index]["class_name"]
                    row[prob_key] = f"{float(ranked[index]['probability']):.6f}"
                else:
                    row[class_key] = ""
                    row[prob_key] = ""
            writer.writerow(row)


def write_markdown_report(
    *,
    output_path: Path,
    checkpoint_path: Path,
    input_dir: Path,
    summary: dict[str, object],
) -> None:
    overall = summary["overall"]
    lines = [
        "# DenseNet201 Prediction Report",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Input directory: `{input_dir}`",
        f"- Total images: {overall['total_images']}",
        "",
        "## Overall Distribution",
        "",
        "| Predicted class | Count | Mean confidence |",
        "| --- | ---: | ---: |",
    ]

    for class_name, count in overall["predicted_class_distribution"].items():
        mean_confidence = overall["mean_confidence_by_predicted_class"][class_name]
        rendered_confidence = f"{mean_confidence:.4f}" if mean_confidence is not None else "-"
        lines.append(f"| {class_name} | {count} | {rendered_confidence} |")

    lines.extend(
        [
            "",
            "## Folder Summary",
            "",
            "| Folder | Images | Dominant prediction | Ratio | Mean confidence | Expected class | Match count |",
            "| --- | ---: | --- | ---: | ---: | --- | ---: |",
        ]
    )

    for folder_summary in summary["by_folder"]:
        expected_class = folder_summary["expected_class_from_folder_name"] or "-"
        match_count = (
            str(folder_summary["expected_class_match_count"])
            if folder_summary["expected_class_match_count"] is not None
            else "-"
        )
        lines.append(
            "| {folder_name} | {total_images} | {dominant_prediction} | {dominant_prediction_ratio:.2%} | "
            "{mean_confidence:.4f} | {expected_class} | {match_count} |".format(
                folder_name=folder_summary["folder_name"],
                total_images=folder_summary["total_images"],
                dominant_prediction=folder_summary["dominant_prediction"],
                dominant_prediction_ratio=folder_summary["dominant_prediction_ratio"],
                mean_confidence=folder_summary["mean_confidence"],
                expected_class=expected_class,
                match_count=match_count,
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    image_paths = iter_images(args.input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, class_names, transform, device, checkpoint = load_model(args.checkpoint, args.device)

    predictions: list[dict[str, object]] = []
    for image_path in image_paths:
        predictions.append(
            predict_image(
                image_path,
                model=model,
                class_names=class_names,
                transform=transform,
                device=device,
                top_k=args.top_k,
            )
        )

    summary = summarize_predictions(predictions, class_names)
    metadata = {
        "checkpoint": str(args.checkpoint),
        "model_name": checkpoint.get("model_name", "unknown"),
        "input_dir": str(args.input_dir),
        "device": str(device),
        "top_k": args.top_k,
        "class_names": class_names,
        "total_images": len(predictions),
    }

    predictions_path = args.output_dir / "predictions.json"
    summary_path = args.output_dir / "summary.json"
    csv_path = args.output_dir / "predictions.csv"
    markdown_path = args.output_dir / "report.md"

    predictions_path.write_text(
        json.dumps({"metadata": metadata, "predictions": predictions}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps({"metadata": metadata, "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(predictions, csv_path)
    write_markdown_report(
        output_path=markdown_path,
        checkpoint_path=args.checkpoint,
        input_dir=args.input_dir,
        summary=summary,
    )

    print(f"[ok] Processed {len(predictions)} image(s)")
    print(f"[ok] JSON predictions: {predictions_path}")
    print(f"[ok] JSON summary: {summary_path}")
    print(f"[ok] CSV predictions: {csv_path}")
    print(f"[ok] Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
