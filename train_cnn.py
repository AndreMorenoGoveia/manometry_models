from __future__ import annotations

import argparse
from pathlib import Path

from manometry_models.model_registry import SUPPORTED_MODEL_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an image classifier for esophageal manometry studies.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory that contains train/val/test.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for checkpoints and metrics. Defaults to artifacts/<run-name>.")
    parser.add_argument("--run-name", type=str, default=None, help="Artifact directory name used when --output-dir is not provided.")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=SUPPORTED_MODEL_NAMES,
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use torchvision pretrained ImageNet weights when available for the selected backbone.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--image-size", type=int, default=None, help="Input image size. Defaults to the selected model recommendation.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.35, help="Dropout rate used in the custom CNN classifier head.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device. 'auto' picks CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable light online augmentation after loading the filtered training split.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class-weighted cross-entropy.",
    )
    parser.add_argument(
        "--include-offline-augmented",
        action="store_true",
        help="Include offline augmented files found in data/train instead of filtering them out.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import torch
    from torch import nn

    from manometry_models.data import compute_class_weights, create_dataloaders
    from manometry_models.model import build_model_config, create_model
    from manometry_models.training import resolve_device, set_seed, train_model

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.image_size is not None and args.image_size < 32:
        raise ValueError("--image-size must be at least 32.")

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model_config = build_model_config(
        args.model,
        pretrained=args.pretrained,
        dropout=args.dropout,
        image_size=args.image_size,
    )
    run_name = args.run_name or (
        f"{model_config.model_name}_pretrained" if model_config.pretrained else model_config.model_name
    )
    output_dir = args.output_dir or (Path("artifacts") / run_name)

    bundle = create_dataloaders(
        data_dir=args.data_dir,
        image_size=model_config.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        include_offline_augmented_train=args.include_offline_augmented,
        normalization_mean=model_config.normalization_mean,
        normalization_std=model_config.normalization_std,
    )
    print("Classes:", ", ".join(bundle.class_names))
    print(f"Model: {model_config.model_name}")
    print(f"Pretrained: {model_config.pretrained}")
    print(f"Image size: {model_config.image_size}")
    print(f"Artifacts dir: {output_dir}")
    print("Training images per class (effective):", bundle.class_counts)
    if any(bundle.excluded_train_class_counts.values()):
        print("Excluded offline augmented training files:", bundle.excluded_train_class_counts)
        print("Training images per class (raw):", bundle.raw_class_counts)

    model = create_model(model_config, num_classes=len(bundle.class_names)).to(device)

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(bundle.class_counts, bundle.class_names).to(device)
        print("Using class weights:", [round(weight.item(), 4) for weight in class_weights])

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    result = train_model(
        model=model,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        test_loader=bundle.test_loader,
        class_names=bundle.class_names,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        output_dir=output_dir,
        image_size=model_config.image_size,
        model_metadata={
            "model_name": model_config.model_name,
            "pretrained": model_config.pretrained,
            "dropout": model_config.dropout,
            "normalization_mean": list(model_config.normalization_mean),
            "normalization_std": list(model_config.normalization_std),
            "aux_logits": model_config.aux_logits,
        },
    )

    print()
    print("Training complete.")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best validation macro F1: {result['best_val_macro_f1']:.4f}")
    print(f"Checkpoint: {result['best_checkpoint_path']}")
    print(f"History CSV: {result['history_path']}")
    print(f"Test metrics JSON: {result['test_metrics_path']}")


if __name__ == "__main__":
    main()
