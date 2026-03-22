from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a CNN to classify esophageal manometry images.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory that contains train/val/test.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cnn"), help="Directory for checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--image-size", type=int, default=224, help="Square image size used by the CNN.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.35, help="Dropout rate used in the classifier head.")
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
        help="Enable light online augmentation on top of the already augmented training split.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class-weighted cross-entropy.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import torch
    from torch import nn

    from manometry_models.data import compute_class_weights, create_dataloaders
    from manometry_models.model import ManometryCNN
    from manometry_models.training import resolve_device, set_seed, train_model

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32.")

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    bundle = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
    )
    print("Classes:", ", ".join(bundle.class_names))
    print("Training images per class:", bundle.class_counts)

    model = ManometryCNN(num_classes=len(bundle.class_names), dropout=args.dropout).to(device)

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
        output_dir=args.output_dir,
        image_size=args.image_size,
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

