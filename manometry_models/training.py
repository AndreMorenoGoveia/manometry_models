from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from manometry_models.metrics import compute_classification_metrics, save_json, update_confusion_matrix
from manometry_models.plots import generate_plots


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if normalized == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            raise RuntimeError("MPS was requested but is not available.")
    return torch.device(normalized)


def unpack_model_outputs(outputs: object) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(outputs, torch.Tensor):
        return outputs, None
    if hasattr(outputs, "logits"):
        return outputs.logits, getattr(outputs, "aux_logits", None)
    if isinstance(outputs, (tuple, list)):
        logits = outputs[0]
        aux_logits = outputs[1] if len(outputs) > 1 else None
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Expected the main model output to be a tensor.")
        if aux_logits is not None and not isinstance(aux_logits, torch.Tensor):
            aux_logits = None
        return logits, aux_logits
    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_index: int,
    num_epochs: int,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch_index}/{num_epochs} [train]",
        leave=False,
    )
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        logits, aux_logits = unpack_model_outputs(outputs)
        loss = criterion(logits, labels)
        if aux_logits is not None:
            loss = loss + 0.4 * criterion(aux_logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{running_loss / max(total_samples, 1):.4f}",
            acc=f"{correct_predictions / max(total_samples, 1):.4f}",
        )

    return {
        "loss": running_loss / max(total_samples, 1),
        "accuracy": correct_predictions / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    split_name: str,
) -> dict[str, object]:
    model.eval()
    running_loss = 0.0
    total_samples = 0
    confusion_matrix = torch.zeros((len(class_names), len(class_names)), dtype=torch.int64, device="cpu")

    progress_bar = tqdm(dataloader, desc=f"{split_name} evaluation", leave=False)
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        logits, _ = unpack_model_outputs(outputs)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        update_confusion_matrix(confusion_matrix, labels.cpu(), predictions.cpu())

    metrics = compute_classification_metrics(confusion_matrix, class_names)
    metrics["loss"] = running_loss / max(total_samples, 1)
    return metrics


def save_history_csv(history_rows: list[dict[str, float]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "learning_rate",
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)


def save_checkpoint(
    model: nn.Module,
    output_path: str | Path,
    class_names: list[str],
    image_size: int,
    epoch: int,
    metrics: dict[str, object],
    model_metadata: dict[str, object],
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": image_size,
            "epoch": epoch,
            "metrics": metrics,
            **model_metadata,
        },
        output,
    )


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epochs: int,
    output_dir: str | Path,
    image_size: int,
    model_metadata: dict[str, object],
) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history_rows: list[dict[str, float]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    best_checkpoint_path = output_path / "best_model.pt"

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_index=epoch,
            num_epochs=epochs,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
            split_name="Validation",
        )

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(float(train_metrics["loss"]), 6),
                "train_accuracy": round(float(train_metrics["accuracy"]), 6),
                "val_loss": round(float(val_metrics["loss"]), 6),
                "val_accuracy": round(float(val_metrics["accuracy"]), 6),
                "val_macro_f1": round(float(val_metrics["macro_f1"]), 6),
                "learning_rate": round(float(current_lr), 8),
            }
        )

        if float(val_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            save_checkpoint(
                model=model,
                output_path=best_checkpoint_path,
                class_names=class_names,
                image_size=image_size,
                epoch=epoch,
                metrics=val_metrics,
                model_metadata=model_metadata,
            )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

    save_history_csv(history_rows, output_path / "history.csv")

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        split_name="Test",
    )
    save_json(test_metrics, output_path / "test_metrics.json")
    plot_paths = generate_plots(
        history_rows=history_rows,
        test_metrics=test_metrics,
        output_dir=output_path,
        model_name=output_path.name,
        best_epoch=best_epoch,
    )
    save_json(
        {
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_macro_f1,
            **model_metadata,
            "history_path": str((output_path / "history.csv").resolve()),
            "best_checkpoint_path": str(best_checkpoint_path.resolve()),
            "test_metrics_path": str((output_path / "test_metrics.json").resolve()),
            **plot_paths,
        },
        output_path / "training_summary.json",
    )

    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "history_path": output_path / "history.csv",
        "best_checkpoint_path": best_checkpoint_path,
        "test_metrics_path": output_path / "test_metrics.json",
        "test_metrics": test_metrics,
        **model_metadata,
        **plot_paths,
    }
