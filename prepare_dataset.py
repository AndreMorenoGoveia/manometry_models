from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from manometry_models.dataset_rules import is_offline_augmented_file, is_supported_image_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a clean dataset copy with offline augmented training files removed.",
    )
    parser.add_argument("--source-dir", type=Path, default=Path("data"), help="Directory that contains train/val/test.")
    parser.add_argument("--output-dir", type=Path, default=Path("data_clean"), help="Directory for the cleaned dataset.")
    parser.add_argument(
        "--mode",
        type=str,
        default="hardlink",
        choices=["hardlink", "copy"],
        help="How to materialize files in the cleaned dataset.",
    )
    parser.add_argument(
        "--include-offline-augmented-train",
        action="store_true",
        help="Keep offline augmented files in the cleaned training split.",
    )
    return parser


def materialize_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def ensure_output_dir_is_safe(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Choose a new directory or remove the existing one first."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_split(
    source_split_dir: Path,
    output_split_dir: Path,
    *,
    remove_offline_augmented: bool,
    mode: str,
    is_offline_augmented_file,
) -> dict[str, int]:
    copied_counts: dict[str, int] = {}
    for class_dir in sorted(path for path in source_split_dir.iterdir() if path.is_dir()):
        copied = 0
        for source_file in sorted(path for path in class_dir.iterdir() if path.is_file()):
            if remove_offline_augmented and is_offline_augmented_file(source_file):
                continue
            destination = output_split_dir / class_dir.name / source_file.name
            materialize_file(source_file, destination, mode=mode)
            copied += 1
        copied_counts[class_dir.name] = copied
    return copied_counts


def save_json(data: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def count_images_by_class(split_dir: Path, include_offline_augmented: bool = True) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        counts[class_dir.name] = sum(
            1
            for path in class_dir.rglob("*")
            if path.is_file()
            and is_supported_image_file(path)
            and (include_offline_augmented or not is_offline_augmented_file(path))
        )
    return counts


def main() -> None:
    args = build_parser().parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir
    remove_offline_augmented = not args.include_offline_augmented_train

    for split in ("train", "val", "test"):
        if not (source_dir / split).exists():
            raise FileNotFoundError(f"Missing split directory: {source_dir / split}")

    ensure_output_dir_is_safe(output_dir)

    train_counts = copy_split(
        source_dir / "train",
        output_dir / "train",
        remove_offline_augmented=remove_offline_augmented,
        mode=args.mode,
        is_offline_augmented_file=is_offline_augmented_file,
    )
    val_counts = copy_split(
        source_dir / "val",
        output_dir / "val",
        remove_offline_augmented=False,
        mode=args.mode,
        is_offline_augmented_file=is_offline_augmented_file,
    )
    test_counts = copy_split(
        source_dir / "test",
        output_dir / "test",
        remove_offline_augmented=False,
        mode=args.mode,
        is_offline_augmented_file=is_offline_augmented_file,
    )

    raw_train_counts = count_images_by_class(source_dir / "train", include_offline_augmented=True)
    excluded_train_counts = {
        class_name: raw_train_counts[class_name] - train_counts[class_name]
        for class_name in raw_train_counts
    }
    report = {
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "mode": args.mode,
        "remove_offline_augmented_train": remove_offline_augmented,
        "counts": {
            "raw_train": raw_train_counts,
            "clean_train": train_counts,
            "val": val_counts,
            "test": test_counts,
        },
        "excluded_train_files": excluded_train_counts,
    }
    save_json(report, output_dir / "dataset_report.json")

    print(f"Prepared dataset at: {output_dir}")
    print("Train counts:", train_counts)
    if remove_offline_augmented:
        print("Excluded offline augmented train files:", excluded_train_counts)
    print("Validation counts:", val_counts)
    print("Test counts:", test_counts)
    print(f"Report: {output_dir / 'dataset_report.json'}")


if __name__ == "__main__":
    main()
