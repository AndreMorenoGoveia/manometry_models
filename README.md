# Manometry CNN Classifier

This repository contains an image dataset of esophageal manometry studies and a complete PyTorch pipeline to train, validate, test, and use a convolutional neural network (CNN) for multi-class classification.

The implementation added to this repository focuses on a pragmatic baseline:

- a custom CNN built from scratch for six image classes,
- training with the existing `train`, `val`, and `test` folder split,
- default filtering of offline augmented files that were mixed into `data/train`,
- class-weighted loss to mitigate imbalance,
- checkpoint saving, training history export, test metrics export, and SVG plots,
- a standalone prediction script for single-image inference.

## Repository Overview

The dataset is already organized in the standard image-classification layout:

```text
data/
  train/
  val/
  test/
    Bradycardia_type_II/
    DES/
    EGJ/
    IEM/
    Jackhammer/
    normal/
```

All files are `.jpg` images. A quick inspection of the repository shows that many original images are `600x588`, while the training split also includes resized and augmented derivatives. The training script therefore resizes every image to a fixed square resolution before feeding it to the network.

The raw training split contains offline augmentation mixed with the original files, based on filename prefixes such as `rotateImage`, `brightnessE`, `addGaussianNoise`, `addSaltAndPepperNoise`, `resizeImage`, `saturationE`, and `cesun`.

That raw layout is useful as source data, but it is a poor default training input because the augmented derivatives are already materialized as separate files. The training pipeline now filters those files out by default and trains on the original-like images only, while keeping `val` and `test` unchanged.

## Classes and Split Sizes

The CNN is configured for the following six classes:

1. `Bradycardia_type_II`
2. `DES`
3. `EGJ`
4. `IEM`
5. `Jackhammer`
6. `normal`

Raw repository size by split:

| Split | Bradycardia_type_II | DES | EGJ | IEM | Jackhammer | normal | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 864 | 1,640 | 1,152 | 2,328 | 480 | 4,680 | 11,144 |
| Validation | 35 | 67 | 48 | 96 | 20 | 195 | 461 |
| Test | 35 | 67 | 48 | 96 | 20 | 195 | 461 |
| Overall | 934 | 1,774 | 1,248 | 2,520 | 520 | 5,070 | 12,066 |

In the raw training split, each original-like image is accompanied by seven offline augmentation variants. After filtering those variants out, the effective training split used by default is:

| Split | Bradycardia_type_II | DES | EGJ | IEM | Jackhammer | normal | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train used by default | 108 | 205 | 144 | 291 | 60 | 585 | 1,393 |

The dataset remains imbalanced, especially between `normal` and `Jackhammer`. To address this, the training code uses class-weighted cross-entropy by default.

## CNN Architecture

The model is defined in [manometry_models/model.py](/home/andre/repos/manometry_models/manometry_models/model.py). It is a compact custom CNN with batch normalization and dropout:

- input: RGB image resized to `224x224` by default,
- feature extractor: stacked `3x3` convolutions with `ReLU`, `BatchNorm2d`, and `MaxPool2d`,
- channel progression: `3 -> 32 -> 64 -> 128 -> 256`,
- global aggregation: `AdaptiveAvgPool2d((1, 1))`,
- classifier head: `Linear(256 -> 128 -> 6)` with dropout.

This is intentionally a baseline architecture: small enough to train on modest hardware, but expressive enough to learn spatial patterns from manometry images.

## Training Pipeline

The training entry point is [train_cnn.py](/home/andre/repos/manometry_models/train_cnn.py).

Main training choices:

- framework: PyTorch,
- optimizer: `AdamW`,
- scheduler: `ReduceLROnPlateau`,
- loss: `CrossEntropyLoss`,
- raw offline training augmentations: excluded by default,
- class imbalance handling: inverse-frequency class weights,
- default image size: `224`,
- default batch size: `32`,
- default epochs: `20`,
- default seed: `42`.

Validation is run after every epoch. The best checkpoint is selected using validation macro F1, which is more appropriate than raw accuracy for this class distribution.

After training, the pipeline also generates versionable SVG plots for:

- loss,
- accuracy,
- confusion matrix.

## Files Added

- [train_cnn.py](/home/andre/repos/manometry_models/train_cnn.py): trains the CNN and evaluates it on the test split.
- [predict_cnn.py](/home/andre/repos/manometry_models/predict_cnn.py): runs inference on a single image with a saved checkpoint.
- [prepare_dataset.py](/home/andre/repos/manometry_models/prepare_dataset.py): creates a clean dataset copy without offline augmented training files.
- [generate_plots.py](/home/andre/repos/manometry_models/generate_plots.py): regenerates plots from an existing artifact directory.
- [manometry_models/data.py](/home/andre/repos/manometry_models/manometry_models/data.py): dataset loading, transforms, and class-weight utilities.
- [manometry_models/model.py](/home/andre/repos/manometry_models/manometry_models/model.py): CNN architecture.
- [manometry_models/metrics.py](/home/andre/repos/manometry_models/manometry_models/metrics.py): confusion matrix and classification metrics.
- [manometry_models/plots.py](/home/andre/repos/manometry_models/manometry_models/plots.py): SVG plot generation for training curves and confusion matrices.
- [manometry_models/training.py](/home/andre/repos/manometry_models/manometry_models/training.py): training loop, evaluation loop, checkpointing, and history export.
- [requirements.txt](/home/andre/repos/manometry_models/requirements.txt): Python dependencies.

## Installation

Use Python 3.10+.

Create a virtual environment if desired, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Train

Run the baseline configuration:

```bash
python3 train_cnn.py
```

By default, this excludes offline augmented files from `data/train`. To reproduce the raw training split exactly as stored in the repository, add:

```bash
python3 train_cnn.py --include-offline-augmented
```

Example with custom settings:

```bash
python3 train_cnn.py \
  --data-dir data \
  --output-dir artifacts/cnn_run_01 \
  --epochs 30 \
  --batch-size 32 \
  --image-size 224 \
  --learning-rate 1e-3 \
  --num-workers 4 \
  --augment
```

Useful options:

- `--device auto|cpu|cuda|mps`
- `--no-class-weights`
- `--include-offline-augmented`
- `--dropout 0.35`
- `--weight-decay 1e-4`
- `--seed 42`

## Create a Clean Dataset Copy

If you want to physically separate the raw mixed training split from the cleaned training split, use:

```bash
python3 prepare_dataset.py \
  --source-dir data \
  --output-dir data_clean
```

This creates:

- `data_clean/train` with offline augmented training files removed,
- `data_clean/val` unchanged,
- `data_clean/test` unchanged,
- `data_clean/dataset_report.json` with counts and excluded-file totals.

The default mode is `hardlink`, which avoids duplicating image bytes when the filesystem supports it. If you prefer independent copies, use:

```bash
python3 prepare_dataset.py --mode copy
```

Once created, you can train from the cleaned directory:

```bash
python3 train_cnn.py --data-dir data_clean
```

## Training Outputs

After training, the output directory contains:

- `best_model.pt`: checkpoint with model weights, class names, image size, epoch, and validation metrics,
- `history.csv`: epoch-by-epoch train/validation history,
- `test_metrics.json`: final metrics on the held-out test set,
- `training_summary.json`: paths to the main artifacts, plots, and the best validation score,
- `plots/loss.svg`: train and validation loss,
- `plots/accuracy.svg`: train and validation accuracy,
- `plots/confusion_matrix.svg`: confusion matrix for the held-out test set.

`test_metrics.json` includes:

- overall accuracy,
- macro precision,
- macro recall,
- macro F1,
- weighted F1,
- per-class precision/recall/F1/support,
- confusion matrix.

If you need to regenerate plots for a trained model, run:

```bash
python3 generate_plots.py --artifacts-dir artifacts/cnn
```

## How to Run Inference

Use a saved checkpoint and a single image:

```bash
python3 predict_cnn.py \
  --checkpoint artifacts/cnn/best_model.pt \
  --image data/test/EGJ/11.jpg
```

Optional JSON output:

```bash
python3 predict_cnn.py \
  --checkpoint artifacts/cnn/best_model.pt \
  --image data/test/EGJ/11.jpg \
  --json
```

## Implementation Notes

- The code assumes the current folder structure is preserved.
- Images are normalized with mean and standard deviation `(0.5, 0.5, 0.5)`.
- Offline augmented files in `data/train` are filtered out by default based on their filename prefixes.
- Online augmentation is intentionally light and optional.
- The checkpoint stores the class order, so inference remains consistent with training.
- The best model is selected by validation macro F1 to reduce bias toward the majority class.
- Plot outputs are SVG files so they can be reviewed in diffs and committed per model.
- The repository ignores binary checkpoints by default but allows `history.csv`, `test_metrics.json`, `training_summary.json`, and SVG plots inside `artifacts/` to be committed.

## Recommended Next Steps

This baseline is a good starting point, but there are several obvious improvements:

1. compare this custom CNN against transfer learning with `ResNet18`, `EfficientNet`, or `ConvNeXt-Tiny`,
2. add early stopping,
3. add experiment tracking,
4. create plots for loss, accuracy, and confusion matrix,

## Quick Start

```bash
pip install -r requirements.txt
python3 train_cnn.py --epochs 20 --output-dir artifacts/cnn
python3 predict_cnn.py --checkpoint artifacts/cnn/best_model.pt --image data/test/normal/13.jpg
```
