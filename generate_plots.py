from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate versionable SVG plots for a trained model artifact directory.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Directory that contains history.csv and test_metrics.json.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from manometry_models.plots import generate_plots_from_artifacts

    plot_paths = generate_plots_from_artifacts(args.artifacts_dir)
    print(f"Generated plots for: {args.artifacts_dir}")
    for label, path in plot_paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
