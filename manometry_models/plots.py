from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path


SVG_BACKGROUND = "#ffffff"
SVG_TEXT = "#0f172a"
SVG_MUTED_TEXT = "#475569"
SVG_GRID = "#dbe4f0"
SVG_BORDER = "#94a3b8"
SVG_SERIES = {
    "train_loss": "#2563eb",
    "val_loss": "#dc2626",
    "train_accuracy": "#16a34a",
    "val_accuracy": "#9333ea",
    "val_macro_f1": "#ea580c",
}


def load_history_rows(history_path: str | Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with Path(history_path).open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "epoch": float(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_accuracy": float(row["train_accuracy"]),
                    "val_loss": float(row["val_loss"]),
                    "val_accuracy": float(row["val_accuracy"]),
                    "val_macro_f1": float(row["val_macro_f1"]),
                    "learning_rate": float(row["learning_rate"]),
                }
            )
    return rows


def load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: dict[str, object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _format_label(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _nice_range(min_value: float, max_value: float, clamp_zero: bool = False, clamp_one: bool = False) -> tuple[float, float]:
    if clamp_zero:
        min_value = min(min_value, 0.0)
    if clamp_one:
        max_value = max(max_value, 1.0)
    if min_value == max_value:
        pad = 1.0 if max_value == 0 else abs(max_value) * 0.1
        min_value -= pad
        max_value += pad
    else:
        pad = (max_value - min_value) * 0.08
        min_value -= pad
        max_value += pad
    if clamp_zero:
        min_value = max(0.0, min_value)
    if clamp_one:
        max_value = min(1.0, max_value)
        min_value = max(0.0, min_value)
    return min_value, max_value


def _generate_ticks(min_value: float, max_value: float, tick_count: int = 5) -> list[float]:
    if tick_count < 2:
        return [min_value, max_value]
    step = (max_value - min_value) / (tick_count - 1)
    return [min_value + index * step for index in range(tick_count)]


def _interpolate_color(start_hex: str, end_hex: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    start = tuple(int(start_hex[index : index + 2], 16) for index in (1, 3, 5))
    end = tuple(int(end_hex[index : index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(round(start[channel] + (end[channel] - start[channel]) * t) for channel in range(3))
    return f"#{mixed[0]:02x}{mixed[1]:02x}{mixed[2]:02x}"


def _write_svg(svg: str, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")


def render_line_chart_svg(
    rows: list[dict[str, float]],
    *,
    title: str,
    subtitle: str,
    series: list[tuple[str, str]],
    output_path: str | Path,
    y_label: str,
    clamp_zero: bool = False,
    clamp_one: bool = False,
    best_epoch: int | None = None,
) -> None:
    if not rows:
        raise ValueError("Cannot render line chart without history rows.")

    width = 980
    height = 560
    margin_left = 88
    margin_right = 36
    margin_top = 84
    margin_bottom = 96
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    epochs = [row["epoch"] for row in rows]
    all_values = [row[key] for row in rows for key, _ in series]
    y_min, y_max = _nice_range(min(all_values), max(all_values), clamp_zero=clamp_zero, clamp_one=clamp_one)

    def x_position(epoch: float) -> float:
        if len(epochs) == 1:
            return margin_left + plot_width / 2
        return margin_left + ((epoch - epochs[0]) / (epochs[-1] - epochs[0])) * plot_width

    def y_position(value: float) -> float:
        if y_max == y_min:
            return margin_top + plot_height / 2
        return margin_top + (1 - (value - y_min) / (y_max - y_min)) * plot_height

    y_ticks = _generate_ticks(y_min, y_max)
    x_ticks = epochs if len(epochs) <= 20 else [epochs[0], epochs[len(epochs) // 2], epochs[-1]]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{SVG_BACKGROUND}" />',
        f'<text x="{margin_left}" y="38" font-family="Arial, sans-serif" font-size="26" font-weight="700" fill="{SVG_TEXT}">{escape(title)}</text>',
        f'<text x="{margin_left}" y="62" font-family="Arial, sans-serif" font-size="14" fill="{SVG_MUTED_TEXT}">{escape(subtitle)}</text>',
    ]

    for tick in y_ticks:
        y = y_position(tick)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="{SVG_GRID}" stroke-width="1" />')
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="{SVG_MUTED_TEXT}">{escape(_format_label(tick))}</text>'
        )

    for tick in x_ticks:
        x = x_position(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="{SVG_GRID}" stroke-width="1" />')
        tick_label = str(int(tick)) if float(tick).is_integer() else _format_label(tick)
        parts.append(
            f'<text x="{x:.2f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="{SVG_MUTED_TEXT}">{escape(tick_label)}</text>'
        )

    parts.append(
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="{SVG_BORDER}" stroke-width="1" />'
    )

    if best_epoch is not None and epochs[0] <= best_epoch <= epochs[-1]:
        x = x_position(float(best_epoch))
        parts.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="#f59e0b" stroke-width="2" stroke-dasharray="6 6" />'
        )
        parts.append(
            f'<text x="{x + 8:.2f}" y="{margin_top + 18}" font-family="Arial, sans-serif" font-size="12" fill="#92400e">best epoch {best_epoch}</text>'
        )

    legend_x = width - margin_right - 210
    legend_y = margin_top - 16
    for index, (key, label) in enumerate(series):
        color = SVG_SERIES[key]
        y = legend_y + index * 22
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="{color}" stroke-width="3" />')
        parts.append(
            f'<text x="{legend_x + 36}" y="{y + 4}" font-family="Arial, sans-serif" font-size="13" fill="{SVG_TEXT}">{escape(label)}</text>'
        )

    for key, _ in series:
        coordinates = " ".join(f"{x_position(row['epoch']):.2f},{y_position(row[key]):.2f}" for row in rows)
        parts.append(
            f'<polyline fill="none" stroke="{SVG_SERIES[key]}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{coordinates}" />'
        )
        for row in rows:
            x = x_position(row["epoch"])
            y = y_position(row[key])
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{SVG_SERIES[key]}" />')

    parts.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="{SVG_MUTED_TEXT}">Epoch</text>'
    )
    parts.append(
        f'<text x="24" y="{margin_top + plot_height / 2:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="{SVG_MUTED_TEXT}" transform="rotate(-90 24 {margin_top + plot_height / 2:.2f})">{escape(y_label)}</text>'
    )
    parts.append("</svg>")

    _write_svg("\n".join(parts), output_path)


def render_confusion_matrix_svg(
    test_metrics: dict[str, object],
    *,
    title: str,
    output_path: str | Path,
) -> None:
    matrix = test_metrics.get("confusion_matrix")
    per_class = test_metrics.get("per_class")
    if not isinstance(matrix, list) or not isinstance(per_class, list):
        raise ValueError("Test metrics do not contain the expected confusion matrix payload.")

    class_names = [str(item["class_name"]) for item in per_class]
    display_names = [name.replace("_", " ") for name in class_names]
    size = len(matrix)
    cell_size = 86
    margin_left = 210
    margin_top = 140
    margin_right = 48
    margin_bottom = 180
    width = margin_left + size * cell_size + margin_right
    height = margin_top + size * cell_size + margin_bottom

    max_value = max(max(int(value) for value in row) for row in matrix) if matrix else 1
    if max_value <= 0:
        max_value = 1

    accuracy = float(test_metrics.get("accuracy", 0.0))
    macro_f1 = float(test_metrics.get("macro_f1", 0.0))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{SVG_BACKGROUND}" />',
        f'<text x="{margin_left}" y="40" font-family="Arial, sans-serif" font-size="28" font-weight="700" fill="{SVG_TEXT}">{escape(title)}</text>',
        f'<text x="{margin_left}" y="66" font-family="Arial, sans-serif" font-size="14" fill="{SVG_MUTED_TEXT}">Rows are true labels, columns are predicted labels. Accuracy={accuracy:.3f}, macro F1={macro_f1:.3f}</text>',
    ]

    for row_index, row in enumerate(matrix):
        row_total = sum(int(value) for value in row) or 1
        for column_index, raw_value in enumerate(row):
            value = int(raw_value)
            ratio = value / max_value
            fill = _interpolate_color("#eff6ff", "#1d4ed8", ratio)
            x = margin_left + column_index * cell_size
            y = margin_top + row_index * cell_size
            text_fill = "#ffffff" if ratio >= 0.55 else SVG_TEXT
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{fill}" stroke="{SVG_BACKGROUND}" stroke-width="2" />')
            parts.append(
                f'<text x="{x + cell_size / 2}" y="{y + cell_size / 2 - 6}" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="{text_fill}">{value}</text>'
            )
            parts.append(
                f'<text x="{x + cell_size / 2}" y="{y + cell_size / 2 + 16}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="{text_fill}">{(value / row_total) * 100:.1f}%</text>'
            )

    for index, label in enumerate(display_names):
        x = margin_left + index * cell_size + cell_size / 2
        y = margin_top + size * cell_size + 20
        parts.append(
            f'<text x="{x}" y="{y}" text-anchor="start" font-family="Arial, sans-serif" font-size="13" fill="{SVG_TEXT}" transform="rotate(40 {x} {y})">{escape(label)}</text>'
        )

    for index, label in enumerate(display_names):
        y = margin_top + index * cell_size + cell_size / 2 + 4
        parts.append(
            f'<text x="{margin_left - 14}" y="{y}" text-anchor="end" font-family="Arial, sans-serif" font-size="13" fill="{SVG_TEXT}">{escape(label)}</text>'
        )

    parts.append(
        f'<text x="{margin_left + (size * cell_size) / 2}" y="{height - 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="{SVG_MUTED_TEXT}">Predicted label</text>'
    )
    parts.append(
        f'<text x="40" y="{margin_top + (size * cell_size) / 2}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="{SVG_MUTED_TEXT}" transform="rotate(-90 40 {margin_top + (size * cell_size) / 2})">True label</text>'
    )
    parts.append("</svg>")

    _write_svg("\n".join(parts), output_path)


def generate_plots(
    *,
    history_rows: list[dict[str, float]],
    test_metrics: dict[str, object],
    output_dir: str | Path,
    model_name: str,
    best_epoch: int | None = None,
) -> dict[str, str]:
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    loss_plot_path = plots_dir / "loss.svg"
    accuracy_plot_path = plots_dir / "accuracy.svg"
    confusion_matrix_plot_path = plots_dir / "confusion_matrix.svg"

    render_line_chart_svg(
        history_rows,
        title=f"{model_name} loss",
        subtitle="Training and validation loss by epoch",
        series=[("train_loss", "train loss"), ("val_loss", "validation loss")],
        output_path=loss_plot_path,
        y_label="Loss",
        clamp_zero=True,
        best_epoch=best_epoch,
    )
    render_line_chart_svg(
        history_rows,
        title=f"{model_name} accuracy",
        subtitle="Training and validation accuracy by epoch",
        series=[("train_accuracy", "train accuracy"), ("val_accuracy", "validation accuracy")],
        output_path=accuracy_plot_path,
        y_label="Accuracy",
        clamp_zero=True,
        clamp_one=True,
        best_epoch=best_epoch,
    )
    render_confusion_matrix_svg(
        test_metrics,
        title=f"{model_name} confusion matrix",
        output_path=confusion_matrix_plot_path,
    )

    return {
        "plots_dir": str(plots_dir.resolve()),
        "loss_plot_path": str(loss_plot_path.resolve()),
        "accuracy_plot_path": str(accuracy_plot_path.resolve()),
        "confusion_matrix_plot_path": str(confusion_matrix_plot_path.resolve()),
    }


def generate_plots_from_artifacts(artifacts_dir: str | Path) -> dict[str, str]:
    artifacts_path = Path(artifacts_dir)
    history_rows = load_history_rows(artifacts_path / "history.csv")
    test_metrics = load_json(artifacts_path / "test_metrics.json")

    summary_path = artifacts_path / "training_summary.json"
    summary_data = load_json(summary_path) if summary_path.exists() else {}
    best_epoch = int(summary_data["best_epoch"]) if "best_epoch" in summary_data else None

    plot_paths = generate_plots(
        history_rows=history_rows,
        test_metrics=test_metrics,
        output_dir=artifacts_path,
        model_name=artifacts_path.name,
        best_epoch=best_epoch,
    )

    summary_data.update(plot_paths)
    save_json(summary_data, summary_path)
    return plot_paths

