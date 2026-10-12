"""Create the recall chart and copy sanitized synthetic failure comparisons."""

from __future__ import annotations

import json
import math
import shutil
from datetime import date
from html import escape
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_REPORT = ROOT / "generated/context_experiment_final/final_comparison/context_comparison.json"
ROBUSTNESS_REPORT = ROOT / "generated/robustness_experiment/metrics.json"
AUDIT_REPORT = ROOT / "generated/hybrid_policy_audit_review/metrics.json"
DOCS = ROOT / "docs"
IMAGE_DESTINATION = DOCS / "images"


def _read_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _recall(report: dict[str, Any], mode: str, report_path: Path) -> float:
    metrics = report.get(mode)
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing {mode!r} metrics in {report_path}")
    value = metrics.get("recall")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid {mode!r} recall in {report_path}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Recall must be between 0 and 1 in {report_path}")
    return float(value)


def _sample_count(report: dict[str, Any], mode: str, report_path: Path) -> int:
    metrics = report.get(mode)
    count = metrics.get("evaluated_screenshots") if isinstance(metrics, dict) else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"Invalid evaluated screenshot count in {report_path}")
    return count


def _copy_visual(
    report_path: Path,
    report: dict[str, Any],
    visual_key: str,
    destination_name: str,
) -> Path:
    visual_paths = report.get("visualizations")
    relative_path = visual_paths.get(visual_key) if isinstance(visual_paths, dict) else None
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"Missing {visual_key!r} sanitized visual in {report_path}")

    report_root = report_path.parent.resolve()
    source = (report_root / relative_path).resolve()
    try:
        source.relative_to(report_root)
    except ValueError as error:
        raise ValueError(f"Visual path escapes its report directory: {relative_path}") from error
    if not source.is_file():
        raise FileNotFoundError(source)

    IMAGE_DESTINATION.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, IMAGE_DESTINATION / destination_name)
    return source


def _write_field_crop(
    source: Path,
    destination_name: str,
    crop_box: tuple[int, int, int, int],
) -> None:
    with Image.open(source) as image:
        if crop_box[0] < 0 or crop_box[1] < 0 or crop_box[2] > image.width or crop_box[3] > image.height:
            raise ValueError(f"Crop box {crop_box} exceeds image bounds {image.size}: {source}")
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            raise ValueError(f"Invalid crop box {crop_box}: {source}")
        crop = image.crop(crop_box)
        enlarged = crop.resize(
            (crop.width * 2, crop.height * 2),
            Image.Resampling.LANCZOS,
        )
        enlarged.save(IMAGE_DESTINATION / destination_name, format="PNG")


def _metric(metrics: dict[str, Any], name: str) -> str:
    return str(_numeric_metric(metrics, name))


def _numeric_metric(metrics: dict[str, Any], name: str) -> int | float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric metric {name!r} from the saved audit report")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite metric {name!r} in the saved audit report")
    return value


def _write_summary_table(final_split: dict[str, Any]) -> None:
    split_metrics = final_split.get("metrics")
    if not isinstance(split_metrics, dict):
        raise ValueError(f"Missing final-split metrics in {AUDIT_REPORT}")
    modes = ("baseline", "context_only", "hybrid")
    if any(not isinstance(split_metrics.get(mode), dict) for mode in modes):
        raise ValueError(f"Missing baseline, context-only, or hybrid metrics in {AUDIT_REPORT}")

    screenshot_counts = {
        _numeric_metric(split_metrics[mode], "evaluated_screenshots")
        for mode in modes
    }
    if len(screenshot_counts) != 1:
        raise ValueError(f"Modes have inconsistent screenshot counts in {AUDIT_REPORT}")
    card_counts = {
        _numeric_metric(split_metrics[mode], "tp")
        + _numeric_metric(split_metrics[mode], "fn")
        for mode in modes
    }
    if len(card_counts) != 1:
        raise ValueError(f"Modes have inconsistent card counts in {AUDIT_REPORT}")
    sample_description = (
        f"Final split: {next(iter(screenshot_counts))} images, "
        f"{next(iter(card_counts))} cards"
    )

    def values(keys: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            " / ".join(_metric(split_metrics[mode], key) for key in keys)
            for mode in modes
        )

    rows = (
        ("Localization TP / FP / FN", values(("tp", "fp", "fn"))),
        (
            "Complete-number TP / FP / FN",
            values(("complete_number_tp", "complete_number_fp", "complete_number_fn")),
        ),
        (
            "Mean card-text coverage",
            tuple(
                f"{_numeric_metric(split_metrics[mode], 'mean_card_text_coverage') * 100:.1f}%"
                for mode in modes
            ),
        ),
        (
            "False redactions: decoy / ordinary-negative",
            values(("decoy_false_redactions", "negative_text_redactions")),
        ),
        (
            "Needs-review candidates: masked / total",
            tuple(
                " / ".join(
                    (
                        _metric(split_metrics[mode], "needs_review_masked_count"),
                        _metric(split_metrics[mode], "needs_review_count"),
                    )
                )
                for mode in modes
            ),
        ),
    )
    row_markup = "\n".join(
        "      <tr>"
        + "".join(f"<td>{escape(cell)}</td>" for cell in (label, *row_values))
        + "</tr>"
        for label, row_values in rows
    )
    summary_path = DOCS / "summary.html"
    document = summary_path.read_text(encoding="utf-8")
    document = _replace_marked_content(
        document,
        "<!-- TABLE_ROWS_START -->",
        "<!-- TABLE_ROWS_END -->",
        row_markup,
        summary_path,
    )
    document = _replace_marked_content(
        document,
        "<!-- SAMPLE_START -->",
        "<!-- SAMPLE_END -->",
        escape(sample_description),
        summary_path,
    )
    document = _replace_marked_content(
        document,
        "<!-- DATE_START -->",
        "<!-- DATE_END -->",
        date.today().strftime("%d %B %Y"),
        summary_path,
    )
    summary_path.write_text(
        document,
        encoding="utf-8",
    )


def _replace_marked_content(
    document: str,
    start_marker: str,
    end_marker: str,
    replacement: str,
    path: Path,
) -> str:
    if document.count(start_marker) != 1 or document.count(end_marker) != 1:
        raise ValueError(f"Expected one pair of {start_marker} and {end_marker} in {path}")
    start = document.index(start_marker) + len(start_marker)
    end = document.index(end_marker)
    if end < start:
        raise ValueError(f"Markers are out of order in {path}: {start_marker}")
    return document[:start] + "\n" + replacement + "\n      " + document[end:]


def main() -> None:
    context_report = _read_report(CONTEXT_REPORT)
    robustness_report = _read_report(ROBUSTNESS_REPORT)

    groups = (
        ("Clean synthetic set", context_report, CONTEXT_REPORT),
        ("Adversarial synthetic set", robustness_report, ROBUSTNESS_REPORT),
    )
    modes = (("Baseline", "baseline"), ("Context-only", "context_aware"))
    values = [
        [_recall(report, mode, path) for _, mode in modes]
        for _, report, path in groups
    ]
    counts = [
        _sample_count(report, "baseline", path)
        for _, report, path in groups
    ]

    figure, axis = plt.subplots(figsize=(8.4, 4.4))
    positions = list(range(len(groups)))
    bar_width = 0.34
    for mode_index, (label, _) in enumerate(modes):
        offsets = [
            position + (mode_index - 0.5) * bar_width
            for position in positions
        ]
        bars = axis.bar(
            offsets,
            [row[mode_index] for row in values],
            bar_width,
            label=label,
            color=("0.3", "0.7")[mode_index],
            edgecolor="black",
            linewidth=0.7,
        )
        for bar, row in zip(bars, values):
            value = row[mode_index]
            axis.annotate(
                f"{value:.2f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axis.set_ylabel("Recall")
    axis.set_ylim(0, 1.12)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"{name}\n(n={count})" for (name, _, _), count in zip(groups, counts)]
    )
    axis.legend(frameon=False, loc="upper right")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    DOCS.mkdir(parents=True, exist_ok=True)
    figure.savefig(DOCS / "recall_chart.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    audit_report = _read_report(AUDIT_REPORT)
    false_positive_source = _copy_visual(
        ROBUSTNESS_REPORT,
        robustness_report,
        "baseline_false_positive",
        "baseline_false_positive_account_reference_luhn_decoy_robust_0009.png",
    )
    _write_field_crop(
        false_positive_source,
        "baseline_false_positive_field_crop.png",
        (1340, 165, 1930, 465),
    )
    missing_label_source = _copy_visual(
        ROBUSTNESS_REPORT,
        robustness_report,
        "context_aware_false_negative",
        "context_aware_false_negative_missing_label_card_robust_0001.png",
    )
    _write_field_crop(
        missing_label_source,
        "context_aware_false_negative_field_crop.png",
        (1340, 165, 1930, 465),
    )
    final_split = audit_report.get("splits", {}).get("final")
    if not isinstance(final_split, dict):
        raise ValueError(f"Missing final split in {AUDIT_REPORT}")
    _write_summary_table(final_split)
    audit_visuals = final_split.get("visualizations")
    relative_path = (
        audit_visuals.get("baseline_false_negative")
        if isinstance(audit_visuals, dict)
        else None
    )
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"Missing baseline false-negative visual in {AUDIT_REPORT}")
    report_root = AUDIT_REPORT.parent.resolve()
    source = (report_root / relative_path).resolve()
    try:
        source.relative_to(report_root)
    except ValueError as error:
        raise ValueError(f"Visual path escapes its report directory: {relative_path}") from error
    if not source.is_file():
        raise FileNotFoundError(source)
    IMAGE_DESTINATION.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        source,
        IMAGE_DESTINATION / "baseline_false_negative_cropped_card_text_final_0002.png",
    )
    _write_field_crop(
        source,
        "baseline_false_negative_field_crop.png",
        (1720, 240, 2470, 590),
    )


if __name__ == "__main__":
    main()
