"""Robustness evaluation for the existing baseline and context detector modes."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from pci_poc.benchmark import _contextualize, _draw_sanitized_comparison
from pci_poc.context import ContextConfig
from pci_poc.detect import Detection, detect_candidates
from pci_poc.evaluate import evaluate_boxes, evaluate_redaction
from pci_poc.ocr_utils import TesseractOCR
from pci_poc.redact import redact_image
from pci_poc.synth import GroundTruthRegion, _font


@dataclass(frozen=True)
class RobustnessSample:
    """One generated adversarial synthetic screenshot with its test condition."""

    image_path: Path
    regions: tuple[GroundTruthRegion, ...]
    condition: str
    template_id: str


ROBUSTNESS_TEMPLATES = (
    {"id": "robust_ultrawide", "size": (1600, 720), "bg": "#eef2f6", "header": "#314b62", "label_x": 95, "value_x": 490, "y": 270, "font": 34, "panel": 110},
    {"id": "robust_narrow_sidebar", "size": (1280, 680), "bg": "#f0f4f2", "header": "#27635f", "label_x": 72, "value_x": 390, "y": 250, "font": 25, "panel": 225},
    {"id": "robust_low_density", "size": (1440, 760), "bg": "#f4f0ec", "header": "#684d3f", "label_x": 155, "value_x": 540, "y": 315, "font": 38, "panel": 150},
    {"id": "robust_compact", "size": (1360, 640), "bg": "#edf1f5", "header": "#3c536b", "label_x": 110, "value_x": 455, "y": 230, "font": 23, "panel": 95},
)


ROBUSTNESS_CASES = (
    ("clear_card", "pan", "Card Number", "left", "Synthetic CRM Account details", "spaced"),
    ("missing_label_card", "pan", "", "missing", "Synthetic CRM Account details", "hyphenated"),
    ("far_label_card", "pan", "Card Number", "far", "Synthetic CRM Account details", "plain"),
    ("vertical_label_card", "pan", "Card Number", "vertical", "Synthetic CRM Account details", "spaced"),
    ("cropped_label_card", "pan", "Card Number", "cropped", "Synthetic CRM Account details", "hyphenated"),
    ("ocr_confusion_card", "pan", "Card Nurnber", "left", "Synthetic CRM Account details", "plain"),
    ("conflicting_labels_card", "pan", "Card Number", "conflict", "Synthetic CRM Account details", "spaced"),
    ("misleading_negative_label_card", "pan", "Account Reference", "left", "Synthetic CRM Account details", "plain"),
    ("misleading_ui_context_card", "pan", "Card Number", "left", "Support Ticket CRM", "spaced"),
    ("account_reference_luhn_decoy", "decoy", "Account Reference", "left", "Synthetic CRM Account details", "spaced"),
    ("order_id_luhn_decoy", "decoy", "Order ID", "left", "Synthetic CRM Account details", "hyphenated"),
    ("phone_luhn_decoy", "decoy", "Phone", "left", "Synthetic CRM Account details", "plain"),
    ("misleading_positive_label_decoy", "decoy", "Card Number", "left", "Synthetic CRM Account details", "spaced"),
    ("conflicting_labels_decoy", "decoy", "Card Number", "conflict", "Synthetic CRM Account details", "plain"),
    ("misleading_ui_context_decoy", "decoy", "Account Reference", "left", "Payment Details CRM", "hyphenated"),
    ("invalid_luhn_decoy", "decoy", "Account Reference", "left", "Synthetic CRM Account details", "plain_invalid"),
    ("negative_text_only", "negative", "Phone", "left", "Synthetic CRM Account details", "phone"),
    ("cropped_number_card", "pan", "Card Number", "number_cropped", "Synthetic CRM Account details", "plain"),
)


def _new_valid_number(rng: random.Random) -> str:
    from pci_poc.luhn import append_luhn_check_digit

    return append_luhn_check_digit(rng.choice("456") + "".join(str(rng.randrange(10)) for _ in range(14)))


def _format_number(number: str, style: str) -> str:
    if style == "spaced":
        return " ".join(number[index : index + 4] for index in range(0, len(number), 4))
    if style == "hyphenated":
        return "-".join(number[index : index + 4] for index in range(0, len(number), 4))
    return number


def generate_robustness_dataset(
    output_dir: str | Path,
    *,
    seed: int = 20261011,
    repetitions: int = 2,
) -> list[RobustnessSample]:
    """Generate new-layout, synthetic-only adversarial cases without changing old data."""
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    rng = random.Random(seed)
    root = Path(output_dir)
    image_dir = root / "screenshots"
    image_dir.mkdir(parents=True, exist_ok=True)
    samples: list[RobustnessSample] = []
    manifest: list[dict[str, Any]] = []

    for case_index, (condition, truth_class, label, placement, ui_header, style) in enumerate(ROBUSTNESS_CASES * repetitions):
        template = ROBUSTNESS_TEMPLATES[case_index % len(ROBUSTNESS_TEMPLATES)]
        width, height = template["size"]
        bg = str(template["bg"])
        image = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(image)
        label_font = _font(int(template["font"]))
        value_font = _font(int(template["font"]) + 3)
        header_font = _font(max(25, int(template["font"]) - 2))
        draw.rectangle((0, 0, width, 78), fill=template["header"])
        draw.text((32, 22), ui_header, font=header_font, fill="white")
        panel_left = int(template["panel"])
        draw.rounded_rectangle(
            (panel_left, 112, width - 45, height - 45),
            radius=8,
            fill="white",
            outline="#d4dde4",
            width=2,
        )
        draw.text((panel_left + 30, 140), rng.choice(("Avery Sample", "Riley Synthetic", "Taylor Example")), font=header_font, fill="#1e3548")

        number = _new_valid_number(rng)
        displayed_number = _format_number(number, style)
        if style == "plain_invalid":
            displayed_number = _format_number(number[:-1] + str((int(number[-1]) + 1) % 10), "plain")
        label_x = int(template["label_x"])
        value_x = int(template["value_x"])
        y = int(template["y"])
        if placement == "far":
            label_x = max(55, panel_left + 25)
            value_x = min(width - 360, value_x + 470)
        elif placement in {"conflict", "number_cropped"}:
            value_x = min(width - 380, value_x + 210)

        value_position = (value_x, y)
        draw.text(value_position, displayed_number, font=value_font, fill="#172b3d")
        number_box = draw.textbbox(value_position, displayed_number, font=value_font)
        if placement == "number_cropped":
            crop_at = value_x + max(20, int((number_box[2] - value_x) * 0.62))
            draw.rectangle((crop_at, y - 5, number_box[2] + 8, number_box[3] + 5), fill="white")

        if placement not in {"missing"}:
            if placement == "vertical":
                label_y = y - int(template["font"]) - 32
            else:
                label_y = y
            draw.text((label_x, label_y), label, font=label_font, fill="#536273")
            label_box = draw.textbbox((label_x, label_y), label, font=label_font)
            if placement == "cropped":
                crop_at = label_x + max(18, int((label_box[2] - label_x) * 0.63))
                draw.rectangle((crop_at, label_y - 3, label_box[2] + 4, label_box[3] + 3), fill="white")
            if placement == "conflict":
                secondary = "Account Reference" if truth_class == "pan" else "Order ID"
                secondary_x = label_box[2] + 24
                draw.text((secondary_x, label_y), secondary, font=label_font, fill="#536273")

        regions: list[GroundTruthRegion] = []
        if truth_class in {"pan", "decoy"}:
            regions.append(GroundTruthRegion(truth_class, tuple(number_box)))
        image_name = f"robust_{case_index:04d}.png"
        image_path = image_dir / image_name
        image.save(image_path)
        sample = RobustnessSample(image_path, tuple(regions), condition, str(template["id"]))
        samples.append(sample)
        manifest.append(
            {
                "image": str(image_path.relative_to(root)),
                "synthetic_only": True,
                "dataset_split": "robustness_final",
                "condition": condition,
                "template_id": template["id"],
                "ui_context": ui_header,
                "regions": [
                    {"label": region.label, "bbox": list(region.bbox)}
                    for region in regions
                ],
            }
        )

    (root / "ground_truth.json").write_text(
        json.dumps({"seed": seed, "samples": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    return samples


def _metrics(rows: Sequence[dict[str, Any]], coverages: Sequence[float], mode: str) -> dict[str, Any]:
    tp = sum(row["true_positives"] for row in rows)
    fp = sum(row["false_positives"] for row in rows)
    fn = sum(row["false_negatives"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    latencies = [row["latency_ms"] for row in rows]
    return {
        "mode": mode,
        "evaluated_screenshots": len(rows),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_redactions": sum(row["false_redaction_count"] for row in rows),
        "needs_review_count": sum(row["needs_review_count"] for row in rows),
        "mean_card_text_coverage": statistics.fmean(coverages) if coverages else 0.0,
        "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "ocr_failures": sum(bool(row["ocr_failed"]) for row in rows),
    }


def run_robustness_experiment(
    output_dir: str | Path,
    *,
    seed: int = 20261011,
    repetitions: int = 2,
    config: ContextConfig | None = None,
) -> dict[str, Any]:
    """Compare the frozen baseline and context config on new adversarial templates."""
    root = Path(output_dir)
    samples = generate_robustness_dataset(root / "dataset", seed=seed, repetitions=repetitions)
    manifest = json.loads((root / "dataset" / "ground_truth.json").read_text(encoding="utf-8"))
    sample_metadata = {Path(item["image"]).name: item for item in manifest["samples"]}
    selected_config = config or ContextConfig()
    ocr = TesseractOCR()

    # Warm each unchanged strategy once; exclude this pass from timing.
    with Image.open(samples[0].image_path) as image:
        warmup_image = image.convert("RGB")
    warmup_tokens = ocr.recognize(warmup_image)
    detect_candidates(warmup_tokens)
    _contextualize(warmup_tokens, selected_config)

    mode_rows: dict[str, list[dict[str, Any]]] = {"baseline": [], "context_aware": []}
    mode_coverages: dict[str, list[float]] = {"baseline": [], "context_aware": []}
    visualizations: dict[str, str] = {}
    output_dirs = {mode: root / f"{mode}_redacted" for mode in mode_rows}
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        metadata = sample_metadata[sample.image_path.name]
        cards = [region for region in sample.regions if region.label == "pan"]
        decoys = [region for region in sample.regions if region.label == "decoy"]
        for mode in ("baseline", "context_aware"):
            with Image.open(sample.image_path) as source:
                image = source.convert("RGB")
            started = time.perf_counter()
            ocr_failed = False
            failure_type = ""
            try:
                tokens = ocr.recognize(image)
            except Exception as error:
                tokens = []
                detections: list[Detection] = []
                ocr_failed = True
                failure_type = type(error).__name__
                review_count = 0
            else:
                if mode == "baseline":
                    detections = detect_candidates(tokens)
                    review_count = 0
                else:
                    result = _contextualize(tokens, selected_config)
                    detections = list(result.detections)
                    review_count = len(result.needs_review)

            redacted_path = output_dirs[mode] / sample.image_path.name
            redact_image(sample.image_path, detections, redacted_path)
            latency_ms = (time.perf_counter() - started) * 1000.0
            box_metrics = evaluate_boxes(
                [detection.bbox for detection in detections],
                [region.bbox for region in cards],
            )
            redaction_metrics = evaluate_redaction(
                sample.image_path,
                redacted_path,
                [region.bbox for region in cards],
                [region.bbox for region in decoys],
            )
            mode_coverages[mode].extend(redaction_metrics.card_coverage)
            row = {
                "image": sample.image_path.name,
                "condition": sample.condition,
                "template_id": sample.template_id,
                "truth_class": "pan" if cards else "decoy" if decoys else "negative",
                "true_positives": box_metrics.true_positives,
                "false_positives": box_metrics.false_positives,
                "false_negatives": box_metrics.false_negatives,
                "false_redaction_count": redaction_metrics.false_redaction_count,
                "needs_review_count": review_count,
                "card_text_coverage": statistics.fmean(redaction_metrics.card_coverage) if redaction_metrics.card_coverage else 0.0,
                "ocr_failed": ocr_failed,
                "ocr_failure_type": failure_type,
                "latency_ms": latency_ms,
            }
            mode_rows[mode].append(row)

            for error_type, error_indices in (
                ("false_positive", box_metrics.false_positive_indices),
                ("false_negative", box_metrics.false_negative_indices),
            ):
                key = f"{mode}_{error_type}"
                if error_indices and key not in visualizations:
                    path = root / "comparisons" / f"{mode}_{error_type}_{sample.condition}_{sample.image_path.stem}.png"
                    _draw_sanitized_comparison(
                        sample,
                        [detection.bbox for detection in detections],
                        path,
                        false_positive=error_type == "false_positive",
                    )
                    visualizations[key] = str(path.relative_to(root))

    summaries = {
        mode: _metrics(mode_rows[mode], mode_coverages[mode], mode)
        for mode in mode_rows
    }
    condition_summaries: list[dict[str, Any]] = []
    for condition in dict.fromkeys(sample.condition for sample in samples):
        for mode in mode_rows:
            rows = [row for row in mode_rows[mode] if row["condition"] == condition]
            summary = _metrics(rows, [row["card_text_coverage"] for row in rows if row["truth_class"] == "pan"], mode)
            condition_summaries.append({"condition": condition, **summary})

    report = {
        "scope": "synthetic robustness cases on new UI templates; not real-world performance",
        "synthetic_only": True,
        "ocr_mode": "real_tesseract",
        "seed": seed,
        "repetitions_per_condition": repetitions,
        "iou_threshold": 0.5,
        "templates": sorted({sample.template_id for sample in samples}),
        "templates_disjoint_from_prior_final": True,
        "warmup_images_per_mode": 1,
        "latency_scope": "per-image OCR, detection and redaction; excludes generation and warmup",
        "context_config": {
            "max_label_distance": selected_config.max_label_distance,
            "row_tolerance": selected_config.row_tolerance,
            "ambiguous_action": selected_config.ambiguous_action,
            "unlabeled_action": selected_config.unlabeled_action,
        },
        "baseline": summaries["baseline"],
        "context_aware": summaries["context_aware"],
        "condition_metrics": condition_summaries,
        "visualizations": visualizations,
        "per_image": [
            {
                "image": baseline_row["image"],
                "condition": baseline_row["condition"],
                "template_id": baseline_row["template_id"],
                **{f"baseline_{key}": value for key, value in baseline_row.items() if key not in {"image", "condition", "template_id"}},
                **{f"context_{key}": value for key, value in context_row.items() if key not in {"image", "condition", "template_id"}},
            }
            for baseline_row, context_row in zip(mode_rows["baseline"], mode_rows["context_aware"])
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(root / "metrics.csv", [summaries["baseline"], summaries["context_aware"]])
    _write_csv(root / "condition_metrics.csv", condition_summaries)
    _write_csv(root / "per_image.csv", report["per_image"])
    print("Synthetic robustness experiment; not real-world performance.")
    for summary in summaries.values():
        print(
            f"{summary['mode']}: TP={summary['true_positives']} FP={summary['false_positives']} "
            f"FN={summary['false_negatives']} Precision={summary['precision']:.4f} "
            f"Recall={summary['recall']:.4f} F1={summary['f1']:.4f} "
            f"FalseRedactions={summary['false_redactions']} Review={summary['needs_review_count']} "
            f"Coverage={summary['mean_card_text_coverage']:.4f} "
            f"Median={summary['median_latency_ms']:.2f}ms P95={summary['p95_latency_ms']:.2f}ms"
        )
    print(f"Screenshots={len(samples)} OCR failures={summaries['baseline']['ocr_failures']}/{summaries['context_aware']['ocr_failures']}")
    print(f"Metrics and sanitized visualizations saved under {root}.")
    return report


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("generated") / "robustness_experiment")
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--repetitions", type=int, default=2)
    arguments = parser.parse_args()
    run_robustness_experiment(
        arguments.output_dir,
        seed=arguments.seed,
        repetitions=arguments.repetitions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())