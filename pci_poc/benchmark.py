"""Run and report the real OCR detector on held-out synthetic screenshots."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from pci_poc.context import ContextConfig, ContextDetectionResult, detect_with_context
from pci_poc.detect import Detection, detect_candidates
from pci_poc.evaluate import evaluate_boxes, evaluate_redaction
from pci_poc.ocr_utils import TesseractOCR
from pci_poc.redact import redact_image
from pci_poc.synth import (
    GroundTruthRegion,
    SyntheticSample,
    generate_benchmark_dataset,
    generate_context_tuning_datasets,
)


def _draw_sanitized_comparison(
    sample: SyntheticSample,
    predicted_boxes: Sequence[Sequence[int]],
    destination: Path,
    *,
    false_positive: bool,
) -> None:
    """Save a box comparison after obscuring all synthetic number text regions."""
    with Image.open(sample.image_path) as source:
        original = source.convert("RGB")
    left_panel = original.copy()
    right_panel = original.copy()
    left_draw = ImageDraw.Draw(left_panel)
    right_draw = ImageDraw.Draw(right_panel)
    for region in sample.regions:
        left_draw.rectangle(region.bbox, fill="#111111", outline="#48a868", width=3)
        right_draw.rectangle(region.bbox, fill="#111111", outline="#48a868", width=3)
    for box in predicted_boxes:
        right_draw.rectangle(tuple(box), outline="#e07a33", width=4)

    comparison = Image.new("RGB", (original.width * 2, original.height + 46), "white")
    comparison.paste(left_panel, (0, 46))
    comparison.paste(right_panel, (original.width, 46))
    draw = ImageDraw.Draw(comparison)
    draw.text((12, 12), "Ground truth", fill="#173b57")
    draw.text((original.width + 12, 12), "Ground truth + prediction", fill="#173b57")
    draw.text(
        (original.width * 2 - 190, 12),
        "False positive" if false_positive else "False negative",
        fill="#a52d2d",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(destination)


def write_benchmark_reports(
    output_dir: str | Path,
    aggregate: dict[str, Any],
    per_image: Sequence[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    """Write machine-readable JSON, aggregate CSV, and per-image CSV reports."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "metrics.json"
    aggregate_csv_path = root / "metrics.csv"
    per_image_csv_path = root / "per_image.csv"
    report = {
        "benchmark_scope": "synthetic held-out UI templates; not real-world performance",
        "synthetic_only": True,
        "ocr_mode": "real_tesseract",
        "aggregate": aggregate,
        "per_image": list(per_image),
    }
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    with aggregate_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate))
        writer.writeheader()
        writer.writerow(aggregate)

    per_image_fields = list(per_image[0]) if per_image else ["image"]
    with per_image_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_image_fields)
        writer.writeheader()
        writer.writerows(per_image)
    return json_path, aggregate_csv_path, per_image_csv_path


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def run_benchmark(
    output_dir: str | Path,
    *,
    count: int = 48,
    seed: int = 20261009,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run actual local Tesseract OCR, detector and cover-redaction on held-out data."""
    root = Path(output_dir)
    dataset_dir = root / "dataset"
    samples = generate_benchmark_dataset(dataset_dir, count=count, seed=seed)
    annotations = json.loads((dataset_dir / "ground_truth.json").read_text(encoding="utf-8"))
    metadata_by_name = {
        Path(item["image"]).name: item for item in annotations["samples"]
    }
    ocr = TesseractOCR()

    warmup_path = root / ".warmup.png"
    try:
        with Image.open(samples[0].image_path) as image:
            warmup_detections = detect_candidates(ocr.recognize(image))
        redact_image(samples[0].image_path, warmup_detections, warmup_path)
    finally:
        warmup_path.unlink(missing_ok=True)

    redacted_dir = root / "redacted"
    redacted_dir.mkdir(parents=True, exist_ok=True)
    image_rows: list[dict[str, Any]] = []
    all_card_coverage: list[float] = []
    total_tp = total_fp = total_fn = total_false_redactions = total_ocr_failures = 0
    latencies_ms: list[float] = []
    visualization_paths: dict[str, str] = {}

    for sample in samples:
        image_started = time.perf_counter()
        detections: list[Detection] = []
        ocr_failed = False
        with Image.open(sample.image_path) as image:
            original = image.convert("RGB")
        try:
            tokens = ocr.recognize(original)
        except Exception as error:
            ocr_failed = True
            total_ocr_failures += 1
            failure_type = type(error).__name__
        else:
            failure_type = ""
            detections = detect_candidates(tokens)

        redacted_path = redacted_dir / sample.image_path.name
        redact_image(sample.image_path, detections, redacted_path)
        elapsed_ms = (time.perf_counter() - image_started) * 1000.0
        latencies_ms.append(elapsed_ms)

        card_regions = [region for region in sample.regions if region.label == "pan"]
        decoy_regions = [region for region in sample.regions if region.label == "decoy"]
        box_metrics = evaluate_boxes(
            [detection.bbox for detection in detections],
            [region.bbox for region in card_regions],
            iou_threshold=iou_threshold,
        )
        redaction_metrics = evaluate_redaction(
            sample.image_path,
            redacted_path,
            [region.bbox for region in card_regions],
            [region.bbox for region in decoy_regions],
        )
        all_card_coverage.extend(redaction_metrics.card_coverage)
        total_tp += box_metrics.true_positives
        total_fp += box_metrics.false_positives
        total_fn += box_metrics.false_negatives
        total_false_redactions += redaction_metrics.false_redaction_count

        metadata = metadata_by_name[sample.image_path.name]
        image_row: dict[str, Any] = {
            "image": sample.image_path.name,
            "dataset_split": metadata["dataset_split"],
            "template_id": metadata["template_id"],
            "card_count": len(card_regions),
            "decoy_count": len(decoy_regions),
            "prediction_count": len(detections),
            "true_positives": box_metrics.true_positives,
            "false_positives": box_metrics.false_positives,
            "false_negatives": box_metrics.false_negatives,
            "mean_card_coverage": _mean(redaction_metrics.card_coverage),
            "false_redaction_count": redaction_metrics.false_redaction_count,
            "ocr_failed": ocr_failed,
            "ocr_failure_type": failure_type,
            "latency_ms": elapsed_ms,
        }
        image_rows.append(image_row)

        if box_metrics.false_positive_indices and "false_positive" not in visualization_paths:
            index = box_metrics.false_positive_indices[0]
            path = root / "comparisons" / f"false_positive_{sample.image_path.stem}.png"
            _draw_sanitized_comparison(
                sample,
                [detections[index].bbox],
                path,
                false_positive=True,
            )
            visualization_paths["false_positive"] = str(path.relative_to(root))
        if box_metrics.false_negative_indices and "false_negative" not in visualization_paths:
            path = root / "comparisons" / f"false_negative_{sample.image_path.stem}.png"
            _draw_sanitized_comparison(sample, [], path, false_positive=False)
            visualization_paths["false_negative"] = str(path.relative_to(root))

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    aggregate: dict[str, Any] = {
        "seed": seed,
        "iou_threshold": iou_threshold,
        "evaluated_screenshots": len(samples),
        "held_out_templates": sorted({str(row["template_id"]) for row in image_rows}),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_card_text_coverage": _mean(all_card_coverage),
        "false_redaction_count": total_false_redactions,
        "ocr_failures": total_ocr_failures,
        "median_latency_ms": statistics.median(latencies_ms) if latencies_ms else 0.0,
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
        "warmup_images": 1,
        "latency_scope": "per-image OCR, detection, and redaction; excludes dataset generation and warmup",
        "visualizations": visualization_paths,
    }
    version_result = subprocess.run(
        [str(ocr.executable), "--version"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    aggregate["tesseract_version"] = (
        (version_result.stdout or version_result.stderr).splitlines()[0]
        if version_result.returncode == 0
        else "version unavailable"
    )
    aggregate["report_files"] = ["metrics.json", "metrics.csv", "per_image.csv"]
    write_benchmark_reports(root, aggregate, image_rows)
    print("Synthetic held-out benchmark; results are not real-world performance claims.")
    print(
        f"Screenshots={len(samples)} TP={total_tp} FP={total_fp} FN={total_fn} "
        f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} "
        f"CardCoverage={aggregate['mean_card_text_coverage']:.4f} "
        f"FalseRedactions={total_false_redactions} OCRFailures={total_ocr_failures}"
    )
    print(
        f"Latency median={aggregate['median_latency_ms']:.2f}ms "
        f"P95={aggregate['p95_latency_ms']:.2f}ms (one warmup image excluded)"
    )
    print(f"Reports and redacted images saved under {root}.")
    return {"aggregate": aggregate, "per_image": image_rows}


def _contextualize(tokens: Sequence[Any], config: ContextConfig) -> ContextDetectionResult:
    ui_context = " ".join(token.text for token in tokens if any(char.isalpha() for char in token.text))
    return detect_with_context(tokens, config=config, ui_context_text=ui_context)


def _summary_from_rows(
    rows: Sequence[dict[str, Any]],
    coverages: Sequence[float],
    *,
    variant: str,
    templates: Sequence[str],
) -> dict[str, Any]:
    tp = sum(row["true_positives"] for row in rows)
    fp = sum(row["false_positives"] for row in rows)
    fn = sum(row["false_negatives"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "variant": variant,
        "evaluated_screenshots": len(rows),
        "templates": sorted(set(templates)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_card_text_coverage": _mean(coverages),
        "false_redaction_count": sum(row["false_redaction_count"] for row in rows),
        "needs_review_count": sum(row.get("needs_review_count", 0) for row in rows),
        "ocr_failures": sum(bool(row["ocr_failed"]) for row in rows),
        "median_latency_ms": statistics.median(row["latency_ms"] for row in rows) if rows else 0.0,
        "p95_latency_ms": (
            float(np.percentile([row["latency_ms"] for row in rows], 95)) if rows else 0.0
        ),
    }


def run_context_tuning(
    output_dir: str | Path,
    *,
    count_per_split: int = 24,
    seed: int = 20261010,
    config: ContextConfig | None = None,
) -> dict[str, Any]:
    """Select label distance on development data, then validate once on separate templates."""
    root = Path(output_dir)
    development, validation = generate_context_tuning_datasets(
        root / "datasets",
        count_per_split=count_per_split,
        seed=seed,
    )
    base_config = config or ContextConfig()
    candidate_distances = sorted(
        {
            max(120, base_config.max_label_distance - 120),
            base_config.max_label_distance,
            max(base_config.max_label_distance + 160, 520),
        }
    )
    candidates = [replace(base_config, max_label_distance=distance) for distance in candidate_distances]
    ocr = TesseractOCR()
    dev_manifest = json.loads(
        (root / "datasets" / "development" / "ground_truth.json").read_text(encoding="utf-8")
    )
    dev_template_names = sorted({item["template_id"] for item in dev_manifest["samples"]})
    candidate_counts = {
        candidate.max_label_distance: {"tp": 0, "fp": 0, "fn": 0, "review": 0}
        for candidate in candidates
    }
    for sample in development:
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
        try:
            tokens = ocr.recognize(image)
        except Exception:
            for counts in candidate_counts.values():
                counts["fn"] += sum(region.label == "pan" for region in sample.regions)
            continue
        cards = [region for region in sample.regions if region.label == "pan"]
        for candidate in candidates:
            contextual = _contextualize(tokens, candidate)
            boxes = evaluate_boxes(
                [detection.bbox for detection in contextual.detections],
                [region.bbox for region in cards],
            )
            counts = candidate_counts[candidate.max_label_distance]
            counts["tp"] += boxes.true_positives
            counts["fp"] += boxes.false_positives
            counts["fn"] += boxes.false_negatives
            counts["review"] += len(contextual.needs_review)

    development_candidates = []
    for distance, counts in candidate_counts.items():
        precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
        recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
        development_candidates.append(
            {
                "max_label_distance": distance,
                **counts,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }
        )
    validation_manifest = json.loads(
        (root / "datasets" / "validation" / "ground_truth.json").read_text(encoding="utf-8")
    )
    validation_templates = sorted({item["template_id"] for item in validation_manifest["samples"]})
    validation_by_distance = {
        candidate.max_label_distance: {"tp": 0, "fp": 0, "fn": 0, "review": 0}
        for candidate in candidates
    }
    validation_ocr_failures = 0
    for sample in validation:
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
        cards = [region for region in sample.regions if region.label == "pan"]
        try:
            tokens = ocr.recognize(image)
        except Exception:
            validation_ocr_failures += 1
            tokens = []
        for candidate in candidates:
            contextual = _contextualize(tokens, candidate)
            boxes = evaluate_boxes(
                [detection.bbox for detection in contextual.detections],
                [region.bbox for region in cards],
            )
            counts = validation_by_distance[candidate.max_label_distance]
            counts["tp"] += boxes.true_positives
            counts["fp"] += boxes.false_positives
            counts["fn"] += boxes.false_negatives
            counts["review"] += len(contextual.needs_review)

    def summarized_counts(counts: dict[str, int], samples_count: int, templates: list[str]) -> dict[str, Any]:
        precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
        recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
        return {
            "templates": templates,
            "evaluated_screenshots": samples_count,
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }

    validation_candidates = [
        {
            "max_label_distance": distance,
            **summarized_counts(counts, len(validation), validation_templates),
            "ocr_failures": validation_ocr_failures,
        }
        for distance, counts in validation_by_distance.items()
    ]
    dev_by_distance = {item["max_label_distance"]: item for item in development_candidates}
    val_by_distance = {item["max_label_distance"]: item for item in validation_candidates}
    selected_distance = max(
        candidate_distances,
        key=lambda distance: (
            (dev_by_distance[distance]["tp"] + val_by_distance[distance]["tp"])
            / max(1, dev_by_distance[distance]["tp"] + val_by_distance[distance]["tp"] + dev_by_distance[distance]["fn"] + val_by_distance[distance]["fn"]),
            (dev_by_distance[distance]["tp"] + val_by_distance[distance]["tp"])
            / max(1, dev_by_distance[distance]["tp"] + val_by_distance[distance]["tp"] + dev_by_distance[distance]["fp"] + val_by_distance[distance]["fp"]),
            -dev_by_distance[distance]["review"] - val_by_distance[distance]["review"],
            -distance,
        ),
    )
    selected_config = replace(base_config, max_label_distance=selected_distance)
    validation_summary = {
        **val_by_distance[selected_distance],
        "selected_max_label_distance": selected_distance,
        "ocr_failures": validation_ocr_failures,
    }

    def config_to_json(selected: ContextConfig) -> dict[str, Any]:
        return {
            "max_label_distance": selected.max_label_distance,
            "row_tolerance": selected.row_tolerance,
            "positive_labels": list(selected.positive_labels),
            "negative_labels": list(selected.negative_labels),
            "positive_ui_context": list(selected.positive_ui_context),
            "negative_ui_context": list(selected.negative_ui_context),
            "ambiguous_action": selected.ambiguous_action,
            "unlabeled_action": selected.unlabeled_action,
        }

    tuning_report = {
        "scope": "synthetic development/validation templates; not final test or real-world results",
        "synthetic_only": True,
        "selected_config": config_to_json(selected_config),
        "seed": seed,
        "development_templates": dev_template_names,
        "development_candidates": development_candidates,
        "validation_candidates": validation_candidates,
        "validation": validation_summary,
        "selection_rule": "maximize combined development+validation recall, then precision; ties prefer fewer reviews and shorter distance",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "tuning_metrics.json").write_text(
        json.dumps(tuning_report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (root / "tuning_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(development_candidates[0]) if development_candidates else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(development_candidates)
    (root / "validation_metrics.json").write_text(
        json.dumps(validation_summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return tuning_report


def _load_frozen_samples(dataset_dir: Path) -> tuple[list[SyntheticSample], dict[str, Any]]:
    manifest = json.loads((dataset_dir / "ground_truth.json").read_text(encoding="utf-8"))
    samples: list[SyntheticSample] = []
    for entry in manifest["samples"]:
        image_path = dataset_dir / entry["image"]
        if not image_path.is_file():
            raise FileNotFoundError(f"frozen benchmark image is missing: {image_path}")
        regions = tuple(
            GroundTruthRegion(str(region["label"]), tuple(region["bbox"]))
            for region in entry["regions"]
        )
        samples.append(SyntheticSample(image_path, regions))
    return samples, manifest


def run_context_comparison(
    output_dir: str | Path,
    *,
    frozen_dataset_dir: str | Path = Path("generated") / "benchmark" / "dataset",
    tuning_report_path: str | Path = Path("generated") / "context_tuning" / "tuning_metrics.json",
    iou_threshold: float = 0.5,
    config: ContextConfig | None = None,
) -> dict[str, Any]:
    """Compare unchanged baseline/context modes on an existing frozen test set."""
    dataset_dir = Path(frozen_dataset_dir)
    samples, manifest = _load_frozen_samples(dataset_dir)
    final_templates = {item["template_id"] for item in manifest["samples"]}
    if any(item.get("dataset_split") != "held_out" for item in manifest["samples"]):
        raise ValueError("final comparison requires only held_out samples")

    tuning_path = Path(tuning_report_path)
    if tuning_path.is_file():
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        tuning_templates: set[str] = set()
        tuning_templates.update(tuning["development_templates"])
        tuning_templates.update(tuning["validation"]["templates"])
        if final_templates & tuning_templates:
            raise ValueError("tuning templates overlap frozen final templates")

    root = Path(output_dir)
    baseline_dir = root / "baseline_redacted"
    context_dir = root / "context_redacted"
    comparison_dir = root / "comparisons"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    selected_config = config or ContextConfig()
    ocr = TesseractOCR()

    # Warm each strategy once outside the measured set.
    first = samples[0]
    with Image.open(first.image_path) as source:
        warmup_image = source.convert("RGB")
    warmup_tokens = ocr.recognize(warmup_image)
    detect_candidates(warmup_tokens)
    _contextualize(warmup_tokens, selected_config)

    baseline_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    baseline_coverages: list[float] = []
    context_coverages: list[float] = []
    visualizations: dict[str, str] = {}

    for sample, metadata in zip(samples, manifest["samples"]):
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
        cards = [region for region in sample.regions if region.label == "pan"]
        decoys = [region for region in sample.regions if region.label == "decoy"]

        baseline_started = time.perf_counter()
        baseline_tokens = ocr.recognize(image)
        baseline_detections = detect_candidates(baseline_tokens)
        baseline_output = baseline_dir / sample.image_path.name
        redact_image(sample.image_path, baseline_detections, baseline_output)
        baseline_elapsed = (time.perf_counter() - baseline_started) * 1000.0
        baseline_boxes = evaluate_boxes(
            [detection.bbox for detection in baseline_detections],
            [region.bbox for region in cards],
            iou_threshold=iou_threshold,
        )
        baseline_redaction = evaluate_redaction(
            sample.image_path,
            baseline_output,
            [region.bbox for region in cards],
            [region.bbox for region in decoys],
        )
        baseline_coverages.extend(baseline_redaction.card_coverage)
        baseline_rows.append(
            {
                "image": sample.image_path.name,
                "template_id": metadata["template_id"],
                "true_positives": baseline_boxes.true_positives,
                "false_positives": baseline_boxes.false_positives,
                "false_negatives": baseline_boxes.false_negatives,
                "false_redaction_count": baseline_redaction.false_redaction_count,
                "needs_review_count": 0,
                "ocr_failed": False,
                "latency_ms": baseline_elapsed,
            }
        )

        context_started = time.perf_counter()
        context_tokens = ocr.recognize(image)
        context_result = _contextualize(context_tokens, selected_config)
        context_detections = list(context_result.detections)
        context_output = context_dir / sample.image_path.name
        redact_image(sample.image_path, context_detections, context_output)
        context_elapsed = (time.perf_counter() - context_started) * 1000.0
        context_boxes = evaluate_boxes(
            [detection.bbox for detection in context_detections],
            [region.bbox for region in cards],
            iou_threshold=iou_threshold,
        )
        context_redaction = evaluate_redaction(
            sample.image_path,
            context_output,
            [region.bbox for region in cards],
            [region.bbox for region in decoys],
        )
        context_coverages.extend(context_redaction.card_coverage)
        context_rows.append(
            {
                "image": sample.image_path.name,
                "template_id": metadata["template_id"],
                "true_positives": context_boxes.true_positives,
                "false_positives": context_boxes.false_positives,
                "false_negatives": context_boxes.false_negatives,
                "false_redaction_count": context_redaction.false_redaction_count,
                "needs_review_count": len(context_result.needs_review),
                "ocr_failed": False,
                "latency_ms": context_elapsed,
            }
        )

        for rows, detections, title in (
            (baseline_rows, baseline_detections, "baseline"),
            (context_rows, context_detections, "context"),
        ):
            row = rows[-1]
            if row["false_positives"] and f"{title}_false_positive" not in visualizations:
                path = comparison_dir / f"{title}_false_positive_{sample.image_path.stem}.png"
                _draw_sanitized_comparison(
                    sample,
                    [detection.bbox for detection in detections],
                    path,
                    false_positive=True,
                )
                visualizations[f"{title}_false_positive"] = str(path.relative_to(root))
            if row["false_negatives"] and f"{title}_false_negative" not in visualizations:
                path = comparison_dir / f"{title}_false_negative_{sample.image_path.stem}.png"
                _draw_sanitized_comparison(
                    sample,
                    [detection.bbox for detection in detections],
                    path,
                    false_positive=False,
                )
                visualizations[f"{title}_false_negative"] = str(path.relative_to(root))

    baseline_summary = _summary_from_rows(
        baseline_rows,
        baseline_coverages,
        variant="baseline_luhn_only",
        templates=list(final_templates),
    )
    context_summary = _summary_from_rows(
        context_rows,
        context_coverages,
        variant="context_aware",
        templates=list(final_templates),
    )
    comparison_rows = [
        {
            "image": baseline_row["image"],
            "template_id": baseline_row["template_id"],
            **{f"baseline_{key}": value for key, value in baseline_row.items() if key not in {"image", "template_id"}},
            **{f"context_{key}": value for key, value in context_row.items() if key not in {"image", "template_id"}},
        }
        for baseline_row, context_row in zip(baseline_rows, context_rows)
    ]
    report = {
        "scope": "synthetic frozen held-out templates; not real-world performance",
        "synthetic_only": True,
        "ocr_mode": "real_tesseract",
        "iou_threshold": iou_threshold,
        "final_templates": sorted(final_templates),
        "tuning_templates_disjoint": True if tuning_path.is_file() else None,
        "warmup_images_per_mode": 1,
        "latency_scope": "per-image OCR, detection, and redaction; excludes warmup and tuning data generation",
        "visualizations": visualizations,
        "baseline": baseline_summary,
        "context_aware": context_summary,
        "per_image": comparison_rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "context_comparison.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    aggregate_csv = root / "context_comparison.csv"
    with aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", *baseline_summary.keys() - {"variant"}])
        writer.writeheader()
        writer.writerows((baseline_summary, context_summary))
    per_image_csv = root / "context_per_image.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]) if comparison_rows else ["image"])
        writer.writeheader()
        writer.writerows(comparison_rows)

    print("Synthetic frozen-test comparison; not a real-world performance claim.")
    for summary in (baseline_summary, context_summary):
        print(
            f"{summary['variant']}: TP={summary['true_positives']} FP={summary['false_positives']} "
            f"FN={summary['false_negatives']} Precision={summary['precision']:.4f} "
            f"Recall={summary['recall']:.4f} F1={summary['f1']:.4f} "
            f"FalseRedactions={summary['false_redaction_count']} "
            f"Coverage={summary['mean_card_text_coverage']:.4f} "
            f"Median={summary['median_latency_ms']:.2f}ms P95={summary['p95_latency_ms']:.2f}ms"
        )
    print(f"Comparison reports saved under {root}.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("generated") / "benchmark")
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--context-experiment",
        action="store_true",
        help="evaluate context rules on separate dev/validation sets, then compare on the frozen final set",
    )
    parser.add_argument(
        "--frozen-dataset-dir",
        type=Path,
        default=Path("generated") / "benchmark" / "dataset",
    )
    arguments = parser.parse_args()
    if arguments.context_experiment:
        tuning_output = arguments.output_dir / "tuning"
        comparison_output = arguments.output_dir / "final_comparison"
        run_context_tuning(
            tuning_output,
            count_per_split=24,
            seed=arguments.seed + 1,
        )
        tuning_report = json.loads((tuning_output / "tuning_metrics.json").read_text(encoding="utf-8"))
        tuned_config_data = tuning_report["selected_config"]
        tuned_config_data["positive_labels"] = tuple(tuned_config_data["positive_labels"])
        tuned_config_data["negative_labels"] = tuple(tuned_config_data["negative_labels"])
        tuned_config_data["positive_ui_context"] = tuple(tuned_config_data["positive_ui_context"])
        tuned_config_data["negative_ui_context"] = tuple(tuned_config_data["negative_ui_context"])
        tuned_config = ContextConfig(**tuned_config_data)
        run_context_comparison(
            comparison_output,
            frozen_dataset_dir=arguments.frozen_dataset_dir,
            tuning_report_path=tuning_output / "tuning_metrics.json",
            iou_threshold=arguments.iou_threshold,
            config=tuned_config,
        )
        return 0
    run_benchmark(
        arguments.output_dir,
        count=arguments.count,
        seed=arguments.seed,
        iou_threshold=arguments.iou_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())