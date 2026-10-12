"""Fresh synthetic comparison of baseline, context-only and hybrid protection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pci_poc.benchmark import _contextualize, _draw_sanitized_comparison
from pci_poc.context import ContextConfig
from pci_poc.crop_ocr import OCRVariantMetric, run_ocr_variants
from pci_poc.detect import Detection, detect_candidates
from pci_poc.evaluate import (
    complete_number_digests,
    evaluate_boxes,
    evaluate_complete_numbers,
    evaluate_redaction,
)
from pci_poc.hybrid import HybridCandidate, candidates_to_detections, detect_hybrid_candidates
from pci_poc.luhn import append_luhn_check_digit
from pci_poc.ocr_utils import OCRToken, TesseractOCR
from pci_poc.redact import redact_image
from pci_poc.synth import GroundTruthRegion, _font


@dataclass(frozen=True)
class HybridSample:
    image_path: Path
    regions: tuple[GroundTruthRegion, ...]
    condition: str
    template_id: str
    split: str
    ui_header: str
    card_digest: str | None


CONDITIONS = (
    "clear_card",
    "missing_label_card",
    "misleading_negative_card",
    "far_label_card",
    "vertical_label_card",
    "cropped_card_text",
    "cropped_label_card",
    "ocr_confusion_card",
    "conflicting_card_labels",
    "account_luhn_decoy",
    "order_id_luhn_decoy",
    "phone_luhn_decoy",
    "misleading_positive_decoy",
    "conflicting_decoy_labels",
    "negative_only",
)

TEMPLATE_SETS = {
    "development": (
        {"id": "hybrid_dev_a", "size": (1536, 760), "label_x": 92, "value_x": 472, "y": 280, "font": 32, "background": "#edf2f5", "header": "#324f65", "panel": 105, "sidebar": False},
        {"id": "hybrid_dev_b", "size": (1280, 700), "label_x": 66, "value_x": 405, "y": 260, "font": 25, "background": "#eef4f1", "header": "#28675e", "panel": 210, "sidebar": True},
    ),
    "validation": (
        {"id": "hybrid_val_a", "size": (1440, 800), "label_x": 155, "value_x": 545, "y": 325, "font": 38, "background": "#f4f0ed", "header": "#745447", "panel": 145, "sidebar": False},
        {"id": "hybrid_val_b", "size": (1360, 660), "label_x": 105, "value_x": 460, "y": 245, "font": 23, "background": "#eff2f7", "header": "#455b79", "panel": 95, "sidebar": False},
    ),
    "final": (
        {"id": "hybrid_final_a", "size": (1600, 760), "label_x": 125, "value_x": 505, "y": 295, "font": 35, "background": "#f0f3ed", "header": "#456343", "panel": 135, "sidebar": False},
        {"id": "hybrid_final_b", "size": (1320, 720), "label_x": 75, "value_x": 425, "y": 270, "font": 27, "background": "#f4eff5", "header": "#66516f", "panel": 205, "sidebar": True},
    ),
}


def _format_number(number: str, style: int) -> str:
    if style == 0:
        return " ".join(number[index : index + 4] for index in range(0, len(number), 4))
    if style == 1:
        return "-".join(number[index : index + 4] for index in range(0, len(number), 4))
    return number


def _draw_sample(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    template: dict[str, Any],
    condition: str,
    label: str,
    displayed_value: str,
    value_font: ImageFont.ImageFont,
    label_font: ImageFont.ImageFont,
) -> tuple[int, int, int, int]:
    label_x = int(template["label_x"])
    value_x = int(template["value_x"])
    y = int(template["y"])
    if condition == "far_label_card":
        label_x = max(45, int(template["panel"]) + 25)
        value_x = min(image.width - 360, value_x + 470)
    elif condition == "conflicting_card_labels":
        value_x = min(image.width - 400, value_x + 170)

    draw.text((value_x, y), displayed_value, font=value_font, fill="#182c3f")
    value_box = draw.textbbox((value_x, y), displayed_value, font=value_font)
    if condition == "cropped_card_text":
        crop_x = value_x + max(20, int((value_box[2] - value_x) * 0.62))
        draw.rectangle((crop_x, y - 4, value_box[2] + 6, value_box[3] + 5), fill="white")

    if condition != "missing_label_card":
        label_y = y - int(template["font"]) - 30 if condition == "vertical_label_card" else y
        draw.text((label_x, label_y), label, font=label_font, fill="#536273")
        label_box = draw.textbbox((label_x, label_y), label, font=label_font)
        if condition == "cropped_label_card":
            crop_x = label_x + max(16, int((label_box[2] - label_x) * 0.56))
            draw.rectangle((crop_x, label_y - 3, label_box[2] + 4, label_box[3] + 3), fill="white")
        if condition == "conflicting_card_labels":
            draw.text((label_box[2] + 16, label_y), "Account Reference", font=label_font, fill="#536273")
        elif condition == "conflicting_decoy_labels":
            draw.text((label_box[2] + 16, label_y), "Order ID", font=label_font, fill="#536273")
    return tuple(value_box)


def generate_hybrid_datasets(
    output_dir: str | Path,
    *,
    seed: int = 20261012,
    repetitions_per_split: int = 1,
) -> dict[str, list[HybridSample]]:
    """Generate new disjoint dev, validation and final-test UI template sets."""
    if repetitions_per_split < 1:
        raise ValueError("repetitions_per_split must be positive")
    root = Path(output_dir)
    output: dict[str, list[HybridSample]] = {}
    for split_index, (split, templates) in enumerate(TEMPLATE_SETS.items()):
        rng = random.Random(seed + split_index)
        split_root = root / split
        image_dir = split_root / "screenshots"
        image_dir.mkdir(parents=True, exist_ok=True)
        case_order = list(CONDITIONS) * repetitions_per_split
        rng.shuffle(case_order)
        samples: list[HybridSample] = []
        manifest: list[dict[str, Any]] = []

        for index, condition in enumerate(case_order):
            template = templates[index % len(templates)]
            width, height = template["size"]
            image = Image.new("RGB", (width, height), template["background"])
            draw = ImageDraw.Draw(image)
            label_font = _font(int(template["font"]))
            value_font = _font(int(template["font"]) + 3)
            header_font = _font(max(24, int(template["font"]) - 2))
            ui_header = "Support Ticket CRM" if condition == "misleading_ui_context_card" else "Synthetic CRM Account Details"
            if condition == "misleading_ui_context_decoy":
                ui_header = "Payment Details CRM"
            draw.rectangle((0, 0, width, 80), fill=template["header"])
            draw.text((30, 22), ui_header, font=header_font, fill="white")
            panel_left = int(template["panel"])
            draw.rounded_rectangle((panel_left, 115, width - 45, height - 45), radius=8, fill="white", outline="#d4dde4", width=2)
            if template["sidebar"]:
                draw.rectangle((panel_left, 116, panel_left + 125, height - 46), fill="#e6eeeb")
                draw.text((panel_left + 16, 142), "PROFILE", font=label_font, fill="#345650")
            draw.text((panel_left + 155, 140), rng.choice(("Avery Example", "Jordan Synthetic", "Morgan Sample")), font=header_font, fill="#1d3447")

            is_card = condition in {
                "clear_card", "missing_label_card", "misleading_negative_card", "far_label_card",
                "vertical_label_card", "cropped_card_text", "cropped_label_card", "ocr_confusion_card",
                "conflicting_card_labels", "misleading_ui_context_card",
            }
            is_decoy = condition in {
                "account_luhn_decoy", "order_id_luhn_decoy", "phone_luhn_decoy",
                "misleading_positive_decoy", "conflicting_decoy_labels", "misleading_ui_context_decoy",
            }
            truth_class = "pan" if is_card else "decoy" if is_decoy else "negative"
            number = append_luhn_check_digit(rng.choice("456") + "".join(str(rng.randrange(10)) for _ in range(14)))
            style = rng.randrange(3)
            displayed = _format_number(number, style)
            labels = {
                "misleading_negative_card": "Account Reference",
                "account_luhn_decoy": "Account Reference",
                "order_id_luhn_decoy": "Order ID",
                "phone_luhn_decoy": "Phone",
                "misleading_positive_decoy": "Card Number",
                "conflicting_decoy_labels": "Card Number",
                "misleading_ui_context_decoy": "Account Reference",
                "ocr_confusion_card": "Card Nurnber",
            }
            label = labels.get(condition, "Card Number")
            if condition == "negative_only":
                label = "Phone"
                displayed = f"555-010-{rng.randrange(1000, 10000)}"
            value_box = _draw_sample(image, draw, template, condition, label, displayed, value_font, label_font)
            if condition == "negative_only":
                draw.text((int(template["value_x"]), int(template["y"]) + 82), "Case ID", font=label_font, fill="#536273")
                draw.text((int(template["value_x"]) + 180, int(template["y"]) + 78), f"CASE-{rng.randrange(100000, 999999)}", font=value_font, fill="#182c3f")

            regions = [GroundTruthRegion(truth_class, value_box)] if truth_class in {"pan", "decoy", "negative"} else []
            card_digest = hashlib.sha256(number.encode("ascii")).hexdigest() if truth_class == "pan" else None
            filename = f"{split}_{index:04d}.png"
            image_path = image_dir / filename
            image.save(image_path)
            sample = HybridSample(image_path, tuple(regions), condition, str(template["id"]), split, ui_header, card_digest)
            samples.append(sample)
            manifest.append(
                {
                    "image": f"screenshots/{filename}",
                    "synthetic_only": True,
                    "split": split,
                    "condition": condition,
                    "template_id": template["id"],
                    "ui_header": ui_header,
                    "regions": [
                        {
                            "label": region.label,
                            "bbox": list(region.bbox),
                            **({"number_sha256": card_digest} if region.label == "pan" else {}),
                        }
                        for region in regions
                    ],
                    "card_number_sha256": card_digest,
                }
            )

        (split_root / "ground_truth.json").write_text(
            json.dumps({"seed": seed + split_index, "samples": manifest}, indent=2) + "\n",
            encoding="utf-8",
        )
        output[split] = samples
    return output


def _safe_tokens(tokens: Sequence[OCRToken]) -> list[dict[str, Any]]:
    """Make a diagnostic OCR-token record without retaining digit strings."""
    return [
        {
            "text": "<digit-token>" if any(character.isdigit() for character in token.text) else token.text,
            "digit_count": sum(character.isdigit() for character in token.text),
            "bbox": list(token.bbox),
            "confidence": token.confidence,
        }
        for token in tokens
    ]


def _run_crop_diagnostics(
    root: Path,
    *,
    diagnostic_dataset: str | Path,
) -> dict[str, Any]:
    dataset_root = Path(diagnostic_dataset)
    manifest_path = dataset_root / "ground_truth.json"
    if not manifest_path.is_file():
        return {"status": "not_run", "reason": "previous diagnostic robustness dataset not found"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = [item for item in manifest["samples"] if item["condition"] == "cropped_number_card"]
    ocr = TesseractOCR()
    examples: list[dict[str, Any]] = []
    for item in failures:
        image_path = dataset_root / item["image"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        baseline_tokens = ocr.recognize(image)
        variants = run_ocr_variants(image, baseline_tokens, tesseract_cmd=ocr.executable)
        digit_tokens = [token for token in baseline_tokens if any(character.isdigit() for character in token.text)]
        examples.append(
            {
                "image": image_path.name,
                "image_dimensions": list(image.size),
                "ground_truth_boxes": [region["bbox"] for region in item["regions"]],
                "ocr_tokens": _safe_tokens(baseline_tokens),
                "variants": [variant.__dict__ for variant in variants],
                "baseline_numeric_token_boxes": [list(token.bbox) for token in digit_tokens],
            }
        )
    result = {
        "scope": "diagnostic-only crops from prior development evidence; not part of fresh final test",
        "candidate_crop_source": "numeric OCR token boxes, not ground-truth boxes",
        "examples": examples,
    }
    (root / "crop_ocr_diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _metric_summary(rows: Sequence[dict[str, Any]], coverage: Sequence[float], mode: str) -> dict[str, Any]:
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    complete_tp = sum(row["complete_number_tp"] for row in rows)
    complete_fp = sum(row["complete_number_fp"] for row in rows)
    complete_fn = sum(row["complete_number_fn"] for row in rows)
    complete_precision = complete_tp / (complete_tp + complete_fp) if complete_tp + complete_fp else 0.0
    complete_recall = complete_tp / (complete_tp + complete_fn) if complete_tp + complete_fn else 0.0
    latencies = [row["latency_ms"] for row in rows]
    return {
        "mode": mode,
        "evaluated_screenshots": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "complete_number_tp": complete_tp,
        "complete_number_fp": complete_fp,
        "complete_number_fn": complete_fn,
        "complete_number_precision": complete_precision,
        "complete_number_recall": complete_recall,
        "complete_number_f1": 2 * complete_precision * complete_recall / (complete_precision + complete_recall) if complete_precision + complete_recall else 0.0,
        "false_redactions": sum(row["false_redactions"] for row in rows),
        "decoy_false_redactions": sum(row["decoy_false_redactions"] for row in rows),
        "negative_text_redactions": sum(row["negative_text_redactions"] for row in rows),
        "needs_review_count": sum(row["needs_review_count"] for row in rows),
        "needs_review_masked_count": sum(row["needs_review_masked_count"] for row in rows),
        "needs_review_unmasked_count": sum(row["needs_review_unmasked_count"] for row in rows),
        "mean_card_text_coverage": statistics.fmean(coverage) if coverage else 0.0,
        "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "ocr_failures": sum(row["ocr_failed"] for row in rows),
    }


def _classification_summary(
    candidates_by_image: Sequence[tuple[Sequence[HybridCandidate], Sequence[GroundTruthRegion]]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for candidates, regions in candidates_by_image:
        pairs = evaluate_boxes(
            [candidate.bbox for candidate in candidates],
            [region.bbox for region in regions],
        )
        for match in pairs.matches:
            candidate = candidates[match.prediction_index]
            region = regions[match.ground_truth_index]
            counts[f"{region.label}_as_{candidate.context_classification}"] += 1
    return dict(sorted(counts.items()))


def run_hybrid_experiment(
    output_dir: str | Path,
    *,
    seed: int = 20261012,
    repetitions_per_split: int = 1,
    diagnostic_dataset: str | Path = Path("generated") / "robustness_experiment" / "dataset",
    config: ContextConfig | None = None,
) -> dict[str, Any]:
    """Evaluate unchanged detectors and conservative hybrid on fresh split templates."""
    root = Path(output_dir)
    datasets = generate_hybrid_datasets(
        root / "datasets",
        seed=seed,
        repetitions_per_split=repetitions_per_split,
    )
    selected_config = config or ContextConfig()
    ocr = TesseractOCR()
    modes = ("baseline", "context_only", "hybrid")
    output_dirs = {mode: root / f"{mode}_redacted" for mode in modes}
    for output_dir_path in output_dirs.values():
        output_dir_path.mkdir(parents=True, exist_ok=True)

    with Image.open(datasets["development"][0].image_path) as source:
        warm_image = source.convert("RGB")
    warm_tokens = ocr.recognize(warm_image)
    detect_candidates(warm_tokens)
    _contextualize(warm_tokens, selected_config)
    detect_hybrid_candidates(warm_tokens, config=selected_config)

    split_results: dict[str, Any] = {}
    viz_root = root / "visualizations"
    for split, samples in datasets.items():
        mode_rows: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
        mode_coverage: dict[str, list[float]] = {mode: [] for mode in modes}
        classification_pairs: list[tuple[Sequence[HybridCandidate], Sequence[GroundTruthRegion]]] = []
        condition_rows: list[dict[str, Any]] = []
        visualizations: dict[str, str] = {}

        for sample in samples:
            cards = [region for region in sample.regions if region.label == "pan"]
            decoys = [region for region in sample.regions if region.label == "decoy"]
            negative_regions = [region for region in sample.regions if region.label == "negative"]
            condition_results: dict[str, dict[str, Any]] = {}
            for mode in modes:
                with Image.open(sample.image_path) as source:
                    image = source.convert("RGB")
                started = time.perf_counter()
                review_candidates: list[Any] = []
                try:
                    tokens = ocr.recognize(image)
                    ocr_failed = False
                    failure_type = ""
                    if mode == "baseline":
                        detections = detect_candidates(tokens)
                        hybrid_candidates: tuple[HybridCandidate, ...] = ()
                        needs_review = 0
                    elif mode == "context_only":
                        context_result = _contextualize(tokens, selected_config)
                        detections = list(context_result.detections)
                        hybrid_candidates = ()
                        review_candidates = list(context_result.needs_review)
                        needs_review = len(context_result.needs_review)
                    else:
                        ui_header = " ".join(
                            token.text for token in tokens
                            if token.bbox[3] <= 85 and not any(char.isdigit() for char in token.text)
                        )
                        hybrid_result = detect_hybrid_candidates(
                            tokens,
                            config=selected_config,
                            ui_context_text=ui_header,
                        )
                        hybrid_candidates = hybrid_result.candidates
                        detections = candidates_to_detections(hybrid_result)
                        review_candidates = [candidate for candidate in hybrid_candidates if candidate.context_classification == "needs_review"]
                        needs_review = len(review_candidates)
                except Exception as error:
                    tokens = []
                    detections = []
                    hybrid_candidates = ()
                    review_candidates = []
                    needs_review = 0
                    ocr_failed = True
                    failure_type = type(error).__name__
                redacted_path = output_dirs[mode] / split / sample.image_path.name
                redact_image(sample.image_path, detections, redacted_path)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                box_metrics = evaluate_boxes([detection.bbox for detection in detections], [region.bbox for region in cards])
                complete_metrics = evaluate_complete_numbers(
                    complete_number_digests(tokens),
                    [sample.card_digest] if sample.card_digest else [],
                )
                card_redaction = evaluate_redaction(sample.image_path, redacted_path, [region.bbox for region in cards])
                decoy_redaction = evaluate_redaction(sample.image_path, redacted_path, [], [region.bbox for region in decoys])
                negative_redaction = evaluate_redaction(sample.image_path, redacted_path, [], [region.bbox for region in negative_regions])
                review_coverage = evaluate_redaction(
                    sample.image_path,
                    redacted_path,
                    [candidate.bbox for candidate in review_candidates],
                ).card_coverage
                review_masked = sum(value > 0.0 for value in review_coverage)
                mode_coverage[mode].extend(card_redaction.card_coverage)
                row = {
                    "image": sample.image_path.name,
                    "condition": sample.condition,
                    "template_id": sample.template_id,
                    "tp": box_metrics.true_positives,
                    "fp": box_metrics.false_positives,
                    "fn": box_metrics.false_negatives,
                    "complete_number_tp": complete_metrics.true_positives,
                    "complete_number_fp": complete_metrics.false_positives,
                    "complete_number_fn": complete_metrics.false_negatives,
                    "decoy_false_redactions": decoy_redaction.false_redaction_count,
                    "negative_text_redactions": negative_redaction.false_redaction_count,
                    "false_redactions": decoy_redaction.false_redaction_count + negative_redaction.false_redaction_count,
                    "needs_review_count": needs_review,
                    "needs_review_masked_count": review_masked,
                    "needs_review_unmasked_count": len(review_candidates) - review_masked,
                    "card_coverage": statistics.fmean(card_redaction.card_coverage) if card_redaction.card_coverage else 0.0,
                    "ocr_failed": ocr_failed,
                    "ocr_failure_type": failure_type,
                    "latency_ms": elapsed_ms,
                }
                mode_rows[mode].append(row)
                condition_results[mode] = row
                if mode == "hybrid":
                    classification_pairs.append((hybrid_candidates, sample.regions))

                for error_kind, indexes in (("false_positive", box_metrics.false_positive_indices), ("false_negative", box_metrics.false_negative_indices)):
                    key = f"{mode}_{error_kind}"
                    if indexes and key not in visualizations:
                        path = viz_root / split / f"{key}_{sample.condition}_{sample.image_path.stem}.png"
                        candidate_boxes = [detection.bbox for detection in detections]
                        _draw_sanitized_comparison(sample, candidate_boxes, path, false_positive=error_kind == "false_positive")
                        visualizations[key] = str(path.relative_to(root))

            condition_rows.append(
                {
                    "image": sample.image_path.name,
                    "condition": sample.condition,
                    "template_id": sample.template_id,
                    **{f"{mode}_{key}": value for mode, row in condition_results.items() for key, value in row.items() if key not in {"image", "condition", "template_id"}},
                }
            )

        summaries = {
            mode: _metric_summary(mode_rows[mode], mode_coverage[mode], mode)
            for mode in modes
        }
        split_results[split] = {
            "screenshots": len(samples),
            "templates": sorted({sample.template_id for sample in samples}),
            "metrics": summaries,
            "hybrid_classification_counts": _classification_summary(classification_pairs),
            "condition_metrics": condition_rows,
            "visualizations": visualizations,
        }

    crop_diagnostics = _run_crop_diagnostics(root, diagnostic_dataset=diagnostic_dataset)
    report = {
        "scope": "synthetic-only experiment on newly generated disjoint UI templates; not production accuracy",
        "synthetic_only": True,
        "ocr_mode": "real_tesseract",
        "metric_definitions": {
            "tp_fp_fn": "one-to-one bbox localization at IoU >= 0.5 against PAN text boxes; a partial fragment may localize and count as TP",
            "complete_number_recognition": "exact SHA-256 equality of an OCR digit group of length 13-19 with a generated full-number digest; partial fragments never count as complete TPs",
            "card_text_coverage": "fraction of pixels changed by more than 8 levels inside a ground-truth PAN text box",
            "false_redactions": "decoy plus ordinary-negative text boxes with any pixel changed by more than 8 levels; subcounts are separate",
            "needs_review_masking": "review candidate boxes with any changed pixel, measured independently of classification",
        },
        "seed": seed,
        "repetitions_per_split": repetitions_per_split,
        "split_template_sets_disjoint": True,
        "context_config": {
            "max_label_distance": selected_config.max_label_distance,
            "row_tolerance": selected_config.row_tolerance,
            "ambiguous_action": selected_config.ambiguous_action,
            "unlabeled_action": selected_config.unlabeled_action,
        },
        "hybrid_partial_digit_threshold": 8,
        "redaction_policy": "mask all Luhn-valid and 8-19 digit partial candidates; context changes classification, never vetoes masking",
        "latency_scope": "per-image real OCR, classification/detection and redaction; excludes generation and one warmup per mode",
        "crop_ocr_diagnostics": crop_diagnostics,
        "splits": split_results,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for split, result in split_results.items():
        summary_rows = list(result["metrics"].values())
        _write_csv(root / split / "metrics.csv", summary_rows)
        _write_csv(root / split / "condition_metrics.csv", result["condition_metrics"])
    print("Fresh synthetic hybrid experiment; no production accuracy claim.")
    for split in ("development", "validation", "final"):
        print(f"{split.upper()} ({split_results[split]['screenshots']} screenshots; templates={len(split_results[split]['templates'])})")
        for summary in split_results[split]["metrics"].values():
            print(
                f"  {summary['mode']}: TP={summary['tp']} FP={summary['fp']} FN={summary['fn']} "
                f"P={summary['precision']:.3f} R={summary['recall']:.3f} F1={summary['f1']:.3f} "
                f"false_redactions={summary['false_redactions']} review={summary['needs_review_count']} "
                f"coverage={summary['mean_card_text_coverage']:.3f} "
                f"median={summary['median_latency_ms']:.1f}ms p95={summary['p95_latency_ms']:.1f}ms "
                f"OCR_failures={summary['ocr_failures']}"
            )
        if split == "final":
            print(f"  Hybrid classification counts: {split_results[split]['hybrid_classification_counts']}")
    print(f"Fresh reports saved under {root}.")
    return report


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("empty\n")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("generated") / "hybrid_policy_experiment")
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--repetitions-per-split", type=int, default=1)
    parser.add_argument("--diagnostic-dataset", type=Path, default=Path("generated") / "robustness_experiment" / "dataset")
    arguments = parser.parse_args()
    run_hybrid_experiment(
        arguments.output_dir,
        seed=arguments.seed,
        repetitions_per_split=arguments.repetitions_per_split,
        diagnostic_dataset=arguments.diagnostic_dataset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())