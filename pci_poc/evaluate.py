"""Bounding-box and image-pixel evaluation for synthetic redaction runs."""

from __future__ import annotations

import math
import hashlib
from collections import Counter
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Sequence, TypeAlias

import numpy as np
from PIL import Image

from pci_poc.detect import group_nearby_tokens, intersection_over_union, normalize_candidate
from pci_poc.ocr_utils import BoundingBox, OCRToken


BoxLike: TypeAlias = Sequence[Real]
ImageSource: TypeAlias = Image.Image | str | Path


@dataclass(frozen=True)
class BoxMatch:
    """One prediction/ground-truth pairing selected by maximum matching."""

    prediction_index: int
    ground_truth_index: int
    iou: float


@dataclass(frozen=True)
class BoxEvaluation:
    """Detection counts, derived metrics and unmatched box indexes."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    matches: tuple[BoxMatch, ...]
    false_positive_indices: tuple[int, ...]
    false_negative_indices: tuple[int, ...]
    invalid_prediction_count: int
    invalid_ground_truth_count: int


@dataclass(frozen=True)
class RedactionEvaluation:
    """Changed-pixel coverage for each card and decoy text box."""

    card_coverage: tuple[float, ...]
    decoy_coverage: tuple[float, ...]
    false_redaction_count: int
    minimum_card_coverage: float
    card_coverage_below_100: int
    card_coverage_below_95: int


@dataclass(frozen=True)
class CompleteNumberEvaluation:
    """Exact full-number recognition counts computed from one-way digests."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def complete_number_digests(tokens: Sequence[OCRToken]) -> tuple[str, ...]:
    """Hash full 13-19 digit OCR groups; never return or log recognized digits.

    This metric is deliberately independent of bbox IoU and redaction pixels.
    Short partial fragments cannot qualify as complete-number recognition.
    """
    digests: list[str] = []
    for group in group_nearby_tokens(tokens):
        candidate = normalize_candidate(group.text)
        if candidate.isascii() and candidate.isdigit() and 13 <= len(candidate) <= 19:
            digests.append(hashlib.sha256(candidate.encode("ascii")).hexdigest())
    return tuple(digests)


def evaluate_complete_numbers(
    predicted_digests: Sequence[str],
    ground_truth_digests: Sequence[str],
) -> CompleteNumberEvaluation:
    """Exact one-to-one matching of full-number digests; inputs/outputs expose no PANs."""
    available_truth = Counter(ground_truth_digests)
    true_positives = 0
    false_positives = 0
    for digest in predicted_digests:
        if available_truth[digest] > 0:
            available_truth[digest] -= 1
            true_positives += 1
        else:
            false_positives += 1
    false_negatives = sum(available_truth.values())
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return CompleteNumberEvaluation(
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1,
    )


def is_valid_box(box: object) -> bool:
    """Return whether box is finite x1,y1,x2,y2 coordinates with positive area."""
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
        return False
    if any(not isinstance(value, Real) or isinstance(value, bool) for value in box):
        return False
    x1, y1, x2, y2 = (float(value) for value in box)
    return all(math.isfinite(value) for value in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def wilson_confidence_interval(
    successes: int,
    total: int,
    *,
    z_score: float = 1.96,
) -> tuple[float, float]:
    """Return a Wilson 95% score interval for a binomial proportion.

    The interval is clamped to the valid probability range [0, 1]. When there are
    no trials, the interval is (0.0, 0.0).
    """
    if total <= 0:
        return (0.0, 0.0)
    if successes < 0 or successes > total:
        raise ValueError("successes must be between 0 and total")
    if not math.isfinite(z_score) or z_score <= 0.0:
        raise ValueError("z_score must be finite and positive")

    proportion = successes / total
    denominator = 1.0 + (z_score ** 2) / total
    center = proportion + (z_score ** 2) / (2 * total)
    margin = z_score * math.sqrt(
        (proportion * (1.0 - proportion)) / total + (z_score ** 2) / (4 * total ** 2)
    )
    lower = max(0.0, (center - margin) / denominator)
    upper = min(1.0, (center + margin) / denominator)
    return lower, upper


def precision_confidence_interval(
    true_positives: int,
    false_positives: int,
    *,
    z_score: float = 1.96,
) -> tuple[float, float]:
    """Wilson 95% confidence interval for precision."""
    total = true_positives + false_positives
    return wilson_confidence_interval(true_positives, total, z_score=z_score)


def recall_confidence_interval(
    true_positives: int,
    false_negatives: int,
    *,
    z_score: float = 1.96,
) -> tuple[float, float]:
    """Wilson 95% confidence interval for recall."""
    total = true_positives + false_negatives
    return wilson_confidence_interval(true_positives, total, z_score=z_score)


def evaluate_boxes(
    predictions: Sequence[BoxLike],
    ground_truth: Sequence[BoxLike],
    *,
    iou_threshold: float = 0.5,
) -> BoxEvaluation:
    """Match boxes one-to-one at the IoU threshold and calculate precision/recall/F1.

    Invalid predictions count as false positives. Invalid ground-truth boxes count
    as false negatives so malformed annotations cannot silently improve recall.
    Duplicate predictions remain separate and unmatched duplicates count as FPs.
    """
    if not math.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be finite and between 0 and 1")

    valid_predictions = [index for index, box in enumerate(predictions) if is_valid_box(box)]
    valid_truth = [index for index, box in enumerate(ground_truth) if is_valid_box(box)]
    invalid_prediction_indexes = tuple(
        index for index, box in enumerate(predictions) if not is_valid_box(box)
    )
    invalid_truth_indexes = tuple(index for index, box in enumerate(ground_truth) if not is_valid_box(box))

    edges: dict[int, list[tuple[int, float]]] = {}
    for prediction_index in valid_predictions:
        eligible = []
        for truth_index in valid_truth:
            overlap = intersection_over_union(
                predictions[prediction_index],  # type: ignore[arg-type]
                ground_truth[truth_index],  # type: ignore[arg-type]
            )
            if overlap >= iou_threshold:
                eligible.append((truth_index, overlap))
        edges[prediction_index] = sorted(eligible, key=lambda edge: (-edge[1], edge[0]))

    truth_to_prediction: dict[int, tuple[int, float]] = {}

    def assign(prediction_index: int, visited_truth: set[int]) -> bool:
        for truth_index, overlap in edges[prediction_index]:
            if truth_index in visited_truth:
                continue
            visited_truth.add(truth_index)
            previous = truth_to_prediction.get(truth_index)
            if previous is None or assign(previous[0], visited_truth):
                truth_to_prediction[truth_index] = (prediction_index, overlap)
                return True
        return False

    prediction_order = sorted(
        valid_predictions,
        key=lambda index: (len(edges[index]), -max((edge[1] for edge in edges[index]), default=0.0), index),
    )
    for prediction_index in prediction_order:
        assign(prediction_index, set())

    matches = tuple(
        sorted(
            (
                BoxMatch(prediction_index, truth_index, overlap)
                for truth_index, (prediction_index, overlap) in truth_to_prediction.items()
            ),
            key=lambda match: (match.prediction_index, match.ground_truth_index),
        )
    )
    matched_predictions = {match.prediction_index for match in matches}
    matched_truth = {match.ground_truth_index for match in matches}
    false_positive_indexes = tuple(
        sorted(
            set(invalid_prediction_indexes)
            | (set(valid_predictions) - matched_predictions)
        )
    )
    false_negative_indexes = tuple(
        sorted(set(invalid_truth_indexes) | (set(valid_truth) - matched_truth))
    )

    true_positives = len(matches)
    false_positives = len(false_positive_indexes)
    false_negatives = len(false_negative_indexes)
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = _ratio(2 * precision * recall, precision + recall)
    return BoxEvaluation(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        matches=matches,
        false_positive_indices=false_positive_indexes,
        false_negative_indices=false_negative_indexes,
        invalid_prediction_count=len(invalid_prediction_indexes),
        invalid_ground_truth_count=len(invalid_truth_indexes),
    )


def _image_array(source: ImageSource) -> np.ndarray:
    if isinstance(source, Image.Image):
        return np.asarray(source.convert("RGB"))
    with Image.open(source) as image:
        return np.asarray(image.convert("RGB")).copy()


def _box_coverage(changed_pixels: np.ndarray, box: object) -> float:
    if not is_valid_box(box):
        return 0.0
    height, width = changed_pixels.shape
    x1 = max(0, min(width, math.floor(float(box[0]))))  # type: ignore[index]
    y1 = max(0, min(height, math.floor(float(box[1]))))  # type: ignore[index]
    x2 = max(0, min(width, math.ceil(float(box[2]))))  # type: ignore[index]
    y2 = max(0, min(height, math.ceil(float(box[3]))))  # type: ignore[index]
    if x1 >= x2 or y1 >= y2:
        return 0.0
    return float(np.count_nonzero(changed_pixels[y1:y2, x1:x2]) / ((x2 - x1) * (y2 - y1)))


def evaluate_redaction(
    original: ImageSource,
    redacted: ImageSource,
    card_boxes: Sequence[BoxLike],
    decoy_boxes: Sequence[BoxLike] = (),
    *,
    pixel_threshold: int = 8,
) -> RedactionEvaluation:
    """Measure changed-pixel coverage; count decoys with any changed pixels.

    A pixel is considered redacted when any RGB channel differs by more than
    ``pixel_threshold``. Coverage is measured within each supplied text box.
    """
    if not 0 <= pixel_threshold <= 255:
        raise ValueError("pixel_threshold must be between 0 and 255")
    original_array = _image_array(original)
    redacted_array = _image_array(redacted)
    if original_array.shape != redacted_array.shape:
        raise ValueError("original and redacted image dimensions must match")
    changed_pixels = np.any(
        np.abs(original_array.astype(np.int16) - redacted_array.astype(np.int16)) > pixel_threshold,
        axis=2,
    )
    card_coverage = tuple(_box_coverage(changed_pixels, box) for box in card_boxes)
    decoy_coverage = tuple(_box_coverage(changed_pixels, box) for box in decoy_boxes)
    minimum_card_coverage = min(card_coverage, default=0.0)
    card_coverage_below_100 = sum(coverage < 1.0 for coverage in card_coverage)
    card_coverage_below_95 = sum(coverage < 0.95 for coverage in card_coverage)
    return RedactionEvaluation(
        card_coverage=card_coverage,
        decoy_coverage=decoy_coverage,
        false_redaction_count=sum(coverage > 0.0 for coverage in decoy_coverage),
        minimum_card_coverage=minimum_card_coverage,
        card_coverage_below_100=card_coverage_below_100,
        card_coverage_below_95=card_coverage_below_95,
    )