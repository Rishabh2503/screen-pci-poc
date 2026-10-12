import numpy as np
import pytest
from PIL import Image

from pci_poc.evaluate import (
    complete_number_digests,
    evaluate_boxes,
    evaluate_complete_numbers,
    evaluate_redaction,
    precision_confidence_interval,
    recall_confidence_interval,
    wilson_confidence_interval,
)
from pci_poc.ocr_utils import OCRToken


def test_metric_formulas_for_partial_detection():
    result = evaluate_boxes(
        [(0, 0, 10, 10), (20, 20, 30, 30)],
        [(0, 0, 10, 10), (40, 40, 50, 50)],
    )

    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 1, 1)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)


def test_empty_predictions_and_no_card_screenshot():
    missed = evaluate_boxes([], [(1, 1, 4, 4)])
    empty = evaluate_boxes([], [])

    assert (missed.true_positives, missed.false_positives, missed.false_negatives) == (0, 0, 1)
    assert (missed.precision, missed.recall, missed.f1) == (0.0, 0.0, 0.0)
    assert (empty.true_positives, empty.false_positives, empty.false_negatives) == (0, 0, 0)
    assert (empty.precision, empty.recall, empty.f1) == (0.0, 0.0, 0.0)


def test_duplicate_predictions_match_ground_truth_only_once():
    result = evaluate_boxes(
        [(2, 2, 12, 12), (2, 2, 12, 12)],
        [(2, 2, 12, 12)],
    )

    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 1, 0)
    assert result.false_positive_indices == (1,)


def test_matching_maximizes_pair_count_instead_of_greedy_iou(monkeypatch):
    import pci_poc.evaluate as evaluation

    overlap_by_pair = {
        ((0, 0, 1, 1), (10, 10, 11, 11)): 0.90,
        ((0, 0, 1, 1), (20, 20, 21, 21)): 0.80,
        ((2, 2, 3, 3), (10, 10, 11, 11)): 0.85,
    }
    monkeypatch.setattr(
        evaluation,
        "intersection_over_union",
        lambda predicted, expected: overlap_by_pair.get((tuple(predicted), tuple(expected)), 0.0),
    )

    result = evaluate_boxes(
        [(0, 0, 1, 1), (2, 2, 3, 3)],
        [(10, 10, 11, 11), (20, 20, 21, 21)],
    )

    assert result.true_positives == 2
    assert {(match.prediction_index, match.ground_truth_index) for match in result.matches} == {
        (0, 1),
        (1, 0),
    }


def test_invalid_boxes_are_counted_as_fp_or_fn():
    result = evaluate_boxes(
        [(0, 0, 0, 1), (float("nan"), 0, 1, 1)],
        [(0, 0, 1, 0), (2, 2, 3, 3)],
    )

    assert (result.true_positives, result.false_positives, result.false_negatives) == (0, 2, 2)
    assert result.invalid_prediction_count == 2
    assert result.invalid_ground_truth_count == 1


def test_iou_threshold_must_be_valid():
    with pytest.raises(ValueError):
        evaluate_boxes([], [], iou_threshold=1.1)


def test_redaction_coverage_and_false_redaction_count():
    original = Image.new("RGB", (30, 20), "white")
    redacted = np.asarray(original).copy()
    redacted[2:10, 3:12] = 0
    redacted[12:18, 20:28] = 0

    result = evaluate_redaction(
        original,
        Image.fromarray(redacted),
        [(3, 2, 12, 10), (float("nan"), 0, 5, 5)],
        [(20, 12, 28, 18), (0, 0, 2, 2)],
    )

    assert result.card_coverage == (1.0, 0.0)
    assert result.decoy_coverage == (1.0, 0.0)
    assert result.false_redaction_count == 1


def test_redaction_images_must_have_equal_dimensions():
    with pytest.raises(ValueError, match="dimensions must match"):
        evaluate_redaction(Image.new("RGB", (2, 2)), Image.new("RGB", (3, 2)), [])


def test_redaction_compares_per_card_coverage_thresholds():
    original = Image.new("RGB", (10, 10), "white")
    redacted = np.asarray(original).copy()
    redacted[:] = 255
    redacted[0:10, 0:10] = 0
    for row, col in [(9, 9), (9, 8), (8, 9), (8, 8), (9, 7), (7, 9)]:
        redacted[row, col] = 255

    result = evaluate_redaction(
        original,
        Image.fromarray(redacted),
        [(0, 0, 10, 10), (0, 0, 10, 10)],
    )

    assert result.card_coverage == pytest.approx((0.94, 0.94))
    assert result.minimum_card_coverage == pytest.approx(0.94)
    assert result.card_coverage_below_100 == 2
    assert result.card_coverage_below_95 == 2


def test_wilson_confidence_interval_for_precision_and_recall():
    assert wilson_confidence_interval(0, 10) == pytest.approx((0.0, 0.2775401687666165))
    assert wilson_confidence_interval(10, 10) == pytest.approx((0.7224598312333834, 1.0))
    assert precision_confidence_interval(8, 2) == pytest.approx((0.49015684672072346, 0.9433190520193067))
    assert recall_confidence_interval(2, 8) == pytest.approx((0.05668094798069332, 0.5098431532792765))


def test_partial_numeric_fragment_is_not_complete_number_recognition():
    partial_tokens = [OCRToken("1234567890", (0, 0, 100, 20))]
    truth_digest = complete_number_digests(
        [OCRToken("4000001234567899", (0, 0, 160, 20))]
    )

    assert complete_number_digests(partial_tokens) == ()
    result = evaluate_complete_numbers(complete_number_digests(partial_tokens), truth_digest)
    assert (result.true_positives, result.false_positives, result.false_negatives) == (0, 0, 1)
    assert result.recall == 0.0


def test_partial_box_can_localize_but_is_not_a_complete_number_tp():
    full_tokens = [OCRToken("4000001234567899", (0, 0, 160, 20))]
    partial_tokens = [OCRToken("4000001234", (0, 0, 100, 20))]
    ground_truth_box = (0, 0, 160, 20)

    localized = evaluate_boxes([partial_tokens[0].bbox], [ground_truth_box])
    complete = evaluate_complete_numbers(
        complete_number_digests(partial_tokens),
        complete_number_digests(full_tokens),
    )

    assert localized.true_positives == 1
    assert localized.matches[0].iou == pytest.approx(0.625)
    assert complete.true_positives == 0
    assert complete.false_negatives == 1


def test_full_pixel_coverage_does_not_imply_complete_number_recognition():
    original = Image.new("RGB", (40, 20), "white")
    redacted = np.asarray(original).copy()
    redacted[0:20, 0:40] = 0
    coverage = evaluate_redaction(original, Image.fromarray(redacted), [(0, 0, 40, 20)])
    recognition = evaluate_complete_numbers([], ["synthetic-truth-digest"])

    assert coverage.card_coverage == (1.0,)
    assert recognition.true_positives == 0
    assert recognition.false_negatives == 1


def test_false_positive_box_and_false_redaction_have_different_denominators():
    result = evaluate_boxes([(40, 40, 60, 60)], [(0, 0, 20, 20)])
    original = Image.new("RGB", (80, 80), "white")
    redacted = np.asarray(original).copy()
    redacted[40:60, 40:60] = 0
    redaction = evaluate_redaction(
        original,
        Image.fromarray(redacted),
        [(0, 0, 20, 20)],
        [(65, 65, 75, 75)],
    )

    assert result.false_positives == 1
    assert redaction.false_redaction_count == 0


def test_exact_complete_number_match_is_one_to_one_and_duplicate_is_false_positive():
    tokens = [OCRToken("4000001234567899", (0, 0, 160, 20))]
    digest = complete_number_digests(tokens)

    result = evaluate_complete_numbers(digest + digest, digest)

    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 1, 0)
    assert result.precision == 0.5
    assert result.recall == 1.0