import numpy as np
from PIL import Image

from pci_poc.context import ContextConfig
from pci_poc.hybrid import candidates_to_detections, detect_hybrid_candidates
from pci_poc.luhn import append_luhn_check_digit
from pci_poc.ocr_utils import OCRToken
from pci_poc.redact import redact_image
from pci_poc.evaluate import evaluate_redaction


def _tokens(label, number, *, number_x=400, number_y=40):
    tokens = []
    label_x = 10
    for word in label.split():
        width = len(word) * 9
        tokens.append(OCRToken(word, (label_x, number_y, label_x + width, number_y + 18)))
        label_x += width + 5
    if number:
        for part in (number[:4], number[4:8], number[8:12], number[12:]):
            if not part:
                continue
            width = len(part) * 10
            tokens.append(OCRToken(part, (number_x, number_y, number_x + width, number_y + 18)))
            number_x += width + 5
    return tokens


def test_negative_labels_classify_as_decoy_but_do_not_veto_mask():
    valid = append_luhn_check_digit("400000123456789")
    for label in ("Account Reference", "Order ID", "Phone"):
        result = detect_hybrid_candidates(_tokens(label, valid))
        candidate = result.candidates[0]

        assert candidate.candidate_detected
        assert candidate.context_classification == "likely_decoy"
        assert candidate.redaction_decision == "mask"
        assert len(candidates_to_detections(result)) == 1
        assert "mask_retained" in candidate.reason


def test_positive_label_classifies_card_and_masks():
    valid = append_luhn_check_digit("400000123456789")
    result = detect_hybrid_candidates(_tokens("Card Number", valid))

    assert result.candidates[0].context_classification == "card_like"
    assert result.candidates[0].redaction_decision == "mask"


def test_missing_and_conflicting_context_remain_reviewable_and_masked():
    valid = append_luhn_check_digit("400000123456789")
    missing = detect_hybrid_candidates(_tokens("", valid))
    mixed = _tokens("Card Number", valid)
    mixed.extend([OCRToken("Account", (120, 41, 180, 59)), OCRToken("Reference", (185, 41, 265, 59))])
    conflict = detect_hybrid_candidates(mixed)

    assert missing.candidates[0].context_classification == "needs_review"
    assert conflict.candidates[0].context_classification == "needs_review"
    assert len(candidates_to_detections(missing)) == 1
    assert len(candidates_to_detections(conflict)) == 1


def test_partial_cropped_digit_sequence_is_reviewed_and_masked():
    tokens = [
        OCRToken("Card", (10, 40, 45, 58)),
        OCRToken("Number", (50, 40, 100, 58)),
        OCRToken("12345678", (400, 40, 480, 58)),
    ]
    result = detect_hybrid_candidates(tokens)

    assert result.valid_luhn_candidate_count == 0
    assert result.partial_candidate_count == 1
    assert result.candidates[0].source == "partial_digit_sequence"
    assert result.candidates[0].context_classification == "needs_review"
    assert result.candidates[0].redaction_decision == "mask"
    assert len(candidates_to_detections(result)) == 1


def test_ui_context_conflict_does_not_suppress_candidate_mask():
    valid = append_luhn_check_digit("400000123456789")
    result = detect_hybrid_candidates(
        _tokens("Card Number", valid),
        config=ContextConfig(),
        ui_context_text="Support Ticket",
    )

    assert result.candidates[0].context_classification == "needs_review"
    assert result.candidates[0].redaction_decision == "mask"


def test_needs_review_and_misleading_negative_label_candidates_are_pixel_masked(tmp_path):
    valid = append_luhn_check_digit("400000123456789")
    for index, label in enumerate(("", "Account Reference")):
        tokens = _tokens(label, valid)
        result = detect_hybrid_candidates(tokens)
        source = tmp_path / f"source_{index}.png"
        destination = tmp_path / f"redacted_{index}.png"
        pixels = np.full((80, 620, 3), 255, dtype=np.uint8)
        pixels[40:58, 400:600] = 160
        Image.fromarray(pixels).save(source)
        redact_image(source, candidates_to_detections(result), destination)
        coverage = evaluate_redaction(source, destination, [result.candidates[0].bbox])

        assert result.candidates[0].redaction_decision == "mask"
        assert coverage.card_coverage[0] > 0.99