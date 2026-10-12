from pci_poc.context import ContextConfig, detect_with_context
from pci_poc.luhn import append_luhn_check_digit
from pci_poc.ocr_utils import OCRToken


def _field_tokens(label, *, x=400, y=40, number=None):
    number = number or append_luhn_check_digit("400000123456789")
    tokens = []
    position = 10
    for word in label.split():
        width = len(word) * 9
        tokens.append(OCRToken(word, (position, y, position + width, y + 18)))
        position += width + 5
    for part in (number[:4], number[4:8], number[8:12], number[12:]):
        width = len(part) * 10
        tokens.append(OCRToken(part, (x, y, x + width, y + 18)))
        x += width + 6
    return tokens


def test_positive_card_label_accepts_candidate():
    result = detect_with_context(_field_tokens("Card Number"))

    assert len(result.detections) == 1
    assert result.detections[0].reason == "context_positive_field_label"
    assert result.needs_review == ()


def test_negative_account_reference_label_rejects_luhn_valid_decoy():
    valid = append_luhn_check_digit("400000123456789")
    tokens = _field_tokens("Account Reference", number=valid)

    result = detect_with_context(tokens)

    assert result.detections == ()
    assert result.rejected_count == 1
    assert result.needs_review == ()


def test_order_id_and_phone_labels_reject_luhn_valid_candidates():
    for label in ("Order ID", "Phone"):
        result = detect_with_context(_field_tokens(label))
        assert result.detections == ()
        assert result.rejected_count == 1


def test_missing_label_routes_candidate_to_review_by_default():
    valid = append_luhn_check_digit("400000123456789")
    result = detect_with_context(_field_tokens("", number=valid))

    assert result.detections == ()
    assert len(result.needs_review) == 1
    assert result.needs_review[0].reason == "missing_positive_field_label"


def test_ocr_label_error_is_normalized_and_unrecognized_error_is_reviewed():
    recognized_ocr_typo = detect_with_context(_field_tokens("Card Nurnber"))
    unrecognized_ocr_error = detect_with_context(_field_tokens("Crd Nmbrr"))

    assert len(recognized_ocr_typo.detections) == 1
    assert len(unrecognized_ocr_error.needs_review) == 1


def test_mixed_positive_and_negative_labels_follow_review_precedence():
    tokens = _field_tokens("Card Number")
    tokens.extend(
        [
            OCRToken("Account", (120, 41, 180, 59)),
            OCRToken("Reference", (185, 41, 265, 59)),
        ]
    )
    result = detect_with_context(tokens)

    assert result.detections == ()
    assert len(result.needs_review) == 1
    assert result.needs_review[0].reason == "conflicting_label_and_ui_evidence"


def test_ui_negative_conflict_is_reviewed_and_ui_alone_does_not_promote():
    positive_conflict = detect_with_context(
        _field_tokens("Card Number"),
        ui_context_text="Support Ticket",
    )
    ui_only = detect_with_context(
        _field_tokens(""),
        ui_context_text="Payment Details",
    )

    assert len(positive_conflict.needs_review) == 1
    assert ui_only.detections == ()
    assert len(ui_only.needs_review) == 1


def test_label_distance_is_configurable():
    tokens = _field_tokens("Card Number", x=600)
    close = detect_with_context(tokens, config=ContextConfig(max_label_distance=600))
    far = detect_with_context(tokens, config=ContextConfig(max_label_distance=40))

    assert len(close.detections) == 1
    assert len(far.needs_review) == 1