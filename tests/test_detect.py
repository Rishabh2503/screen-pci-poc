from pci_poc.detect import (
    detect_candidates,
    group_nearby_tokens,
    intersection_over_union,
    normalize_candidate,
)
from pci_poc.luhn import append_luhn_check_digit
from pci_poc.ocr_utils import OCRToken


def _tokens_for_number(number, *, y=20):
    fragments = (number[:4], "-", number[4:8], number[8:12], number[12:])
    tokens = []
    x = 10
    for fragment in fragments:
        width = 8 if fragment == "-" else len(fragment) * 10
        tokens.append(OCRToken(fragment, (x, y, x + width, y + 18)))
        x += width + 4
    return tokens


def test_groups_spaced_and_hyphenated_tokens_with_union_box():
    number = append_luhn_check_digit("400000123456789")
    tokens = _tokens_for_number(number)
    groups = group_nearby_tokens(tokens)

    assert len(groups) == 1
    assert normalize_candidate(groups[0].text) == number
    assert groups[0].bbox == (10, 20, tokens[-1].bbox[2], 38)


def test_detects_candidate_and_matches_ground_truth_box():
    number = append_luhn_check_digit("400000123456789")
    tokens = _tokens_for_number(number)
    detections = detect_candidates(tokens)

    assert len(detections) == 1
    assert detections[0].reason == "length_and_luhn_valid"
    assert not hasattr(detections[0], "candidate")
    assert intersection_over_union(detections[0].bbox, (10, 20, tokens[-1].bbox[2], 38)) == 1.0


def test_rejects_decoy_negative_and_separate_rows():
    valid = append_luhn_check_digit("400000123456789")
    decoy = valid[:-1] + str((int(valid[-1]) + 1) % 10)
    tokens = _tokens_for_number(decoy)
    tokens.extend(_tokens_for_number("123456789012", y=80))
    tokens.append(OCRToken("account", (10, 120, 90, 138)))

    assert detect_candidates(tokens) == []