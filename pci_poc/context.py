"""Optional OCR-label context for classifying Luhn-valid candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from pci_poc.detect import Detection, group_nearby_tokens, normalize_candidate
from pci_poc.luhn import is_luhn_valid
from pci_poc.ocr_utils import OCRToken


AmbiguousAction = Literal["needs_review", "accept", "reject"]


@dataclass(frozen=True)
class ContextConfig:
    """Configurable label proximity, phrases, weak UI terms and conflict policy."""

    max_label_distance: int = 360
    row_tolerance: int = 22
    positive_labels: tuple[str, ...] = (
        "card number",
        "card no",
        "credit card",
        "debit card",
        "payment card",
        "payment method",
    )
    negative_labels: tuple[str, ...] = (
        "account reference",
        "legacy reference",
        "order id",
        "order number",
        "phone",
        "telephone",
    )
    positive_ui_context: tuple[str, ...] = (
        "payment details",
        "card details",
        "billing payment",
    )
    negative_ui_context: tuple[str, ...] = (
        "support ticket",
        "order lookup",
    )
    ambiguous_action: AmbiguousAction = "needs_review"
    unlabeled_action: AmbiguousAction = "needs_review"

    def __post_init__(self) -> None:
        if self.max_label_distance < 0 or self.row_tolerance < 0:
            raise ValueError("label distance and row tolerance must be non-negative")
        if self.ambiguous_action not in {"needs_review", "accept", "reject"}:
            raise ValueError("invalid ambiguous_action")
        if self.unlabeled_action not in {"needs_review", "accept", "reject"}:
            raise ValueError("invalid unlabeled_action")


@dataclass(frozen=True)
class ReviewCandidate:
    """A valid candidate withheld from automatic redaction for human review."""

    bbox: tuple[int, int, int, int]
    reason: str
    nearby_label: str
    token_count: int


@dataclass(frozen=True)
class ContextDetectionResult:
    """Accepted detections, needs-review candidates and rejected count."""

    detections: tuple[Detection, ...]
    needs_review: tuple[ReviewCandidate, ...]
    rejected_count: int


def _normalize_label(text: str) -> str:
    text = text.lower().replace("rn", "m").replace("0", "o").replace("1", "l")
    return re.sub(r"[^a-z]+", " ", text).strip()


def _has_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(f" {_normalize_label(phrase)} " in f" {text} " for phrase in phrases)


def _nearby_label(
    tokens: list[OCRToken],
    candidate_box: tuple[int, int, int, int],
    config: ContextConfig,
) -> str:
    candidate_center_y = (candidate_box[1] + candidate_box[3]) / 2
    words = [
        token
        for token in tokens
        if token.bbox[2] <= candidate_box[0]
        and candidate_box[0] - token.bbox[2] <= config.max_label_distance
        and abs((token.bbox[1] + token.bbox[3]) / 2 - candidate_center_y)
        <= config.row_tolerance
        and any(character.isalpha() for character in token.text)
    ]
    return _normalize_label(" ".join(token.text for token in sorted(words, key=lambda item: item.bbox[0])))


def detect_with_context(
    tokens: Iterable[OCRToken],
    *,
    config: ContextConfig | None = None,
    ui_context_text: str = "",
) -> ContextDetectionResult:
    """Classify Luhn-valid groups using nearby field labels and weak UI context.

    Precedence: a nearby negative label vetoes automatic acceptance; simultaneous
    positive and negative field labels use ``ambiguous_action``. A positive field
    label accepts only when no negative field/UI evidence conflicts. UI context by
    itself never promotes a candidate. With no field label, ``unlabeled_action``
    controls the outcome and defaults to needs-review.
    """
    selected_config = config or ContextConfig()
    token_list = list(tokens)
    normalized_ui = _normalize_label(ui_context_text)
    ui_positive = _has_phrase(normalized_ui, selected_config.positive_ui_context)
    ui_negative = _has_phrase(normalized_ui, selected_config.negative_ui_context)
    accepted: list[Detection] = []
    review: list[ReviewCandidate] = []
    rejected_count = 0

    for group in group_nearby_tokens(token_list):
        candidate = normalize_candidate(group.text)
        if not candidate.isascii() or not candidate.isdigit() or not is_luhn_valid(candidate):
            continue

        label = _nearby_label(token_list, group.bbox, selected_config)
        positive = _has_phrase(label, selected_config.positive_labels)
        negative = _has_phrase(label, selected_config.negative_labels)
        conflicting = (positive and negative) or (positive and ui_negative) or (negative and ui_positive)

        if conflicting:
            action = selected_config.ambiguous_action
            reason = "conflicting_label_and_ui_evidence"
        elif negative:
            action = "reject"
            reason = "negative_field_label"
        elif positive:
            action = "accept"
            reason = "positive_field_label"
        else:
            action = selected_config.unlabeled_action
            reason = "missing_positive_field_label"
            if ui_positive or ui_negative:
                reason = "ui_context_without_field_label"

        if action == "accept":
            accepted.append(Detection(group.bbox, f"context_{reason}", group.token_count))
        elif action == "needs_review":
            review.append(ReviewCandidate(group.bbox, reason, label, group.token_count))
        else:
            rejected_count += 1

    return ContextDetectionResult(tuple(accepted), tuple(review), rejected_count)