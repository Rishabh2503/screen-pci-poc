"""Conservative candidate protection with separate semantic context labels."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from pci_poc.context import ContextConfig
from pci_poc.detect import Detection, detect_candidates, group_nearby_tokens, normalize_candidate
from pci_poc.luhn import is_luhn_valid
from pci_poc.ocr_utils import BoundingBox, OCRToken


ContextClass = Literal["card_like", "likely_decoy", "needs_review"]


@dataclass(frozen=True)
class HybridCandidate:
    """Candidate, semantic classification and independent conservative decision."""

    candidate_detected: bool
    bbox: BoundingBox
    source: Literal["luhn_valid", "partial_digit_sequence"]
    context_classification: ContextClass
    redaction_decision: Literal["mask"]
    reason: str
    nearby_label: str
    token_count: int


@dataclass(frozen=True)
class HybridDetectionResult:
    """All candidates and the number of partial numeric groups considered."""

    candidates: tuple[HybridCandidate, ...]
    valid_luhn_candidate_count: int
    partial_candidate_count: int


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z]+", " ", text.lower().replace("rn", "m").replace("0", "o").replace("1", "l")).strip()


def _phrase_match(text: str, phrases: tuple[str, ...]) -> bool:
    return any(f" {_normalize(phrase)} " in f" {text} " for phrase in phrases)


def _context_evidence(
    tokens: list[OCRToken],
    bbox: BoundingBox,
    config: ContextConfig,
) -> tuple[str, bool, bool]:
    nearby: list[OCRToken] = []
    candidate_x1, candidate_y1, _, _ = bbox
    candidate_center_y = (bbox[1] + bbox[3]) / 2
    for token in tokens:
        if not any(character.isalpha() for character in token.text):
            continue
        horizontal_gap = candidate_x1 - token.bbox[2]
        same_row = (
            horizontal_gap >= 0
            and horizontal_gap <= config.max_label_distance
            and abs((token.bbox[1] + token.bbox[3]) / 2 - candidate_center_y) <= config.row_tolerance
        )
        above = (
            token.bbox[3] <= candidate_y1
            and candidate_y1 - token.bbox[3] <= config.max_label_distance
            and abs((token.bbox[0] + token.bbox[2]) / 2 - (bbox[0] + bbox[2]) / 2)
            <= config.max_label_distance
        )
        if same_row or above:
            nearby.append(token)

    label = _normalize(" ".join(token.text for token in sorted(nearby, key=lambda item: (item.bbox[1], item.bbox[0]))))
    positive = _phrase_match(label, config.positive_labels)
    negative = _phrase_match(label, config.negative_labels)
    return label, positive, negative


def detect_hybrid_candidates(
    tokens: Iterable[OCRToken],
    *,
    config: ContextConfig | None = None,
    ui_context_text: str = "",
    minimum_partial_digits: int = 8,
) -> HybridDetectionResult:
    """Detect full Luhn candidates plus partial digit groups and mask them all.

    Context labels classify candidates but never veto their mask decision. A
    plausible non-Luhn digit group defaults to needs-review even with a positive
    label because OCR may have missed characters. Negative labels classify a
    candidate as likely-decoy, yet the candidate remains masked.
    """
    if not 1 <= minimum_partial_digits <= 19:
        raise ValueError("minimum_partial_digits must be from 1 to 19")
    selected_config = config or ContextConfig()
    token_list = list(tokens)
    normalized_ui = _normalize(ui_context_text)
    ui_positive = _phrase_match(normalized_ui, selected_config.positive_ui_context)
    ui_negative = _phrase_match(normalized_ui, selected_config.negative_ui_context)

    candidate_sources: dict[BoundingBox, tuple[str, int]] = {
        detection.bbox: ("luhn_valid", detection.token_count)
        for detection in detect_candidates(token_list)
    }
    partial_count = 0
    for group in group_nearby_tokens(token_list):
        candidate = normalize_candidate(group.text)
        if (
            candidate.isascii()
            and candidate.isdigit()
            and minimum_partial_digits <= len(candidate) <= 19
            and not is_luhn_valid(candidate)
        ):
            partial_count += 1
            candidate_sources.setdefault(group.bbox, ("partial_digit_sequence", group.token_count))

    assessments: list[HybridCandidate] = []
    for bbox, (source, token_count) in sorted(candidate_sources.items(), key=lambda item: (item[0][1], item[0][0])):
        label, positive, negative = _context_evidence(token_list, bbox, selected_config)
        conflict = (positive and negative) or (positive and ui_negative) or (negative and ui_positive)

        if source == "partial_digit_sequence":
            classification: ContextClass = "needs_review"
            reason = "partial_numeric_sequence_may_be_cropped_or_ocr_corrupted"
        elif conflict:
            classification = "needs_review"
            reason = "conflicting_field_or_ui_context"
        elif positive:
            classification = "card_like"
            reason = "positive_field_label"
        elif negative:
            classification = "likely_decoy"
            reason = "negative_field_label_but_mask_retained"
        else:
            classification = "needs_review"
            reason = "missing_or_distant_field_label"

        if ui_positive and not positive:
            reason += ";ui_context_not_sufficient_without_field_label"
        if ui_negative and not negative and not conflict:
            reason += ";negative_ui_context_without_field_label"

        assessments.append(
            HybridCandidate(
                candidate_detected=True,
                bbox=bbox,
                source=source,  # type: ignore[arg-type]
                context_classification=classification,
                redaction_decision="mask",
                reason=reason,
                nearby_label=label,
                token_count=token_count,
            )
        )

    return HybridDetectionResult(
        candidates=tuple(assessments),
        valid_luhn_candidate_count=sum(candidate.source == "luhn_valid" for candidate in assessments),
        partial_candidate_count=partial_count,
    )


def candidates_to_detections(result: HybridDetectionResult) -> list[Detection]:
    """Adapt every protected hybrid candidate to the existing redactor interface."""
    return [
        Detection(candidate.bbox, f"hybrid_{candidate.reason}", candidate.token_count)
        for candidate in result.candidates
        if candidate.redaction_decision == "mask"
    ]