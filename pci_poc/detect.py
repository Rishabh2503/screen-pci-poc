"""Group OCR tokens and detect Luhn-valid digit candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from pci_poc.luhn import is_luhn_valid
from pci_poc.ocr_utils import BoundingBox, OCRToken


@dataclass(frozen=True)
class TokenGroup:
    """Nearby OCR tokens combined with their union bounding box."""

    text: str
    bbox: BoundingBox
    token_count: int


@dataclass(frozen=True)
class Detection:
    """A validated candidate region; intentionally excludes its digit text."""

    bbox: BoundingBox
    reason: str
    token_count: int


def normalize_candidate(text: str) -> str:
    """Remove common OCR separators while retaining all other characters."""
    return re.sub(r"[\s-]", "", text)


def _union_box(tokens: list[OCRToken]) -> BoundingBox:
    return (
        min(token.bbox[0] for token in tokens),
        min(token.bbox[1] for token in tokens),
        max(token.bbox[2] for token in tokens),
        max(token.bbox[3] for token in tokens),
    )


def group_nearby_tokens(
    tokens: Iterable[OCRToken],
    *,
    max_horizontal_gap: int = 28,
    row_tolerance: int = 14,
) -> list[TokenGroup]:
    """Group tokens on the same row when their horizontal gap is small."""
    rows: list[list[OCRToken]] = []
    for token in sorted(tokens, key=lambda item: ((item.bbox[1] + item.bbox[3]) / 2, item.bbox[0])):
        center_y = (token.bbox[1] + token.bbox[3]) / 2
        matching_row = next(
            (
                row
                for row in rows
                if abs(
                    center_y
                    - sum((item.bbox[1] + item.bbox[3]) / 2 for item in row) / len(row)
                )
                <= row_tolerance
            ),
            None,
        )
        if matching_row is None:
            rows.append([token])
        else:
            matching_row.append(token)

    groups: list[TokenGroup] = []
    for row in rows:
        ordered = sorted(row, key=lambda item: item.bbox[0])
        chunk: list[OCRToken] = []
        for token in ordered:
            if chunk and token.bbox[0] - chunk[-1].bbox[2] > max_horizontal_gap:
                groups.append(TokenGroup("".join(item.text for item in chunk), _union_box(chunk), len(chunk)))
                chunk = []
            chunk.append(token)
        if chunk:
            groups.append(TokenGroup("".join(item.text for item in chunk), _union_box(chunk), len(chunk)))
    return groups


def detect_candidates(tokens: Iterable[OCRToken]) -> list[Detection]:
    """Return bounding boxes for grouped candidates passing length and Luhn checks."""
    detections: list[Detection] = []
    for group in group_nearby_tokens(tokens):
        candidate = normalize_candidate(group.text)
        if candidate.isascii() and candidate.isdigit() and is_luhn_valid(candidate):
            detections.append(Detection(group.bbox, "length_and_luhn_valid", group.token_count))
    return detections


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    """Calculate intersection-over-union for two x1,y1,x2,y2 boxes."""
    intersection_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0