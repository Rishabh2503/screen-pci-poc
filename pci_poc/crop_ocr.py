"""Isolated OCR preprocessing experiments for diagnosed synthetic crop failures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output

from pci_poc.detect import detect_candidates, group_nearby_tokens, normalize_candidate
from pci_poc.luhn import is_luhn_valid
from pci_poc.ocr_utils import BoundingBox, OCRToken


@dataclass(frozen=True)
class OCRVariantMetric:
    """Privacy-safe OCR experiment output; contains no recognized text."""

    variant: str
    recognized_digit_characters: int
    luhn_candidate_count: int
    partial_numeric_group_count: int


def numeric_token_region(
    tokens: Sequence[OCRToken],
    image_size: tuple[int, int],
    *,
    padding: int = 18,
) -> BoundingBox | None:
    """Return a padded box around the strongest OCR digit-token group."""
    numeric_tokens = [token for token in tokens if any(character.isdigit() for character in token.text)]
    groups = group_nearby_tokens(numeric_tokens)
    if not groups:
        return None
    selected = max(
        groups,
        key=lambda group: (sum(character.isdigit() for character in group.text), group.bbox[2] - group.bbox[0]),
    )
    width, height = image_size
    return (
        max(0, selected.bbox[0] - padding),
        max(0, selected.bbox[1] - padding),
        min(width, selected.bbox[2] + padding),
        min(height, selected.bbox[3] + padding),
    )


def _recognize(
    image: Image.Image,
    *,
    tesseract_cmd: str | Path,
    psm: int,
    scale: int = 1,
    offset: tuple[int, int] = (0, 0),
) -> list[OCRToken]:
    previous_command = pytesseract.pytesseract.tesseract_cmd
    pytesseract.pytesseract.tesseract_cmd = str(tesseract_cmd)
    try:
        data = pytesseract.image_to_data(image, config=f"--psm {psm}", output_type=Output.DICT)
    finally:
        pytesseract.pytesseract.tesseract_cmd = previous_command

    tokens: list[OCRToken] = []
    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        if not text:
            continue
        x = int(data["left"][index])
        y = int(data["top"][index])
        right = x + int(data["width"][index])
        bottom = y + int(data["height"][index])
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0
        tokens.append(
            OCRToken(
                text,
                (
                    offset[0] + round(x / scale),
                    offset[1] + round(y / scale),
                    offset[0] + round(right / scale),
                    offset[1] + round(bottom / scale),
                ),
                confidence,
            )
        )
    return tokens


def _metric(name: str, tokens: Sequence[OCRToken]) -> OCRVariantMetric:
    groups = group_nearby_tokens(tokens)
    partial_count = 0
    for group in groups:
        candidate = normalize_candidate(group.text)
        if candidate.isascii() and candidate.isdigit() and 8 <= len(candidate) <= 19 and not is_luhn_valid(candidate):
            partial_count += 1
    return OCRVariantMetric(
        variant=name,
        recognized_digit_characters=sum(
            character.isdigit() for token in tokens for character in token.text
        ),
        luhn_candidate_count=len(detect_candidates(tokens)),
        partial_numeric_group_count=partial_count,
    )


def run_ocr_variants(
    image: Image.Image,
    baseline_tokens: Sequence[OCRToken],
    *,
    tesseract_cmd: str | Path,
) -> list[OCRVariantMetric]:
    """Compare full-frame, PSM, contrast, token-derived crop and overlapping crops.

    The candidate ROI is derived from OCR digit-token coordinates, not ground truth.
    Metrics intentionally report only digit counts and validated candidate counts.
    """
    source = image.convert("RGB")
    trials = [_metric("baseline_full_frame_default", baseline_tokens)]

    for psm in (3, 6, 7, 11):
        tokens = _recognize(source, tesseract_cmd=tesseract_cmd, psm=psm)
        trials.append(_metric(f"full_frame_psm{psm}", tokens))

    enlarged = source.resize((source.width * 2, source.height * 2), Image.Resampling.LANCZOS)
    scaled_tokens = _recognize(enlarged, tesseract_cmd=tesseract_cmd, psm=3, scale=2)
    trials.append(_metric("full_frame_scale2_psm3", scaled_tokens))
    contrast_enlarged = ImageOps.autocontrast(enlarged.convert("L"))
    contrast_tokens = _recognize(contrast_enlarged, tesseract_cmd=tesseract_cmd, psm=3, scale=2)
    trials.append(_metric("full_frame_scale2_autocontrast_psm3", contrast_tokens))

    region = numeric_token_region(baseline_tokens, source.size)
    if region is None:
        return trials
    left, top, right, bottom = region
    crop = source.crop(region)
    crop_tokens = _recognize(crop, tesseract_cmd=tesseract_cmd, psm=6, offset=(left, top))
    trials.append(_metric("ocr_digit_roi_psm6", crop_tokens))
    crop_enlarged = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.LANCZOS)
    for psm in (6, 7, 11):
        tokens = _recognize(
            crop_enlarged,
            tesseract_cmd=tesseract_cmd,
            psm=psm,
            scale=3,
            offset=(left, top),
        )
        trials.append(_metric(f"ocr_digit_roi_scale3_psm{psm}", tokens))
    crop_contrast = ImageOps.autocontrast(crop_enlarged.convert("L"))
    contrast_crop_tokens = _recognize(
        crop_contrast,
        tesseract_cmd=tesseract_cmd,
        psm=7,
        scale=3,
        offset=(left, top),
    )
    trials.append(_metric("ocr_digit_roi_scale3_autocontrast_psm7", contrast_crop_tokens))

    roi_width = right - left
    window_width = max(1, round(roi_width * 0.68))
    stride = max(1, round(window_width * 0.62))
    window_tokens: list[OCRToken] = []
    for window_left in range(left, right, stride):
        window_right = min(right, window_left + window_width)
        if window_right - window_left < 12:
            continue
        window = source.crop((window_left, top, window_right, bottom))
        window = window.resize((window.width * 3, window.height * 3), Image.Resampling.LANCZOS)
        window_tokens.extend(
            _recognize(
                window,
                tesseract_cmd=tesseract_cmd,
                psm=7,
                scale=3,
                offset=(window_left, top),
            )
        )
    trials.append(_metric("ocr_digit_roi_overlapping_windows_scale3_psm7", window_tokens))
    return trials