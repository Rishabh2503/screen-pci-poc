"""Tesseract OCR adapter and token types."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import pytesseract
from PIL import Image
from pytesseract import Output


BoundingBox: TypeAlias = tuple[int, int, int, int]
ImageSource: TypeAlias = Image.Image | str | Path

COMMON_TESSERACT_PATHS = (
    Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    / "Tesseract-OCR"
    / "tesseract.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    / "Tesseract-OCR"
    / "tesseract.exe",
)


class TesseractUnavailableError(RuntimeError):
    """Raised when no usable Tesseract executable can be located."""


@dataclass(frozen=True)
class OCRToken:
    """One non-empty OCR word and its image-space bounding box."""

    text: str
    bbox: BoundingBox
    confidence: float = -1.0


def find_tesseract_executable(
    configured_path: str | Path | None = None,
) -> Path | None:
    """Resolve an explicit path, TESSERACT_CMD, PATH, then common install paths."""
    requested = configured_path if configured_path is not None else os.environ.get("TESSERACT_CMD")
    if requested:
        requested_text = str(requested).strip('"')
        command_path = shutil.which(requested_text)
        candidate = Path(command_path or requested_text)
        return candidate if candidate.is_file() else None

    path_executable = shutil.which("tesseract")
    if path_executable:
        return Path(path_executable)

    return next((path for path in COMMON_TESSERACT_PATHS if path.is_file()), None)


class TesseractOCR:
    """Run pytesseract using a discovered or explicitly configured executable."""

    def __init__(self, tesseract_cmd: str | Path | None = None) -> None:
        self.executable = find_tesseract_executable(tesseract_cmd)
        if self.executable is None:
            if tesseract_cmd or os.environ.get("TESSERACT_CMD"):
                detail = "TESSERACT_CMD does not point to an existing executable file"
            else:
                detail = "Tesseract was not found on PATH or in common install locations"
            raise TesseractUnavailableError(
                f"{detail}. Install Tesseract or set TESSERACT_CMD to its executable path."
            )

    def recognize(self, image: ImageSource) -> list[OCRToken]:
        """Return non-empty OCR word tokens without logging recognized text."""
        previous_command = pytesseract.pytesseract.tesseract_cmd
        pytesseract.pytesseract.tesseract_cmd = str(self.executable)
        try:
            data = pytesseract.image_to_data(image, output_type=Output.DICT)
        finally:
            pytesseract.pytesseract.tesseract_cmd = previous_command

        tokens: list[OCRToken] = []
        for index, raw_text in enumerate(data["text"]):
            text = raw_text.strip()
            if not text:
                continue
            left = int(data["left"][index])
            top = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])
            try:
                confidence = float(data["conf"][index])
            except (TypeError, ValueError):
                confidence = -1.0
            tokens.append(
                OCRToken(text, (left, top, left + width, top + height), confidence)
            )
        return tokens