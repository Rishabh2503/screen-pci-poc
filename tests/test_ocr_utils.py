from pathlib import Path

import pytest
import pytesseract
from PIL import Image

from pci_poc.ocr_utils import (
    TesseractOCR,
    TesseractUnavailableError,
    find_tesseract_executable,
)


def test_explicit_tesseract_environment_path_takes_precedence(tmp_path, monkeypatch):
    executable = tmp_path / "tesseract.exe"
    executable.write_bytes(b"synthetic test executable")
    monkeypatch.setenv("TESSERACT_CMD", str(executable))
    monkeypatch.setattr("pci_poc.ocr_utils.shutil.which", lambda _command: None)

    assert find_tesseract_executable() == executable


def test_missing_tesseract_has_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setenv("TESSERACT_CMD", str(tmp_path / "missing.exe"))
    with pytest.raises(TesseractUnavailableError, match="set TESSERACT_CMD"):
        TesseractOCR()


def test_recognize_maps_mock_ocr_words_to_bounding_boxes(tmp_path, monkeypatch):
    executable = tmp_path / "tesseract.exe"
    executable.write_bytes(b"synthetic test executable")
    adapter = TesseractOCR(executable)
    original_command = "previous-tesseract-command"
    import pci_poc.ocr_utils as ocr_utils

    monkeypatch.setattr(ocr_utils.pytesseract.pytesseract, "tesseract_cmd", original_command)
    monkeypatch.setattr(
        ocr_utils.pytesseract,
        "image_to_data",
        lambda _image, output_type: {
            "text": ["4000", "", "0012"],
            "left": [10, 40, 44],
            "top": [20, 20, 20],
            "width": [28, 0, 28],
            "height": [18, 0, 18],
            "conf": ["91.5", "-1", "87"],
        },
    )

    tokens = adapter.recognize(Image.new("RGB", (100, 60), "white"))

    assert [(token.text, token.bbox, token.confidence) for token in tokens] == [
        ("4000", (10, 20, 38, 38), 91.5),
        ("0012", (44, 20, 72, 38), 87.0),
    ]
    assert ocr_utils.pytesseract.pytesseract.tesseract_cmd == original_command


def test_tesseract_integration_on_synthetic_image():
    executable = find_tesseract_executable()
    if executable is None:
        pytest.skip("Tesseract executable unavailable; real OCR integration was not run")

    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (900, 150), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 58)
    except OSError:
        font = ImageFont.load_default()
    expected = "SYNTHETIC OCR CHECK"
    draw.text((24, 30), expected, fill="black", font=font)
    try:
        tokens = TesseractOCR(executable).recognize(image)
    except (OSError, pytesseract.TesseractError, pytesseract.TesseractNotFoundError) as error:
        pytest.skip(f"Tesseract could not run the real OCR integration: {error}")
    recognized = " ".join(token.text for token in tokens).upper()
    assert recognized == expected