from pci_poc.crop_ocr import numeric_token_region
from pci_poc.ocr_utils import OCRToken


def test_numeric_crop_uses_ocr_boxes_and_clips_to_image_bounds():
    tokens = [
        OCRToken("Card", (10, 10, 45, 28)),
        OCRToken("1234", (4, 6, 40, 24)),
        OCRToken("5678", (44, 6, 80, 24)),
    ]

    assert numeric_token_region(tokens, (100, 60), padding=10) == (0, 0, 90, 34)


def test_numeric_crop_returns_none_without_digit_tokens():
    assert numeric_token_region([OCRToken("Reference", (3, 3, 60, 20))], (100, 60)) is None