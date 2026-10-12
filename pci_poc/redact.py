"""Apply conservative image-space redaction to detected regions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import cv2

from pci_poc.detect import Detection


RedactionMethod = Literal["cover", "blur"]


def redact_image(
    image_path: str | Path,
    detections: Sequence[Detection],
    output_path: str | Path,
    *,
    method: RedactionMethod = "cover",
) -> Path:
    """Cover or blur detected boxes and write the result to output_path."""
    if method not in {"cover", "blur"}:
        raise ValueError("method must be 'cover' or 'blur'")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {image_path}")
    image_height, image_width = image.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        x1 = max(0, min(image_width, x1))
        x2 = max(0, min(image_width, x2))
        y1 = max(0, min(image_height, y1))
        y2 = max(0, min(image_height, y2))
        if x1 >= x2 or y1 >= y2:
            continue
        if method == "cover":
            image[y1:y2, x1:x2] = (0, 0, 0)
        else:
            region = image[y1:y2, x1:x2]
            kernel_width = max(3, min(31, (x2 - x1) // 2 * 2 + 1))
            kernel_height = max(3, min(31, (y2 - y1) // 2 * 2 + 1))
            image[y1:y2, x1:x2] = cv2.GaussianBlur(
                region,
                (kernel_width, kernel_height),
                0,
            )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"could not write redacted image: {destination}")
    return destination