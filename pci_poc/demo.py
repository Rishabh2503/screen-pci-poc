"""Generate synthetic examples and optionally run local OCR and redaction."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from pci_poc.detect import detect_candidates
from pci_poc.ocr_utils import TesseractOCR, TesseractUnavailableError
from pci_poc.redact import redact_image
from pci_poc.synth import SyntheticSample, generate_synthetic_dataset


def _save_ground_truth_visualization(sample: SyntheticSample, destination: Path) -> None:
    with Image.open(sample.image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for region in sample.regions:
        color = "#11845b" if region.label == "pan" else "#cf6a32"
        draw.rectangle(region.bbox, outline=color, width=4)
        draw.text((region.bbox[0], max(0, region.bbox[1] - 24)), f"GT: {region.label}", fill=color)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def run_demo(output_dir: str | Path) -> int:
    """Generate synthetic data; return 2 if real OCR is unavailable."""
    root = Path(output_dir)
    samples = generate_synthetic_dataset(root)
    visualization_dir = root / "ground_truth_visualizations"
    for sample in samples:
        _save_ground_truth_visualization(
            sample,
            visualization_dir / sample.image_path.name,
        )

    print(f"Generated {len(samples)} synthetic screenshots and ground-truth annotations in {root}.")
    try:
        ocr = TesseractOCR()
    except TesseractUnavailableError as error:
        print(f"OCR and redacted outputs skipped: {error}")
        print("No OCR end-to-end result is claimed.")
        return 2

    redacted_dir = root / "redacted"
    redacted_count = 0
    for sample in samples:
        with Image.open(sample.image_path) as image:
            detections = detect_candidates(ocr.recognize(image))
        redact_image(
            sample.image_path,
            detections,
            redacted_dir / sample.image_path.name,
        )
        redacted_count += len(detections)

    print(f"Saved redacted screenshots; detected {redacted_count} candidate regions.")
    print("Only synthetic screenshots were processed. Candidate digits are not logged.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated") / "demo",
        help="directory for synthetic input, annotations, visualizations, and redacted output",
    )
    arguments = parser.parse_args()
    return run_demo(arguments.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())