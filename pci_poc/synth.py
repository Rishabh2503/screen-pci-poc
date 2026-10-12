"""Generate deterministic CRM-like screenshots containing synthetic examples."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pci_poc.luhn import append_luhn_check_digit
from pci_poc.ocr_utils import BoundingBox


@dataclass(frozen=True)
class GroundTruthRegion:
    """A labeled synthetic region and its image-space bounding box."""

    label: str
    bbox: BoundingBox


@dataclass(frozen=True)
class SyntheticSample:
    """A generated image and its non-sensitive ground-truth regions."""

    image_path: Path
    regions: tuple[GroundTruthRegion, ...]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    for font_path in (windows_dir / "Fonts" / "arial.ttf", windows_dir / "Fonts" / "segoeui.ttf"):
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _draw_field(
    draw: ImageDraw.ImageDraw,
    label: str,
    value: str,
    y: int,
    label_font: ImageFont.ImageFont,
    value_font: ImageFont.ImageFont,
    *,
    value_x: int = 430,
) -> BoundingBox:
    draw.text((110, y), label, font=label_font, fill="#536273")
    value_position = (value_x, y - 4)
    draw.text(value_position, value, font=value_font, fill="#14283f")
    left, top, right, bottom = draw.textbbox(value_position, value, font=value_font)
    return left, top, right, bottom


def generate_synthetic_dataset(output_dir: str | Path) -> list[SyntheticSample]:
    """Create three repeatable synthetic CRM screenshots and a JSON manifest."""
    root = Path(output_dir)
    image_dir = root / "screenshots"
    image_dir.mkdir(parents=True, exist_ok=True)

    valid_number = append_luhn_check_digit("400000123456789")
    invalid_decoy = valid_number[:-1] + str((int(valid_number[-1]) + 1) % 10)
    grouped_valid = " ".join(valid_number[index : index + 4] for index in range(0, len(valid_number), 4))
    grouped_decoy = " ".join(invalid_decoy[index : index + 4] for index in range(0, len(invalid_decoy), 4))
    scenarios = (
        (
            "customer_001.png",
            "Alex Example",
            [
                ("Card number", grouped_valid, "pan"),
                ("Account reference", grouped_decoy, "decoy"),
                ("Phone", "555-010-2048", "negative"),
            ],
        ),
        (
            "customer_002.png",
            "Jordan Sample",
            [
                ("Support reference", grouped_decoy, "decoy"),
                ("Order number", "ORD-2026-004281", "negative"),
                ("Phone", "555-010-8812", "negative"),
            ],
        ),
        (
            "customer_003.png",
            "Casey Demo",
            [
                ("Account ID", "4827319062", "negative"),
                ("Expiration date", "12/29", "negative"),
                ("Contact email", "casey@example.invalid", "negative"),
            ],
        ),
    )

    label_font = _font(23)
    value_font = _font(32)
    heading_font = _font(31)
    samples: list[SyntheticSample] = []
    manifest: list[dict[str, object]] = []

    for filename, customer_name, fields in scenarios:
        image = Image.new("RGB", (1440, 620), "#f4f7fa")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1440, 82), fill="#173b57")
        draw.text((42, 24), "Northstar CRM  /  Customer profile", font=heading_font, fill="white")
        draw.rounded_rectangle((55, 118, 1385, 570), radius=8, fill="white", outline="#d7e0e7", width=2)
        draw.text((110, 145), customer_name, font=heading_font, fill="#14283f")
        draw.line((110, 195, 1320, 195), fill="#d7e0e7", width=2)

        regions: list[GroundTruthRegion] = []
        for index, (label, value, kind) in enumerate(fields):
            bbox = _draw_field(draw, label, value, 230 + index * 100, label_font, value_font)
            if kind in {"pan", "decoy"}:
                regions.append(GroundTruthRegion(kind, bbox))

        image_path = image_dir / filename
        image.save(image_path)
        sample = SyntheticSample(image_path, tuple(regions))
        samples.append(sample)
        manifest.append(
            {
                "image": str(image_path.relative_to(root)),
                "synthetic_only": True,
                "regions": [
                    {"label": region.label, "bbox": list(region.bbox)}
                    for region in regions
                ],
            }
        )

    (root / "ground_truth.json").write_text(
        json.dumps({"samples": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    return samples


HELD_OUT_TEMPLATES = (
    {
        "id": "heldout_slate",
        "background": "#eef2f5",
        "header": "#28465e",
        "panel": "#ffffff",
        "value_x": 430,
        "first_y": 230,
        "row_step": 92,
        "sidebar": False,
    },
    {
        "id": "heldout_teal_sidebar",
        "background": "#f1f5f3",
        "header": "#176b68",
        "panel": "#ffffff",
        "value_x": 495,
        "first_y": 218,
        "row_step": 86,
        "sidebar": True,
    },
    {
        "id": "heldout_charcoal",
        "background": "#e9edf1",
        "header": "#303b46",
        "panel": "#fbfcfd",
        "value_x": 455,
        "first_y": 240,
        "row_step": 88,
        "sidebar": False,
    },
)

CONTEXT_DEVELOPMENT_TEMPLATES = (
    {
        "id": "dev_compact_silver",
        "background": "#edf1f4",
        "header": "#40586b",
        "panel": "#ffffff",
        "value_x": 405,
        "first_y": 210,
        "row_step": 78,
        "sidebar": False,
    },
    {
        "id": "dev_blue_sidebar",
        "background": "#f0f3f8",
        "header": "#315a84",
        "panel": "#ffffff",
        "value_x": 500,
        "first_y": 220,
        "row_step": 82,
        "sidebar": True,
    },
)

CONTEXT_VALIDATION_TEMPLATES = (
    {
        "id": "validation_mint_wide",
        "background": "#eff5f1",
        "header": "#2b6755",
        "panel": "#ffffff",
        "value_x": 465,
        "first_y": 225,
        "row_step": 80,
        "sidebar": False,
    },
    {
        "id": "validation_copper_sidebar",
        "background": "#f6f1ed",
        "header": "#785244",
        "panel": "#fffefd",
        "value_x": 510,
        "first_y": 215,
        "row_step": 80,
        "sidebar": True,
    },
)


def _format_synthetic_number(number: str, style: int) -> str:
    if style == 0:
        return " ".join(number[index : index + 4] for index in range(0, len(number), 4))
    if style == 1:
        return "-".join(number[index : index + 4] for index in range(0, len(number), 4))
    return number


def generate_benchmark_dataset(
    output_dir: str | Path,
    *,
    count: int = 48,
    seed: int = 20261009,
    _templates: tuple[dict[str, object], ...] = HELD_OUT_TEMPLATES,
    _split: str = "held_out",
) -> list[SyntheticSample]:
    """Generate reproducible held-out screenshots with PANs, decoys and negatives.

    The benchmark uses only UI templates absent from the original demo generator.
    No tuning templates or real-world images are included in this dataset.
    Ground truth stores labels and boxes only, never candidate digit text.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    rng = random.Random(seed)
    root = Path(output_dir)
    image_dir = root / "screenshots"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_font = _font(23)
    value_font = _font(31)
    heading_font = _font(29)
    names = ("Alex Example", "Jordan Sample", "Casey Demo", "Morgan Synthetic")

    scenarios = ["card_and_decoy", "card_only", "decoys_only", "negative_only"]
    scenario_order = (scenarios * ((count + len(scenarios) - 1) // len(scenarios)))[:count]
    rng.shuffle(scenario_order)
    templates = list(_templates)
    samples: list[SyntheticSample] = []
    manifest: list[dict[str, object]] = []

    for index, scenario in enumerate(scenario_order):
        template = templates[index % len(templates)]
        image = Image.new("RGB", (1440, 640), template["background"])
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1440, 78), fill=template["header"])
        draw.text((38, 22), "Synthetic CRM  /  Account details", font=heading_font, fill="white")
        draw.rounded_rectangle((55, 108, 1385, 590), radius=8, fill=template["panel"], outline="#d5dde3", width=2)
        if template["sidebar"]:
            draw.rectangle((56, 109, 230, 589), fill="#e3eeeb")
            draw.text((78, 142), "PROFILE", font=label_font, fill="#315b57")
        draw.text((270, 138), rng.choice(names), font=heading_font, fill="#172d3f")
        draw.line((270, 185, 1330, 185), fill="#d8e0e5", width=2)

        card_number = append_luhn_check_digit(
            rng.choice("456") + "".join(str(rng.randrange(10)) for _ in range(14))
        )
        decoy_numbers = [
            append_luhn_check_digit(
                rng.choice("456") + "".join(str(rng.randrange(10)) for _ in range(14))
            )
            for _ in range(2)
        ]
        invalid_decoy = decoy_numbers[0][:-1] + str((int(decoy_numbers[0][-1]) + 1) % 10)
        fields: list[tuple[str, str, str]] = []
        if scenario in {"card_and_decoy", "card_only"}:
            fields.append(("Card number", _format_synthetic_number(card_number, rng.randrange(3)), "pan"))
        if scenario in {"card_and_decoy", "decoys_only"}:
            fields.append(("Account reference", _format_synthetic_number(decoy_numbers[0], rng.randrange(3)), "decoy"))
        if scenario == "decoys_only":
            fields.append(("Legacy reference", _format_synthetic_number(invalid_decoy, rng.randrange(3)), "decoy"))
        fields.extend(
            [
                ("Phone", f"555-010-{rng.randrange(1000, 10000)}", "negative"),
                ("Case ID", f"CASE-{rng.randrange(100000, 999999)}", "negative"),
            ]
        )
        rng.shuffle(fields)

        regions: list[GroundTruthRegion] = []
        for field_index, (label, value, kind) in enumerate(fields):
            y = int(template["first_y"]) + field_index * int(template["row_step"])
            bbox = _draw_field(
                draw,
                label,
                value,
                y,
                label_font,
                value_font,
                value_x=int(template["value_x"]),
            )
            if kind in {"pan", "decoy"}:
                regions.append(GroundTruthRegion(kind, bbox))

        filename = f"heldout_{index:04d}.png"
        image_path = image_dir / filename
        image.save(image_path)
        sample = SyntheticSample(image_path, tuple(regions))
        samples.append(sample)
        manifest.append(
            {
                "image": str(image_path.relative_to(root)),
                "synthetic_only": True,
                "dataset_split": _split,
                "template_id": template["id"],
                "regions": [
                    {"label": region.label, "bbox": list(region.bbox)}
                    for region in regions
                ],
            }
        )

    (root / "ground_truth.json").write_text(
        json.dumps({"seed": seed, "samples": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    return samples


def generate_context_tuning_datasets(
    output_dir: str | Path,
    *,
    count_per_split: int = 24,
    seed: int = 20261010,
) -> tuple[list[SyntheticSample], list[SyntheticSample]]:
    """Create disjoint development/validation sets on templates outside final test."""
    if count_per_split < 1:
        raise ValueError("count_per_split must be at least 1")
    root = Path(output_dir)
    development = generate_benchmark_dataset(
        root / "development",
        count=count_per_split,
        seed=seed,
        _templates=CONTEXT_DEVELOPMENT_TEMPLATES,
        _split="development",
    )
    validation = generate_benchmark_dataset(
        root / "validation",
        count=count_per_split,
        seed=seed + 1,
        _templates=CONTEXT_VALIDATION_TEMPLATES,
        _split="validation",
    )
    return development, validation