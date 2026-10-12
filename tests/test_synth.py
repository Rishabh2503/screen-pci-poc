import json

from pci_poc.luhn import append_luhn_check_digit
from pci_poc.synth import (
    CONTEXT_DEVELOPMENT_TEMPLATES,
    CONTEXT_VALIDATION_TEMPLATES,
    HELD_OUT_TEMPLATES,
    generate_context_tuning_datasets,
    generate_synthetic_dataset,
)


def test_dataset_is_deterministic_and_manifest_contains_only_box_annotations(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_samples = generate_synthetic_dataset(first_dir)
    second_samples = generate_synthetic_dataset(second_dir)
    first_manifest_text = (first_dir / "ground_truth.json").read_text(encoding="utf-8")
    first_manifest = json.loads(first_manifest_text)
    labels_by_image = [
        [region["label"] for region in sample["regions"]]
        for sample in first_manifest["samples"]
    ]

    assert len(first_samples) == 3
    assert labels_by_image[0] == ["pan", "decoy"]
    assert labels_by_image[1] == ["decoy"]
    assert labels_by_image[2] == []
    assert all(sample.image_path.is_file() for sample in first_samples)
    assert first_manifest_text == (second_dir / "ground_truth.json").read_text(encoding="utf-8")
    assert [path.read_bytes() for path in sorted((first_dir / "screenshots").glob("*.png"))] == [
        path.read_bytes() for path in sorted((second_dir / "screenshots").glob("*.png"))
    ]
    assert append_luhn_check_digit("400000123456789") not in first_manifest_text


def test_context_tuning_splits_use_separate_templates(tmp_path):
    development, validation = generate_context_tuning_datasets(
        tmp_path / "tuning",
        count_per_split=8,
        seed=71,
    )
    split_templates = {}
    for split in ("development", "validation"):
        manifest = json.loads(
            (tmp_path / "tuning" / split / "ground_truth.json").read_text(encoding="utf-8")
        )
        assert all(item["dataset_split"] == split for item in manifest["samples"])
        split_templates[split] = {item["template_id"] for item in manifest["samples"]}

    final_templates = {template["id"] for template in HELD_OUT_TEMPLATES}
    assert len(development) == len(validation) == 8
    assert split_templates["development"] == {template["id"] for template in CONTEXT_DEVELOPMENT_TEMPLATES}
    assert split_templates["validation"] == {template["id"] for template in CONTEXT_VALIDATION_TEMPLATES}
    assert split_templates["development"].isdisjoint(split_templates["validation"])
    assert split_templates["development"].isdisjoint(final_templates)
    assert split_templates["validation"].isdisjoint(final_templates)