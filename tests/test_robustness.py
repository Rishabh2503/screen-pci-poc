import json

from pci_poc.robustness import ROBUSTNESS_CASES, ROBUSTNESS_TEMPLATES, generate_robustness_dataset
from pci_poc.synth import HELD_OUT_TEMPLATES


def test_robustness_dataset_is_deterministic_and_uses_new_templates(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = generate_robustness_dataset(first_root, seed=113, repetitions=1)
    second = generate_robustness_dataset(second_root, seed=113, repetitions=1)
    manifest_text = (first_root / "ground_truth.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    conditions = {entry["condition"] for entry in manifest["samples"]}
    robustness_templates = {template["id"] for template in ROBUSTNESS_TEMPLATES}
    old_templates = {template["id"] for template in HELD_OUT_TEMPLATES}

    assert len(first) == len(ROBUSTNESS_CASES)
    assert len(second) == len(first)
    assert conditions == {case[0] for case in ROBUSTNESS_CASES}
    assert robustness_templates.isdisjoint(old_templates)
    assert {entry["template_id"] for entry in manifest["samples"]} == robustness_templates
    assert all(entry["dataset_split"] == "robustness_final" for entry in manifest["samples"])
    assert all(sample.image_path.is_file() for sample in first)
    assert manifest_text == (second_root / "ground_truth.json").read_text(encoding="utf-8")
    assert [path.read_bytes() for path in sorted((first_root / "screenshots").glob("*.png"))] == [
        path.read_bytes() for path in sorted((second_root / "screenshots").glob("*.png"))
    ]
    assert '"value"' not in manifest_text


def test_robustness_repetitions_increase_case_count(tmp_path):
    samples = generate_robustness_dataset(tmp_path / "dataset", seed=7, repetitions=2)

    assert len(samples) == len(ROBUSTNESS_CASES) * 2
    assert len({sample.condition for sample in samples}) == len(ROBUSTNESS_CASES)