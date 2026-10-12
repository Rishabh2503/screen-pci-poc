import json

from pci_poc.hybrid_experiment import CONDITIONS, TEMPLATE_SETS, generate_hybrid_datasets


def test_fresh_hybrid_splits_are_disjoint_and_cover_all_conditions(tmp_path):
    datasets = generate_hybrid_datasets(tmp_path / "hybrid", seed=81)
    templates_by_split = {}

    for split, samples in datasets.items():
        manifest = json.loads(
            (tmp_path / "hybrid" / split / "ground_truth.json").read_text(encoding="utf-8")
        )
        assert len(samples) == len(CONDITIONS)
        assert {item["condition"] for item in manifest["samples"]} == set(CONDITIONS)
        assert all(item["split"] == split for item in manifest["samples"])
        for item in manifest["samples"]:
            for region in item["regions"]:
                if region["label"] == "pan":
                    assert len(region["number_sha256"]) == 64
                    assert item["card_number_sha256"] == region["number_sha256"]
                else:
                    assert "number_sha256" not in region
        templates_by_split[split] = {item["template_id"] for item in manifest["samples"]}
        assert templates_by_split[split] == {template["id"] for template in TEMPLATE_SETS[split]}
        assert '"value"' not in json.dumps(manifest)

    assert templates_by_split["development"].isdisjoint(templates_by_split["validation"])
    assert templates_by_split["development"].isdisjoint(templates_by_split["final"])
    assert templates_by_split["validation"].isdisjoint(templates_by_split["final"])