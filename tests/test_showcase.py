import json
import re

import pytest
from PIL import Image, ImageDraw

from pci_poc.showcase import DEFAULT_REPORTS, generate_showcase


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sanitized_visual(path):
    image = Image.new("RGB", (120, 60), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 100, 40), fill="black", outline="green", width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def test_showcase_normalizes_report_schemas_and_copies_sanitized_failures(tmp_path):
    report_root = tmp_path / "reports"
    account_visual = _sanitized_visual(report_root / "comparisons" / "account.png")
    missing_visual = _sanitized_visual(report_root / "comparisons" / "missing.png")
    crop_visual = _sanitized_visual(report_root / "visualizations" / "final" / "crop.png")
    candidate_digits = "4111111111111111"

    baseline_path = _write_json(
        report_root / "baseline.json",
        {
            "benchmark_scope": "synthetic",
            "aggregate": {
                "evaluated_screenshots": 3,
                "held_out_templates": ["template-a"],
                "true_positives": 1,
                "false_positives": 1,
                "false_negatives": 0,
                "precision": 0.5,
                "recall": 1.0,
                "f1": 2 / 3,
                "candidate_text": candidate_digits,
            },
        },
    )
    robustness_path = _write_json(
        report_root / "robustness.json",
        {
            "baseline": {"evaluated_screenshots": 2, "true_positives": 1, "false_positives": 1, "false_negatives": 0},
            "context_aware": {"evaluated_screenshots": 2, "true_positives": 0, "false_positives": 0, "false_negatives": 1},
            "templates": ["robust-a"],
            "visualizations": {
                "baseline_false_positive": "comparisons/account.png",
                "context_aware_false_negative": "comparisons/missing.png",
            },
        },
    )
    hybrid_path = _write_json(
        report_root / "hybrid.json",
        {
            "splits": {
                "final": {
                    "screenshots": 1,
                    "templates": ["hybrid-final"],
                    "metrics": {"hybrid": {"tp": 1, "fp": 0, "fn": 0}},
                    "visualizations": {"baseline_false_negative": "visualizations/final/crop.png"},
                }
            }
        },
    )
    paths = {key: tmp_path / "absent" / f"{key}.json" for key in DEFAULT_REPORTS}
    paths.update({"baseline": baseline_path, "robustness": robustness_path, "hybrid_audit": hybrid_path})
    output = tmp_path / "showcase"

    result = generate_showcase(output, report_paths=paths)

    assert result["report_status"]["context"]["status"] == "missing"
    assert result["output_paths_verified"] is True
    assert (output / "report.md").is_file()
    assert (output / "summary.csv").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "failures/account_reference_fp.png").read_bytes() == account_visual.read_bytes()
    assert (output / "failures/missing_label_fn.png").read_bytes() == missing_visual.read_bytes()
    assert (output / "failures/cropped_number_fn.png").read_bytes() == crop_visual.read_bytes()
    assert len({row["experiment"] for row in result["metrics"]}) == 3

    for path in (output / "report.md", output / "summary.json", output / "summary.csv"):
        text = path.read_text(encoding="utf-8")
        assert candidate_digits not in text
        assert re.search(r"(?<!\d)\d{13,19}(?!\d)", text) is None


def test_showcase_reports_missing_reports_and_missing_visuals(tmp_path):
    report_paths = {key: tmp_path / "not-found" / f"{key}.json" for key in DEFAULT_REPORTS}
    result = generate_showcase(tmp_path / "showcase", report_paths=report_paths)

    assert all(status["status"] == "missing" for status in result["report_status"].values())
    assert result["metrics"] == []
    assert len(result["warnings"]) >= len(DEFAULT_REPORTS)
    assert all(item["status"] == "missing_visual" for item in result["failure_examples"])


def test_showcase_does_not_overwrite_nonempty_output_directory(tmp_path):
    output = tmp_path / "already-used"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing", encoding="utf-8")
    report_paths = {key: tmp_path / "absent" / f"{key}.json" for key in DEFAULT_REPORTS}

    with pytest.raises(FileExistsError):
        generate_showcase(output, report_paths=report_paths)

    assert marker.read_text(encoding="utf-8") == "existing"


def test_showcase_can_refresh_its_own_manifested_output_directory(tmp_path):
    output = tmp_path / "showcase"
    report_paths = {key: tmp_path / "absent" / f"{key}.json" for key in DEFAULT_REPORTS}

    first = generate_showcase(output, report_paths=report_paths)
    second = generate_showcase(output, report_paths=report_paths)

    assert first["generated_files"] == second["generated_files"]
    assert (output / "summary.json").is_file()