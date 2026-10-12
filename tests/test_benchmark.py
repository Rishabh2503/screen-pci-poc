import csv
import json

from pci_poc.benchmark import write_benchmark_reports


def test_report_writer_creates_machine_readable_summary_without_candidate_text(tmp_path):
    aggregate = {
        "evaluated_screenshots": 2,
        "true_positives": 1,
        "false_positives": 0,
        "false_negatives": 1,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 2 / 3,
        "synthetic_only": True,
    }
    per_image = [
        {
            "image": "heldout_0000.png",
            "template_id": "heldout_slate",
            "true_positives": 1,
            "false_positives": 0,
            "false_negatives": 0,
            "latency_ms": 42.0,
        }
    ]

    json_path, aggregate_csv, per_image_csv = write_benchmark_reports(
        tmp_path,
        aggregate,
        per_image,
    )

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["synthetic_only"] is True
    assert report["ocr_mode"] == "real_tesseract"
    assert report["aggregate"] == aggregate
    assert report["per_image"] == per_image
    with aggregate_csv.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    with per_image_csv.open(newline="", encoding="utf-8") as handle:
        image_rows = list(csv.DictReader(handle))
    assert summary_rows[0]["evaluated_screenshots"] == "2"
    assert image_rows[0]["image"] == "heldout_0000.png"
    assert "4000001234567899" not in json_path.read_text(encoding="utf-8")