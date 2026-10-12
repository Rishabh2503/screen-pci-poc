"""Create a synthetic-only showcase from saved experiment reports."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Mapping


DEFAULT_REPORTS = {
    "baseline": Path("generated/benchmark/metrics.json"),
    "context": Path("generated/context_experiment_final/final_comparison/context_comparison.json"),
    "robustness": Path("generated/robustness_experiment/metrics.json"),
    "hybrid_legacy": Path("generated/hybrid_policy_experiment/metrics.json"),
    "hybrid_audit": Path("generated/hybrid_policy_audit_review/metrics.json"),
}
SHOWCASE_TITLE = "Screen PCI Redaction POC showcase"
DESCRIPTIONS = {
    "baseline": "Original OCR, numeric grouping, Luhn and redaction baseline on held-out synthetic screenshots.",
    "context": "Paired baseline/context-only comparison on the same frozen 48 screenshots.",
    "robustness": "Adversarial synthetic robustness set with missing, distant, conflicting and misleading labels.",
    "hybrid_legacy": "Earlier three-split hybrid run; this report does not contain complete-number recognition metrics.",
    "hybrid_audit": "Later three-split hybrid audit with separate complete-number, coverage and review-mask metrics.",
}
FIELDS = (
    "tp", "fp", "fn", "precision", "recall", "f1",
    "complete_number_tp", "complete_number_fp", "complete_number_fn",
    "complete_number_precision", "complete_number_recall", "complete_number_f1",
    "false_redactions", "decoy_false_redactions", "negative_text_redactions",
    "needs_review_count", "needs_review_masked_count", "needs_review_unmasked_count",
    "mean_card_text_coverage", "median_latency_ms", "p95_latency_ms", "ocr_failures",
)


def _read_report(path: Path) -> tuple[dict[str, Any] | None, dict[str, str]]:
    if not path.is_file():
        return None, {"status": "missing", "path": path.as_posix()}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, {"status": "invalid", "path": path.as_posix(), "detail": type(error).__name__}
    if not isinstance(value, dict):
        return None, {"status": "invalid", "path": path.as_posix(), "detail": "JSON root is not an object"}
    return value, {"status": "loaded", "path": path.as_posix()}


def _metrics(source: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {
        "tp": ("tp", "true_positives"),
        "fp": ("fp", "false_positives"),
        "fn": ("fn", "false_negatives"),
        "false_redactions": ("false_redactions", "false_redaction_count"),
    }
    normalized: dict[str, Any] = {}
    for field in FIELDS:
        value = next((source[key] for key in aliases.get(field, (field,)) if key in source), None)
        normalized[field] = round(value, 6) if isinstance(value, float) else value
    return normalized


def _row(rows: list[dict[str, Any]], experiment: str, split: str, count: Any,
         templates: list[str], mode: str, values: Mapping[str, Any]) -> None:
    rows.append({
        "experiment": experiment,
        "split": split,
        "sample_count": count,
        "templates": "; ".join(templates),
        "mode": mode,
        **_metrics(values),
    })


def _collect(reports: Mapping[str, dict[str, Any] | None]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    experiments: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    report = reports.get("baseline")
    if report:
        summary = report.get("aggregate", {})
        templates = list(summary.get("held_out_templates", []))
        count = summary.get("evaluated_screenshots")
        experiments.append({"id": "baseline_48", "description": DESCRIPTIONS["baseline"], "sample_count": count, "templates": templates})
        _row(rows, "baseline_48", "held_out", count, templates, "luhn_baseline", summary)

    report = reports.get("context")
    if report:
        count = report.get("baseline", {}).get("evaluated_screenshots")
        templates = list(report.get("final_templates", []))
        experiments.append({"id": "context_comparison_48", "description": DESCRIPTIONS["context"], "sample_count": count, "templates": templates})
        for mode in ("baseline", "context_aware"):
            if mode in report:
                _row(rows, "context_comparison_48", "same_frozen_test", count, templates, mode, report[mode])

    report = reports.get("robustness")
    if report:
        count = report.get("baseline", {}).get("evaluated_screenshots")
        templates = list(report.get("templates", []))
        experiments.append({"id": "robustness_36", "description": DESCRIPTIONS["robustness"], "sample_count": count, "templates": templates})
        for mode in ("baseline", "context_aware"):
            if mode in report:
                _row(rows, "robustness_36", "adversarial_synthetic", count, templates, mode, report[mode])

    for key in ("hybrid_legacy", "hybrid_audit"):
        report = reports.get(key)
        if not report:
            continue
        splits = report.get("splits", {})
        split_descriptions = []
        for split_name, split in splits.items():
            templates = list(split.get("templates", []))
            count = split.get("screenshots")
            split_descriptions.append({"split": split_name, "sample_count": count, "templates": templates})
            for mode, values in split.get("metrics", {}).items():
                _row(rows, key, split_name, count, templates, mode, values)
        experiments.append({"id": key, "description": DESCRIPTIONS[key], "splits": split_descriptions})
    return experiments, rows


def _safe_visual(report_path: Path, relative: str | None) -> Path | None:
    if not relative:
        return None
    report_root = report_path.parent.resolve()
    candidate = (report_root / Path(relative)).resolve()
    try:
        candidate.relative_to(report_root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _copy_failures(reports: Mapping[str, dict[str, Any] | None], paths: Mapping[str, Path], output: Path):
    robustness = reports.get("robustness") or {}
    hybrid = reports.get("hybrid_audit") or {}
    hybrid_visuals = hybrid.get("splits", {}).get("final", {}).get("visualizations", {})
    cases = (
        ("account_reference_fp", "Luhn-valid Account Reference false positive", "The baseline accepted a Luhn-valid candidate in an account-reference field.", "robustness", (robustness.get("visualizations") or {}).get("baseline_false_positive")),
        ("missing_label_fn", "Missing label causes context-only recall loss", "A context-only false negative from the synthetic missing-label case; no nearby label evidence was available.", "robustness", (robustness.get("visualizations") or {}).get("context_aware_false_negative")),
        ("cropped_number_fn", "Cropped number: bbox localization is not full recognition", "A cropped-text false negative; the comparison masks the annotated number region.", "hybrid_audit", hybrid_visuals.get("baseline_false_negative")),
    )
    failures = []
    warnings = []
    for case_id, title, description, key, reference in cases:
        source = _safe_visual(paths[key], reference)
        record = {"id": case_id, "title": title, "description": description}
        if source is None:
            record.update({"status": "missing_visual", "image": None})
            warnings.append(f"Sanitized visual missing: {case_id}")
        else:
            destination = output / "failures" / f"{case_id}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            record.update({"status": "copied_sanitized_artifact", "image": destination.relative_to(output).as_posix()})
        failures.append(record)
    return failures, warnings


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "not reported"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _render_markdown(summary: dict[str, Any]) -> str:
    out = [
        "# Screen PCI Redaction POC: Showcase",
        "",
        "> Synthetic-data evidence only. Not production-ready, PCI-compliant, or validated on real customer screens.",
        "",
        "## Architecture",
        "",
        "Synthetic screenshot → OCR → numeric candidate grouping → Luhn validation → contextual classification → conservative redaction → evaluation.",
        "",
        "The hybrid mode separates candidate protection from semantic classification: Luhn-valid and plausible partial numeric candidates are masked, while context labels classify them as card-like, likely-decoy, or needs-review.",
        "",
        "## Metric Definitions",
        "",
        "- **BBox localization P/R/F1:** one-to-one IoU matching to annotated card-text boxes. A partial fragment can count as localized when its box overlaps; this is not full-number recognition.",
        "- **Complete-number recognition:** exact full-number digest match, shown only in reports that recorded it.",
        "- **Sensitive-region coverage:** changed-pixel fraction inside annotated card-text boxes.",
        "- **False redactions:** report-specific changed pixels within decoy/negative boxes.",
        "- **Needs review:** uncertain candidates; masked counts are shown when measured.",
        "- **Latency:** report-specific per-image median/P95; timing scopes differ and are not pooled.",
        "",
        "## Experiments",
        "",
        "Each experiment is reported separately. Numbers from different datasets are not combined into a leaderboard.",
        "",
    ]
    for experiment in summary["experiments"]:
        out.extend([f"### {experiment['id']}", "", experiment["description"], ""])
        if "splits" in experiment:
            for split in experiment["splits"]:
                out.append(f"- `{split['split']}`: {split['sample_count']} screenshots; templates: {', '.join(split['templates'])}")
        else:
            out.append(f"Sample size: {experiment['sample_count']}; templates: {', '.join(experiment['templates'])}")
        out.append("")
        rows = [row for row in summary["metrics"] if row["experiment"] == experiment["id"]]
        if rows:
            out.extend([
                "| Split | Mode | N | TP/FP/FN | P/R/F1 | Complete TP/FP/FN | False redactions | Review (masked) | Coverage | Median/P95 ms |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for row in rows:
                complete = [row.get(f"complete_number_{part}") for part in ("tp", "fp", "fn")]
                complete_text = "not reported" if all(value is None for value in complete) else "/".join(_format(value, 0) for value in complete)
                review = _format(row.get("needs_review_count"), 0)
                if row.get("needs_review_masked_count") is not None:
                    review += f" ({row['needs_review_masked_count']} masked)"
                out.append(
                    f"| {row['split']} | {row['mode']} | {_format(row['sample_count'],0)} | "
                    f"{'/'.join(_format(row.get(key),0) for key in ('tp','fp','fn'))} | "
                    f"{'/'.join(_format(row.get(key)) for key in ('precision','recall','f1'))} | "
                    f"{complete_text} | {_format(row.get('false_redactions'),0)} | {review} | "
                    f"{_format(row.get('mean_card_text_coverage'))} | "
                    f"{_format(row.get('median_latency_ms'),1)}/{_format(row.get('p95_latency_ms'),1)} |"
                )
            out.append("")

    out.extend(["## Failure Cases", "", "The following images are copied from sanitized experiment comparisons; numeric candidate text is not included.", ""])
    for failure in summary["failure_examples"]:
        out.extend([f"### {failure['title']}", "", failure["description"], ""])
        if failure.get("image"):
            out.append(f"![{failure['title']}]({failure['image']})")
        else:
            out.append("Sanitized comparison unavailable in the source reports.")
        out.append("")
    out.extend([
        "## Generalization and Trade-offs",
        "",
        "Context scored perfectly on the earlier clean 48-image synthetic set, but on a separate adversarial 36-image set its recall fell to 0.20 (16 false negatives). Missing, distant, vertical, cropped, conflicting, and misleading labels explain why that earlier score did not generalize.",
        "",
        "On the hybrid audit final split, hybrid coverage was 94.9% versus 88.0% for baseline, with five false redactions (four decoy boxes and one ordinary-negative box). Exact complete-number recognition was 8/4/1 even though bbox localization was 9/5/0: one partial fragment localized but was not a fully recognized number. Six candidates needed review and were masked. This is a synthetic protection/false-positive trade-off, not production evidence.",
        "",
        "Limitations: synthetic layouts and labels do not represent real UI diversity; OCR and redaction can miss or misclassify sensitive data. This is not a PCI control or compliance validation.",
        "",
    ])
    if summary["warnings"]:
        out.extend(["## Missing Artifacts", "", *[f"- {warning}" for warning in summary["warnings"]], ""])
    return "\n".join(out)


def generate_showcase(
    output_dir: str | Path,
    *,
    report_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Write Markdown, summary JSON/CSV, and copies of sanitized failure images."""
    root = Path(output_dir)
    previous_generated_files: set[str] = set()
    if root.exists() and any(root.iterdir()):
        prior_summary_path = root / "summary.json"
        try:
            prior_summary = json.loads(prior_summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior_summary = None
        expected_files = set(prior_summary.get("generated_files", [])) if isinstance(prior_summary, dict) else set()
        actual_files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        if (
            not isinstance(prior_summary, dict)
            or prior_summary.get("title") != SHOWCASE_TITLE
            or not expected_files
            or actual_files != expected_files
        ):
            raise FileExistsError(f"output directory is not an owned showcase package: {root}")
        previous_generated_files = expected_files
    root.mkdir(parents=True, exist_ok=True)
    paths = {key: Path(value) for key, value in DEFAULT_REPORTS.items()}
    if report_paths:
        paths.update({key: Path(value) for key, value in report_paths.items()})

    reports: dict[str, dict[str, Any] | None] = {}
    statuses: dict[str, dict[str, str]] = {}
    for key, path in paths.items():
        report, status = _read_report(path)
        reports[key] = report
        statuses[key] = status
    experiments, metric_rows = _collect(reports)
    failures, visual_warnings = _copy_failures(reports, paths, root)
    intended_files = {"report.md", "summary.csv", "summary.json"}
    intended_files.update(item["image"] for item in failures if item.get("image"))
    for stale_name in previous_generated_files - intended_files:
        stale_path = (root / Path(stale_name)).resolve()
        try:
            stale_path.relative_to(root.resolve())
        except ValueError:
            continue
        if stale_path.is_file():
            stale_path.unlink()
    warnings = [
        f"Report {key} is {status['status']}: {status['path']}"
        for key, status in statuses.items()
        if status["status"] != "loaded"
    ] + visual_warnings
    summary: dict[str, Any] = {
        "title": SHOWCASE_TITLE,
        "scope": "synthetic-data evidence only; not production accuracy or PCI compliance",
        "architecture": ["synthetic screenshot", "OCR", "numeric candidate grouping", "Luhn validation", "contextual classification", "conservative redaction", "evaluation"],
        "report_status": statuses,
        "experiments": experiments,
        "metrics": metric_rows,
        "failure_examples": failures,
        "warnings": warnings,
    }
    (root / "report.md").write_text(_render_markdown(summary), encoding="utf-8")
    csv_path = root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["experiment", "split", "sample_count", "templates", "mode", *FIELDS]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metric_rows)
    outputs_before_summary = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]
    summary["generated_files"] = sorted(set(outputs_before_summary) | {"summary.json"})
    summary["output_paths_verified"] = True
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary["output_paths_verified"] = all((root / name).is_file() for name in summary["generated_files"])
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Showcase created: {root} ({len(summary['generated_files'])} files; verified={summary['output_paths_verified']}).")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("generated") / "final_demo")
    args = parser.parse_args()
    generate_showcase(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())