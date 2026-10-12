"""Export docs/summary.html to a one-page PDF with Microsoft Edge."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "docs" / "summary.html"
PDF = ROOT / "docs" / "summary.pdf"


def _find_edge() -> Path:
    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", ""))
        / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("ProgramFiles", ""))
        / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Microsoft/Edge/Application/msedge.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    on_path = shutil.which("msedge")
    if on_path:
        return Path(on_path)
    raise FileNotFoundError(
        "Microsoft Edge was not found; install Edge or add msedge to PATH."
    )


def main() -> None:
    if not HTML.is_file():
        raise FileNotFoundError(HTML)
    browser = _find_edge()
    with tempfile.TemporaryDirectory(prefix="screen-pci-summary-") as profile:
        command = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={PDF}",
            HTML.as_uri(),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            "Microsoft Edge PDF export failed "
            f"(exit {result.returncode}): {result.stderr.strip()}"
        )
    if not PDF.is_file() or PDF.stat().st_size == 0:
        raise RuntimeError(
            f"Microsoft Edge did not create the expected PDF: {PDF}"
        )
    print(f"Created {PDF}")


if __name__ == "__main__":
    main()
