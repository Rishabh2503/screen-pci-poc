"""Report core Python and external executable availability."""

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version


CORE_MODULES = {
    "pillow": "PIL",
    "numpy": "numpy",
    "opencv-python": "cv2",
    "pytesseract": "pytesseract",
    "pytest": "pytest",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
}
OPTIONAL_MODULES = {
    "presidio-image-redactor": "presidio_image_redactor",
    "ultralytics": "ultralytics",
}
COMMON_TESSERACT_PATHS = (
    Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    / "Tesseract-OCR"
    / "tesseract.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    / "Tesseract-OCR"
    / "tesseract.exe",
)


def find_tesseract():
    configured_path = os.environ.get("TESSERACT_CMD")
    if configured_path:
        candidate = Path(configured_path.strip('"'))
        if candidate.is_file():
            return candidate, "TESSERACT_CMD"
        return None, f"TESSERACT_CMD points to a missing file: {candidate}"

    path_executable = shutil.which("tesseract")
    if path_executable:
        return Path(path_executable), "PATH"

    for candidate in COMMON_TESSERACT_PATHS:
        if candidate.is_file():
            return candidate, "common installation location"

    return None, (
        "Tesseract was not found. Install it from a verified Windows package, "
        "or set TESSERACT_CMD to the full path of tesseract.exe."
    )


def check_tesseract():
    executable, source = find_tesseract()
    if not executable:
        print(f"Tesseract: NOT AVAILABLE ({source})")
        return False

    try:
        version_result = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if version_result.returncode != 0:
            detail = version_result.stderr.strip() or version_result.stdout.strip()
            print(f"Tesseract: FAILED to run from {source}: {detail}")
            return False

        version_line = (version_result.stdout or version_result.stderr).splitlines()[0]
        print(f"Tesseract: {version_line} ({source})")

        languages_result = subprocess.run(
            [str(executable), "--list-langs"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if languages_result.returncode != 0:
            detail = languages_result.stderr.strip() or languages_result.stdout.strip()
            print(f"Tesseract English data: FAILED to list languages: {detail}")
            return False

        languages = {
            line.strip()
            for line in languages_result.stdout.splitlines()
            if line.strip() and not line.lower().startswith("list of available")
        }
        if "eng" not in languages:
            print("Tesseract English data: NOT AVAILABLE (install eng.traineddata)")
            return False

        print("Tesseract English data: available")
        return True
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Tesseract: FAILED to run: {error}")
        return False


def main():
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    python_ok = sys.version_info[:2] == (3, 12)
    print(f"Python 3.12: {'OK' if python_ok else 'NOT FOUND (this interpreter is not Python 3.12)'}")

    missing_core = []
    print("Core package imports:")
    for package, module in CORE_MODULES.items():
        try:
            importlib.import_module(module)
        except Exception as error:
            missing_core.append(package)
            print(f"  MISSING {package}: {error}")
        else:
            try:
                package_version = version(package)
            except PackageNotFoundError:
                package_version = "version unavailable"
            print(f"  OK {package}=={package_version}")

    tesseract_ok = check_tesseract()

    print("Optional package availability:")
    for package, module in OPTIONAL_MODULES.items():
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        if available:
            try:
                package_version = version(package)
            except PackageNotFoundError:
                package_version = "version unavailable"
            print(f"  INSTALLED {package}=={package_version}")
        else:
            print(f"  not installed: {package}")

    if not python_ok or missing_core or not tesseract_ok:
        print("Preflight failed: use Python 3.12, install core requirements, and configure Tesseract with English data.")
        return 1

    print("Environment preflight passed. No OCR, model inference, or GPU test was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())