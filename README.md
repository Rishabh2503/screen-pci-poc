# screen-pci-poc

Python 3.12 synthetic-only proof of concept for OCR-assisted card-number
detection and redaction. It does not process real customer screens, train
models, or perform screen analytics.

## Prerequisites

- Windows and PowerShell.
- Python 3.12 available through the Windows `py` launcher. Check with
  `py -3.12 --version`.
- Tesseract OCR installed separately, with its executable available on PATH,
  discoverable in a common Windows install location, or configured using
  `TESSERACT_CMD`.

Tesseract is an external executable; installing `pytesseract` with pip does not
install Tesseract itself. Preflight checks the executable version and that
English language data (`eng`) is available; it does not run OCR.

## Core setup

Run these commands from the project directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/preflight.py
```

Current Tesseract version in this environment:

```powershell
tesseract --version
# tesseract v5.5.3.20260724
#  leptonica-1.87.0
#   libgif 5.2.2 : libjpeg 8d (libjpeg-turbo 3.1.3) : libpng 1.6.58 : libtiff 4.7.2 : zlib 1.3.1 : libwebp 1.6.0 : libopenjp2 2.5.4
#  Found AVX512BW
#  Found AVX512F
#  Found AVX512VNNI
#  Found AVX2
#  Found AVX
#  Found FMA
#  Found SSE4.1
#  Found libarchive 3.8.8 zlib/1.3.2 liblzma/5.8.3 bz2lib/1.0.8 liblz4/1.10.0 libzstd/1.5.7 expat/expat_2.8.2 cng/1.0 libb2/system
#  Found libcurl/8.21.0 Schannel zlib/1.3.2 brotli/1.2.0 zstd/1.5.7 libidn2/2.3.8 libpsl/0.21.5 libssh2/1.11.1 WinLDAP
```

The preflight reports the Python version, checks imports for the core packages,
looks for Tesseract, and reports whether optional Presidio and Ultralytics
packages are installed. Missing core imports or a Python version other than
3.12 produce a nonzero exit code. A missing Tesseract executable is reported
but does not mean an OCR test was run or passed.

If Tesseract is not on PATH, set its executable path for the current
PowerShell session before running preflight:

```powershell
$env:TESSERACT_CMD = "C:\path\to\tesseract.exe"
python scripts/preflight.py
```

Use the actual path on your machine; no machine-specific path is assumed here.
`TESSERACT_CMD` takes precedence over PATH and common installation locations.

## Optional dependencies

Install these only when their implementation stages begin. Each file includes
the core requirements as well.

### Presidio image redaction

```powershell
python -m pip install -r requirements-presidio.txt
```

Presidio's NLP analyzer may also need a spaCy language model. Do not assume a
model name or compatibility: after installation, inspect the installed
`presidio-analyzer` and `spacy` versions and follow the matching Presidio
documentation for a compatible model. Validate spaCy compatibility with
`python -m spacy validate`, then install the documented model with:

```powershell
python -m spacy download <compatible-model-name>
```

Replace the placeholder only after confirming compatibility for the installed
versions. This scaffold does not download a model.

### Ultralytics YOLO

```powershell
python -m pip install -r requirements-yolo.txt
```

Ultralytics declares PyTorch and related packages as dependencies, which pip
will resolve during installation. GPU support is not assumed. For a CPU-only
setup, use the official PyTorch installation selector to obtain its current
Windows, pip, Python 3.12 CPU-only command, run that command first, and then
install `requirements-yolo.txt`. Do not use a CUDA command unless the hardware
and driver setup have been confirmed. Importing Ultralytics is not a GPU test.

## Verification

With the core environment installed and active, run:

```powershell
python scripts/preflight.py
python -m pytest
```

Summary: 50 passed, 0 failed, 1 skipped in 6.03s.

For the requested direct checks, run:

```powershell
python --version
tesseract --version
python -c "import PIL, cv2, pytesseract; print('Core imports OK')"
python scripts/preflight.py
```

If Tesseract is configured only through `TESSERACT_CMD`, run its version check
with `& $env:TESSERACT_CMD --version` instead of `tesseract --version`.

The preflight prints the installed versions of core and optional packages when
available. To capture the full resolved dependency set from the active virtual
environment for a reproducible report, run:

```powershell
python -m pip freeze
```

The preflight checks core package imports, Tesseract's executable version and
English language data, and optional package availability. It does not run an
OCR recognition test or claim that model inference or GPU execution succeeded.

## Synthetic demo

Run from the project root with the virtual environment active:

```powershell
python -m pci_poc.demo --output-dir generated/demo
```

The command creates deterministic synthetic screenshots, a ground-truth JSON
manifest, and ground-truth visualizations. When Tesseract is available, it
also runs OCR, detects Luhn-valid candidates, and writes covered outputs. If
Tesseract is unavailable, generated inputs and annotations remain available;
OCR and redacted outputs are skipped and the command returns exit code 2. No
candidate digits are written to logs.

Offline unit tests use mock OCR tokens and do not require Tesseract. The single
real OCR integration test is skipped with a reason when Tesseract is absent.

## Synthetic benchmark

Run a seeded, held-out-template real OCR benchmark with:

```powershell
python -m pci_poc.benchmark --output-dir generated/benchmark --count 48 --seed 20261009
```

The benchmark warms OCR before timing and reports per-image OCR, detection, and
redaction latency. It writes aggregate `metrics.json` and `metrics.csv`,
`per_image.csv`, redacted synthetic screenshots, and sanitized representative
false-positive/false-negative comparisons when those cases occur. UI templates
are held out from tuning; this is a synthetic experiment and is not a claim of
real-world PCI detection or protection performance. Card and decoy text is
obscured in comparison images and never included in report records.

To tune optional field-label context on separate development/validation
templates and compare it with the existing frozen test set, run:

```powershell
python -m pci_poc.benchmark --context-experiment --output-dir generated/context_experiment_final --frozen-dataset-dir generated/benchmark/dataset --seed 20261009 --iou-threshold 0.5
```

This command requires the previously generated frozen benchmark dataset. It
does not regenerate or overwrite that baseline; tuning and paired comparison
outputs are written under the specified new directory.

## Security and privacy

- Use synthetic screenshots only. Never place real customer screenshots,
  recordings, payment-card data, or other sensitive data in this prototype.
- Processing is local. This project does not send images or source files to
  external AI services, image services, or cloud storage.
- Generated screenshots, recordings, raw and unredacted artifacts, temporary
  data, local configuration, virtual environments, and model checkpoints are
  excluded from source control. Ignoring a path does not encrypt or securely
  erase its contents.
- This is an early prototype, not a validated payment-card data protection
  system. OCR and redaction can miss or misclassify sensitive information;
  never rely on it to protect real data.
- The project lives in a local workspace. Keep files on this machine and do not
  store real sensitive data in this location.