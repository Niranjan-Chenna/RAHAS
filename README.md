# RAHAS

RAHAS is a research codebase for a unified Brahmi estampage workflow covering:

1. restoration of degraded estampage scans;
2. segmentation of complete character instances, including multi-component graphemes; and
3. OCR using continuous grayscale and spatial stroke evidence.

## Public release scope

This public repository intentionally contains **code only**.

The dataset, trained model weights, checkpoints, experimental outputs, evaluation artifacts, paper figures and tables, manuscript files, and the complete supporting evidence package are not included because the research paper is still being drafted and prepared for journal submission. These materials remain private during paper preparation and review. Any later release will depend on the publication stage and the permissions governing the underlying heritage data.

Consequently, this repository is intended for inspection of the implementation and cannot reproduce the reported research pipeline end to end without the separately maintained data, weights, configurations, and evidence artifacts.

## Repository layout

```text
pipeline/scripts/       Restoration, segmentation, training, evaluation, and analysis scripts
pipeline/src/           Core restoration and OCR model modules
pipeline/tests/         Automated tests for the public implementation
pipeline/streamlit_app.py
                        Interactive RAHAS workbench
requirements.txt        Python dependencies
```

## Installation

Create a Python virtual environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest pipeline\tests -q
```

## Data and model availability

No dataset samples, source scans, annotations, trained weights, or model checkpoints are distributed in this repository. Please do not treat paths referenced by the research scripts as publicly available resources.

## Citation

Citation information will be added after the associated paper is finalized for publication.
