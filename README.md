# Multi-Input Insurance Document Image Classifier | PyTorch Prototype

> **Scope:** An educational, multi-input *image classification* proof of concept. **Not a complete OCR engine** and not validated on real insurance documents. No real documents or sensitive identifiers are included.

## What it does

- Accepts a grayscale **64×64 image** and an auxiliary **one-hot document-type vector**.
- Encodes images with `image_layer` (`Conv2d` 3×3, padding=1, pooling, fully connected layers).
- Encodes type information with `type_layer`, fuses both representations, and produces category logits.
- Trains for **10 epochs** with Adam and cross-entropy loss, evaluates on a holdout, and saves a loss plot.

**Important distinction:** The model predicts a document/code *category* from image + metadata. It does **not** locate text or transcribe identity codes. A real OCR system additionally needs text detection/recognition and evaluation (e.g., CER/WER). Class labels and type count in the demo are illustrative.

## Project structure

```text
src/ocr_model.py          Network architecture (OCRModel)
src/data.py               Synthetic demo + optional CSV image loader
src/train.py              10-epoch training, holdout metrics, loss plot
notebooks/01_walkthrough.ipynb   Guided runnable notebook
examples/dataset_format.csv     Sample CSV schema (paths illustrative)
assets/architecture.svg   Model diagram
LINKEDIN_POST.md          Ready-to-personalize announcement
requirements.txt
```

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --demo --epochs 10
jupyter lab notebooks/01_walkthrough.ipynb
```

Demo output is saved under `outputs/` (ignored by Git). Synthetic examples are **simple geometric patterns** with known category labels, designed only to verify that the code trains. Demo validation accuracy is not evidence of insurance-document OCR performance.

## Use your own legally obtained, redacted data

Prepare a CSV with columns `image_path,type_id,label`, where `image_path` points to a grayscale-compatible image relative to the CSV folder; `type_id` in `[0, num_types - 1]`, label in `[0, num_classes - 1]`. See the *schema-only* example CSV. Paths shown there do not represent files included in this repository.

```bash
python -m src.train --csv private_data/labels.csv --num-types 5 --num-classes 4 --epochs 10
```

The loader resizes images to 64×64; assess if this resolution destroys small printed characters before using it for real OCR. This training script makes a random 80/20 split **for demonstration**. For real documents use train/validation/test splits grouped by customer or document and evaluate on held-out sources; add class balance, precision/recall/F1, confusion matrix, and robustness checks. Do not upload any unredacted policyholder details, IDs, or private images.

## Reproducibility and limitations

- Random seeds are fixed; exact results can differ by environment/device.
- Illustrative 5 document types and 4 classes; adjust to your dataset.
- The training plots and metrics come from **synthetic data only** until you supply real images.
- No claim of operational, commercial, or regulatory suitability.

## Next milestones

1. Gather ethically sourced, anonymized, labeled images.
2. Introduce a real OCR stage (text detection + transcription) and CER/WER evaluation.
3. Evaluate category classification with robust holdout datasets, F1, and confusion matrix.
4. Address deployment, human review, privacy, and error handling.

## License

MIT — see LICENSE.
