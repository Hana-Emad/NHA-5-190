# M2 — Baseline garment category classifier (ResNet-50): handoff package

One package with everything for this task. **Status: In progress. Handoff Ready: No.**

## READ THIS FIRST (if you are integrating or comparing this model)
1. **What it is:** a ResNet-50 baseline and a bounded fine-tune, trained on **Fashion-MNIST** with the M1 split unchanged. It is a *pipeline baseline*. It is **not** a garment classifier for real photos.
2. **Labels are the 10 Fashion-MNIST classes** (`Ankle boot`, `Bag`, `Coat`, `Dress`, `Pullover`, `Sandal`, `Shirt`, `Sneaker`, `T-shirt/top`, `Trouser`). They are **not** the common taxonomy (`top`, `bottom`, ...). The output key of `predict_item` is called `common_category` for schema compatibility, but for these checkpoints its value is a Fashion-MNIST label such as `"Ankle boot"`. Do not feed it to code that expects common-taxonomy names.
3. **Checkpoints** (`baseline_fmnist.pt`, `finetune_fmnist.pt`) are not in Git. The Drive link is on the Notion task page; verify the sha256 (listed below and in `docs/m2/resnet_category.md`).
4. **Load and predict** (run from the repo root; needs torch and torchvision):
```python
from PIL import Image
from src.wardiq.clothing.resnet_category import load_for_inference, predict_item
model, classes, size = load_for_inference("baseline_fmnist.pt")   # or finetune_fmnist.pt
predict_item(model, classes, Image.open("some_image.png"), size, item_id="example")
# -> {"item_id": "example", "category": {"common_category": "<Fashion-MNIST label>", "confidence": 0.0-1.0}}
```
5. **Decisions still needed (this is why Handoff Ready is No):**
   - *Comparison / representation owner:* is a Fashion-MNIST baseline enough for the M2 comparison for now, or is a model trained on real garments required?
   - *Splits owner (or Project Lead):* if real garments are required, `split_manifest.csv` has no train split for Fashionpedia and Polyvore (all items are in `validation`). Provide a train split, or approve a derived group-aware split inside validation (seed 42) and record it as a split-protocol decision.
   - Until these are decided, treat this package as a pipeline baseline only.

## What is in here
| Path | What it is |
|---|---|
| `src/wardiq/clothing/resnet_category.py` | training, evaluation, checkpoint saving, inference function (`predict_item`) |
| `scripts/build_classification_manifest.py` | builds the single-label training table; labels resolved by source category id |
| `configs/m2/` | run configurations (baseline, fine-tune, Fashion-MNIST variants, CPU check) |
| `tests/test_m2_resnet.py` | tests (3 passed, 1 skipped: needs torch) |
| `docs/m2/resnet_category.md` | run guide, decisions, run record, results, error analysis, limitations |
| `docs/m2/run_on_colab.ipynb` | step-by-step Colab notebook that reproduces the runs |
| `reports/m2/resnet/resnet50_baseline_fmnist/` | baseline results: metrics, item-keyed validation predictions, confusion matrix, class map, reload check |
| `reports/m2/resnet/resnet50_finetune_fmnist/` | same for the bounded fine-tune |
| `reports/m2/resnet/error_examples.csv`, `error_examples_*.png` | most confident mistakes for each run's five largest confusions, as a table and as contact sheets |

## Results (validation, 6,000 items, seed 42, test not used)
Run on Fashion-MNIST with the M1 split unchanged (train 54,000 / validation 6,000 / test 10,000). This is a **pipeline baseline**, not the garment classifier.
| Run | Accuracy | Macro-F1 | Best epoch | Runtime |
|---|---|---|---|---|
| Baseline (frozen backbone, head only) | 0.835 | 0.833 | 5 | 1235.0 s |
| Fine-tune (`layer4` unfrozen) | 0.899 | 0.900 | 4 | 1330.6 s |

## How to consume the outputs
- `val_predictions.csv`: one row per validation item — `item_id`, `true`, `pred`, `confidence`, `source_dataset`.
- `class_to_idx.json`: class order used by the model. Labels are Fashion-MNIST source labels.
- Inference: `load_for_inference(...)` and `predict_item(...)` in `resnet_category.py` return `{item_id, category: {common_category, confidence}}`.

## Required inputs to re-run (from M1, not included)
`configs/m1/taxonomy.json`, `scripts/apply_taxonomy.py`, `data/manifests/split_manifest.csv`, `raw_manifest.csv` or `clean_manifest.csv`. Fashion-MNIST images download automatically.

## Not in this package / open items
- **Checkpoints:** `baseline_fmnist.pt` and `finetune_fmnist.pt` (about 94 MB each) are not in this package or in Git. They are stored on Drive (link in the Notion task). sha256: baseline `2eb8e1a4002500c87b577648744b18da7d15eaf63194ab281daf239bcfe8cb0e`, fine-tune `298e82ae57ad39ef46b094996aeb2464735951d22407d6edaf2e4f3a3ceaf7a9`. They come from a re-run with the same configs and seed 42, which reproduced the first run's predictions, confusion matrices and metrics exactly (only runtimes differ: first run 1214.4 s / 1277.8 s).
- **Real-garment run (Fashionpedia + Polyvore):** not run, see decision 2 in the box at the top.
- **Is the Fashion-MNIST baseline enough for the downstream comparison?** Open, see decision 1 in the box at the top.
- Late: the task deadline was October 3, 2026.

## Known limitations
One seed, no variance estimate; both runs ended at their epoch cap while still improving; the fine-tune had extra training epochs; Fashion-MNIST only; only 50 misclassified examples were viewed (the most confident mistakes), so causes of errors and label-noise rates are not established. Details in `docs/m2/resnet_category.md`.

Model Checkpoints

The trained M2 checkpoints are stored separately from this Git repository.

* Google Drive: [https://drive.google.com/drive/folders/1GIZIXv6dueHUKX-El0MfkDdDga_veFYt?usp=sharing]
* Files: ResNet-50 baseline and fine-tuned checkpoints (.pt)
* Access: Team members must have permission to access the shared folder.
* Usage: Download the required checkpoint and follow the M2 configuration and documentation for loading and evaluation.

The checkpoints are not included in the Git repository to keep it lightweight.