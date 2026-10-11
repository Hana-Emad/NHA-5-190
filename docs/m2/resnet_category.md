# M2 — Baseline garment category classifier and ResNet-50 fine-tuning

Tasks: *Implement baseline garment category classifier* and *Fine-tune and evaluate ResNet-50*.
**Status of this document: Path A (Fashion-MNIST) training results are recorded below from the run outputs; Path B (real garments) is not run; error analysis is pending.**

## Required inputs
| Input | Path | Owner |
|---|---|---|
| Taxonomy v1.0.0 (+ `scripts/apply_taxonomy.py`) | `configs/m1/taxonomy.json` | M1 taxonomy task |
| Clean manifest, split manifest | `data/manifests/clean_manifest.csv`, `split_manifest.csv` | Ziad, Omar |
| Raw images (not committed) | `data/raw/` — refs in the manifest are repo-relative; Fashionpedia refs are `data/raw/...zip::member`. `image_root: .` in the configs | Eyad |

## Decision
1. The M2 baseline run is **Path A**: Fashion-MNIST with the M1 split unchanged. It needs no new approval and no raw image files.
2. It is reported as a **pipeline baseline**, not as the garment classifier. Handoff Ready stays **No** until Path B (real garments) is run, or the Project Lead accepts Path A as sufficient for this stage.
3. **Path B is not run** and no derived split is used for any reported number until the split decision (below) is recorded. `--derive-split` is allowed only for local exploration.
4. Fine-tuning follows the baseline on the same split; if time or compute is short it is recorded as a deferral and the baseline stays the deliverable.

## Two runs, and what each one means
| | Path A — Fashion-MNIST, Omar's M1 split unchanged | Path B — Fashionpedia + Polyvore |
|---|---|---|
| Split | train 54,000 / validation 6,000 / test 10,000 (exactly `split_manifest.csv`) | **no train exists** in M1 (0 / 1,158 / 0 and 0 / 14,657 / 0) |
| Images | official IDX files, downloaded automatically | raw images in `data/raw/` |
| Labels | 10 Fashion-MNIST classes | `common_category` (working label, not yet approved) |
| What it proves | the ResNet pipeline, metrics and checkpoint work on the agreed split | the real garment classifier M2 needs |
| Limitation | Fashion-MNIST was selected in M1 for pipeline checks only; its classes are not mapped to the other datasets, so this is **not** the final garment model | needs a train split (Omar/Eyad) or an approved derived split |

## Setup and run
```bash
pip install torch torchvision scikit-learn pandas pyyaml pillow
# Path A: Fashion-MNIST with Omar's split
python scripts/build_classification_manifest.py --datasets "Fashion-MNIST" --label-space source_label --out data/manifests/classification_manifest_fmnist.csv
python -m src.wardiq.clothing.resnet_category --config configs/m2/resnet50_baseline_fmnist.yaml
python -m src.wardiq.clothing.resnet_category --config configs/m2/resnet50_finetune_fmnist.yaml   # needs the baseline checkpoint

# Path B: real garments (see "Open decision" before using --derive-split)
python scripts/build_classification_manifest.py
python -m src.wardiq.clothing.resnet_category --config configs/m2/resnet50_baseline.yaml
python -m src.wardiq.clothing.resnet_category --config configs/m2/resnet50_finetune.yaml
python -m pytest tests/test_m2_resnet.py -q
# optional laptop-CPU check (not a result): configs/m2/resnet50_baseline_fmnist_cpu_check.yaml
```
Outputs under `artifacts/m2/<run>/`: `checkpoint.pt` (not committed), and the small result files that are committed: `class_to_idx.json`, `val_predictions.csv` (item-keyed), `val_metrics.json` (accuracy, macro-F1, per-class support/P/R/F1, cost, history), `val_confusion_matrix.csv`, `reload_check.json` (`image_coverage.csv` as well for Path B); `artifacts/m2/error_examples.csv` holds the error examples for both runs. Keep model weights out of Git (`*.pt` is git-ignored) and document where the checkpoint is stored.

## Design decisions
- Labels resolved **by source category id** via `map_category`, never by zipping `category_ids`/`category_labels`.
- Default label space `common_category`, default `--allowed-status direct`. The 11-category vocabulary is **not approved for cross-dataset training**, so treat it as a working label; use `--label-space source_label` for a per-dataset model.
- Fashionpedia images with several garments are excluded (single-label task); Hana's per-garment crops are the right source for them.
- Part categories (ids 27–45) are ignored as labels. Fashion-MNIST excluded unless `--include-fashion-mnist` (pipeline checks only).
- Fit on `train`; choose checkpoint and stop on `validation` macro-F1; `test` is read only with `--final-test`, after the checkpoint is frozen.
- Baseline: frozen backbone, head only. Fine-tune: start from the **unchanged** baseline, unfreeze `layer4`, backbone LR 10x lower, patience 2. Compare on the same validation IDs and class map.

## Open decision (blocks trustworthy results)
`split_manifest.csv` puts every Fashionpedia (1,158) and Polyvore (14,657) item in `validation`; there is no train or test for them. The builder **refuses to run** in that case. Either Omar/Eyad provide a train split, or the team approves a derived group-aware split (`--derive-split`, grouped by outfit, seed 42, recorded as `split_source = derived_..._PENDING_TEAM_APPROVAL`) and logs it in Notion as a split-protocol decision.

## Verification so far
| Check | Result |
|---|---|
| Builder refuses when no train split; derived split keeps each outfit in one split and reproduces exactly | passed (`tests/test_m2_resnet.py`) |
| Fashion-MNIST builder keeps Omar's split assignments untouched (synthetic rows) | passed (`tests/test_m2_resnet.py`) |
| `build_model`: baseline trains only the head; fine-tune adds `layer4` | exercised by the two Colab runs; the unit test itself was not run locally (no torch there) |
| Re-run with the same configs and seed 42 (second training run) | `val_predictions.csv`, both confusion matrices and `class_to_idx.json` are byte-identical to the first run; accuracy, macro-F1, per-class metrics and best epoch are identical; only the runtime differs |
| Path A baseline and fine-tune training runs (Colab GPU, `run_on_colab.ipynb`) | completed; checkpoint reload check 20/20 sample predictions reproduced for both runs |

## Results (from `val_metrics.json`; runtimes are from the re-run that produced the saved checkpoints — the first run took 1214.4 s and 1277.8 s)
| Run (Path A = Fashion-MNIST, Path B = real garments) | Eval items | Accuracy | Macro-F1 | Best epoch | Runtime | Device |
|---|---|---|---|---|---|---|
| A: Baseline (Fashion-MNIST) | 6,000 | 0.835 | 0.833 | 5 | 1235.0 s | cuda (GPU) |
| A: Fine-tune layer4 (Fashion-MNIST) | 6,000 | 0.899 | 0.900 | 4 | 1330.6 s | cuda (GPU) |
| B: Baseline (real garments) | not run, blocked on the split decision | | | | | |
| B: Fine-tune layer4 (real garments) | not run, blocked on the split decision | | | | | |

### Run record (Path A)
- Split: M1 `split_manifest.csv` unchanged (train 54,000 / validation 6,000 / test 10,000), `split_source = m1_split_manifest`; **test was not read**.
- Seed 42, ResNet-50 (ImageNet weights), image size 224, trainable parameters 20,490 (baseline: new head only) and 14,985,226 (fine-tune: head plus `layer4`), checkpoint and early stopping chosen on validation macro-F1 only.
- Baseline: frozen backbone, head only, lr 0.001, stopped at the 5-epoch cap (validation macro-F1 still rising: 0.795, 0.814, 0.821, 0.827, 0.833).
- Fine-tune: from the unchanged baseline, `layer4` unfrozen, head lr 0.0003, backbone lr 0.00003, stopped at the 4-epoch cap (validation macro-F1: 0.871, 0.886, 0.892, 0.900).
- Fine-tune is +0.065 accuracy and +0.067 macro-F1 over the baseline on the same 6,000 validation items. The fine-tune also received extra training epochs on top of the baseline, so the gain is not attributable to unfreezing alone.

### Checkpoints
| File | Size (bytes) | sha256 |
|---|---|---|
| `baseline_fmnist.pt` | 94,433,745 | `2eb8e1a4002500c87b577648744b18da7d15eaf63194ab281daf239bcfe8cb0e` |
| `finetune_fmnist.pt` | 94,433,873 | `298e82ae57ad39ef46b094996aeb2464735951d22407d6edaf2e4f3a3ceaf7a9` |

Stored outside Git (about 94 MB each); the download link is recorded in the Notion task. Both files contain `state_dict` (320 tensors), the class list, the training config, the epoch (5 and 4) and the architecture name `resnet50`; the metadata matches the run record. The reload check (20 sample predictions reproduced, both runs) was run in the training session from the saved checkpoint and is stored in `reload_check.json`. Load with `load_for_inference(path)` in `resnet_category.py`.

### Transforms and versions
- Inference and evaluation (deterministic): resize to 224×224, convert to RGB (Fashion-MNIST is grayscale), tensor, normalise with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. No statistics are fitted from the data, so no test leakage.
- Training only: random resized crop (224, scale 0.7–1.0) and random horizontal flip, then the same normalisation.
- Model: torchvision ResNet-50 with `IMAGENET1K_V2` weights, new 10-class head; baseline trains the head only, fine-tune also unfreezes `layer4`.
- Labels: Fashion-MNIST source labels (taxonomy `1.0.0`, mapping status `direct`); class order is alphabetical, saved in `class_to_idx.json`.
- Split: M1 `split_manifest.csv` (method `stratified_random_split`, seed 42; official Fashion-MNIST test kept as test): train 54,000 / validation 6,000 / test 10,000, recorded as `split_source = m1_split_manifest`.
- Training seed 42; learning rates, batch sizes, epochs and stopping rule are in `configs/m2/*_fmnist.yaml`.

### Per-class results (validation, 600 items per class; precision / recall / F1, from `val_metrics.json`)
| Class | Baseline | Fine-tune |
|---|---|---|
| Ankle boot | 0.931 / 0.918 / 0.924 | 0.970 / 0.957 / 0.963 |
| Bag | 0.956 / 0.938 / 0.947 | 0.980 / 0.972 / 0.976 |
| Coat | 0.723 / 0.753 / 0.738 | 0.815 / 0.850 / 0.832 |
| Dress | 0.743 / 0.885 / 0.808 | 0.907 / 0.890 / 0.898 |
| Pullover | 0.817 / 0.745 / 0.779 | 0.882 / 0.808 / 0.843 |
| Sandal | 0.871 / 0.925 / 0.897 | 0.968 / 0.967 / 0.967 |
| Shirt | 0.669 / 0.540 / 0.598 | 0.697 / 0.783 / 0.738 |
| Sneaker | 0.892 / 0.878 / 0.885 | 0.938 / 0.962 / 0.950 |
| T-shirt/top | 0.760 / 0.817 / 0.787 | 0.870 / 0.825 / 0.847 |
| Trouser | 0.988 / 0.947 / 0.967 | 0.995 / 0.980 / 0.987 |

### Error analysis (Path A, from `val_confusion_matrix.csv` and `val_predictions.csv`)
Each class has 600 validation items, so a count of N is N/600 of that class. Total errors: baseline 992 of 6,000, fine-tune 604 of 6,000. Accuracy and both confusion matrices were recomputed from `val_predictions.csv` and match `val_metrics.json`; all 6,000 evaluated IDs in each run are `validation` items in the M1 split, and both runs use the same IDs.

| Baseline: true → predicted | Baseline | Fine-tune | Crosses a common category? |
|---|---|---|---|
| Shirt → T-shirt/top | 101 | 49 | no (both `top`) |
| Shirt → Coat | 87 | 39 | yes (`top` → `outerwear`) |
| Pullover → Coat | 71 | 48 | yes (`top` → `outerwear`) |
| Coat → Dress | 57 | 13 | yes (`outerwear` → `one_piece`) |
| T-shirt/top → Dress | 57 | 23 | yes (`top` → `one_piece`) |

| Fine-tune's largest: true → predicted | Baseline | Fine-tune | Crosses a common category? |
|---|---|---|---|
| T-shirt/top → Shirt | 38 | 72 | no |
| Pullover → Shirt | 53 | 51 | no |
| Shirt → T-shirt/top | 101 | 49 | no |
| Pullover → Coat | 71 | 48 | yes (`top` → `outerwear`) |
| Coat → Shirt | 41 | 47 | yes (`outerwear` → `top`) |

What the numbers show:
- In both runs the five largest confusions involve only T-shirt/top, Shirt, Pullover, Coat and Dress. None involves footwear, bags or trousers. Trouser is the best class in both runs (F1 0.967 and 0.987); Shirt is the weakest (F1 0.598 and 0.738).
- Fine-tuning reduced the baseline's five largest confusions: 101 → 49, 87 → 39, 71 → 48, 57 → 13 and 57 → 23.
- It did not reduce every confusion: T-shirt/top → Shirt rose from 38 to 72 and Coat → Shirt from 41 to 47; Pullover → Shirt stayed about the same (53 → 51).
- The five largest confusions are 373 of the baseline's 992 errors (37.6%) and 267 of the fine-tune's 604 (44.2%), so the fine-tune's remaining errors are more concentrated in a few pairs, mostly among top-wear classes.
- Error propagation: T-shirt/top, Shirt and Pullover all map to the common category `top`, so confusions among them do not change the common category. Shirt → Coat, Pullover → Coat, Coat → Dress and T-shirt/top → Dress do (`top`/`outerwear`/`one_piece`). In a pipeline that used this common taxonomy, those would change which outfit templates an item can fill. Fashion-MNIST is not used downstream, so this is illustrative.

### Error examples and inspection
`artifacts/m2/error_examples.csv` lists, for each run's five largest confusions, the five most confident mistakes (item ID, labels, confidence, image reference). The selection rule is mechanical (highest confidence first). Contact sheets: `artifacts/m2/error_examples_baseline.png` and `error_examples_finetune.png` (50 images in total, enlarged 4×).

The 50 images were looked at once (a qualitative, subjective reading, not a labelling audit). Because these are the most *confident* mistakes, the sample is biased toward ambiguous or doubtful labels and says nothing about how common they are.
- **Genuinely ambiguous classes.** Many images sit between two classes: plain long-sleeve tops are labelled Pullover but predicted Coat or Shirt; button-front long-sleeve tops labelled Shirt or Coat are predicted as the other; sleeveless long tops labelled T-shirt/top are predicted Dress.
- **Some labels look debatable.** Camisole or tank-top-like items labelled Shirt (for example #46017 and #37304) and long, narrow items labelled Coat that look like dresses (for example #8917 and #29825). In these the prediction looks as plausible as the given label.
- **Some mistakes look like real model errors**, e.g. plain long-sleeve tops labelled Pullover and predicted Coat.
- **Label integrity of the pipeline:** the category id and label text in `raw_manifest.csv` equal the official Fashion-MNIST label file for all 60,000 training-IDX rows (checked against the official file from the Fashion-MNIST repository). This shows the pipeline did not scramble labels; it does not say the official labels are always right.
- **Class imbalance:** none in this setup. The official classes are balanced; in the M1 split each class has 5,400 train, 600 validation and 1,000 test items.
- **Source domain:** one source only (Fashion-MNIST, 28×28 grayscale). Behaviour on real photos is untested.

Not established: how frequent doubtful labels are, and the cause of each error. A full separation of model errors from label problems would need a review of many more images.

## Limitations
The first run's checkpoints were not saved (the Colab session ended), so the saved checkpoints come from a re-run with the same configs and seed, which reproduced the first run exactly; one seed (no estimate of variation across seeds); both runs ended at their epoch cap while still improving; Path A is Fashion-MNIST (pipeline baseline, not the garment classifier); Path B not run (split decision pending); image coverage for Fashionpedia/Polyvore unverified; the error inspection covers 50 hand-viewed examples (the most confident mistakes), so causes and label-noise rates are not established; reload check compares 20 sample predictions only.
