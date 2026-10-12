# WARDIQ Milestone 2: Model Comparison Report

## 1. Objective

This report summarizes three candidate image-category prediction experiments: a ResNet-50 baseline, a fine-tuned ResNet-50, and CLIP zero-shot classification. The goal is to document their results, limitations, and relevance to selecting a representation for subsequent WARDIQ development.

## 2. Experiment Results

| Metric | ResNet-50 Baseline | Fine-Tuned ResNet-50 | CLIP Zero-Shot |
|---|---:|---:|---:|
| Evaluation items | 6,000 | 6,000 | 18 |
| Dataset | Fashion-MNIST | Fashion-MNIST | Fashionpedia + Polyvore |
| Accuracy | 83.47% | 89.93% | 66.67% |
| Macro-F1 | 0.8331 | 0.9002 | 0.5988 |
| Training epochs | 5 | 4 | None |
| Trainable parameters | 20,490 | 14,985,226 | No task-specific training |
| Recorded runtime | 1,235.0 s | 1,330.6 s | See experiment artifacts |
| Evaluation status | Validation result | Validation result | Exploratory only |

The ResNet results were generated using seed 42 and the M1 split manifest. The CLIP experiment used `openai/clip-vit-base-patch32`, with model revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.

The recorded ResNet runtime and device fields are retained from the original experiment outputs. They should not be interpreted as a controlled hardware benchmark against CLIP.

## 3. ResNet-50 Comparison

Fine-tuning improved validation accuracy by approximately 6.47 percentage points, from 83.47% to 89.93%.

Macro-F1 increased from 0.8331 to 0.9002, an absolute improvement of approximately 0.0671.

The fine-tuned model improved the reported F1 score for every Fashion-MNIST class. The baseline's lowest class F1 was for Shirt, at approximately 0.598, while the fine-tuned model's Shirt F1 increased to approximately 0.738.

These results support fine-tuning over the baseline for this Fashion-MNIST classification task. They do not establish performance on real-world garment photographs.

## 4. CLIP Zero-Shot Findings

The CLIP experiment evaluated 18 eligible items: 13 Fashionpedia crops and 5 Polyvore items. It achieved 66.67% accuracy and a macro-F1 of approximately 0.599 over the eight categories represented in the sample.

Six items were misclassified, all from Fashionpedia. Visual inspection found that the garments were visible and their expected categories appeared reasonable. Some crops contained two items, which may introduce competing visual signals, but this has not been established as the cause of the errors.

The sample did not contain ground-truth examples for three taxonomy categories: `one_piece`, `headwear`, and `scarf`. Their recall values are therefore undefined in this experiment.

The CLIP results are exploratory only. The sample is too small to support reliable general-performance claims, and crops from the same source image may be correlated.

## 5. Important Comparability Limitation

The ResNet-50 experiments used Fashion-MNIST, which contains small, standardized grayscale images across ten clothing classes. CLIP used Fashionpedia and Polyvore items mapped into WARDIQ's eleven-category taxonomy.

The datasets, label vocabularies, sample sizes, and evaluation conditions differ. Consequently, the three accuracy and macro-F1 values must not be interpreted as a fair head-to-head ranking.

The ResNet experiments demonstrate the effect of fine-tuning on Fashion-MNIST. The CLIP experiment provides an initial indication of zero-shot classification behavior on the available WARDIQ-related samples. Neither result alone determines the best representation for WARDIQ's final use case.

## 6. Limitations and Follow-Up Work

1. Expand the CLIP evaluation sample and cover all taxonomy categories where data is available.
2. Review source-image grouping and potential correlation between Fashionpedia crops.
3. Keep the existing taxonomy and recorded metrics unchanged until any proposed revisions are reviewed and approved.
4. Evaluate candidate approaches on a shared, appropriately labeled dataset before making a direct performance comparison.
5. Compare model representations for their usefulness in outfit compatibility and recommendation tasks, not just category classification.
6. Record the environment, configuration, model revision, split manifest, and output artifacts for each experiment.

## 7. Source Artifacts

### ResNet-50 baseline
- Configuration: `configs/m2/resnet50_baseline_fmnist.yaml`
- Metrics: `reports/m2/resnet/resnet50_baseline_fmnist/val_metrics.json`
- Error examples: `reports/m2/resnet/error_examples_baseline.png`

### Fine-tuned ResNet-50
- Configuration: `configs/m2/resnet50_finetune_fmnist.yaml`
- Metrics: `reports/m2/resnet/resnet50_finetune_fmnist/val_metrics.json`
- Error examples: `reports/m2/resnet/error_examples_finetune.png`

### CLIP zero-shot
- Script: `scripts/m2/run_clip_zero_shot.py`
- Predictions: `artifacts/m2/clip_zero_shot/predictions.csv`
- Summary: `artifacts/m2/clip_zero_shot/summary.json`
- Error records: `artifacts/m2/clip_zero_shot/error_analysis.csv`
- Contact sheet: `artifacts/m2/clip_zero_shot/error_contact_sheet.jpg`

## 8. Current Conclusion

Fine-tuning ResNet-50 improved results on Fashion-MNIST. CLIP zero-shot classification has produced a useful preliminary result on a very small sample of Fashionpedia and Polyvore items, but broader evaluation is needed.

No final winner is declared. Selecting a WARDIQ representation requires evaluation under comparable conditions and evidence that the representation supports the project's outfit compatibility and recommendation objectives.

Report status: Preliminary M2 comparison; further evaluation required.