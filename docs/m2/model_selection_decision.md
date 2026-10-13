# WARDIQ M2 Model Selection Decision

**Status:** Provisional decision based on available evidence
**Taxonomy:** M1 taxonomy v1.0.0, pending lead review
**Test-set policy:** The M1 test split remains untouched for model selection.

## 1. Decision summary

M2 has produced working ResNet-50 classification experiments, an exploratory CLIP zero-shot evaluation, a CLIP embedding extraction prototype, and an exploratory nearest-neighbor diagnostic.

The current evidence does not establish a fair, common-population winner for real-garment category prediction or outfit-oriented visual retrieval. The decisions below distinguish a practical candidate from a model whose superiority has been demonstrated.

| Decision | Current outcome | Confidence |
|---|---|---|
| Category predictor | No final winner established. Retain the fine-tuned ResNet-50 as a classification baseline, not as a validated real-garment predictor. | Insufficient comparative evidence |
| Visual representation | Retain CLIP ViT-B/32 as the leading implementation candidate for further M3 integration evaluation. Do not claim that it is proven superior for outfit compatibility or retrieval. | Prototype verified; task suitability unproven |
| CLIP zero-shot classification | Exploratory result only; not a final category-model selection. | Low, due to small sample |
| CLIP linear probe | Not established as completed; exclude from quantitative comparisons until verified results exist. | Unavailable |
| Retrieval quality | Not validated; the nearest-neighbor run verifies pipeline execution only. | Insufficient evidence |

## 2. Evidence reviewed

### ResNet-50 category classification

The recorded Fashion-MNIST validation results are:

| Experiment | Validation items | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Baseline | 6,000 | 0.835 | 0.833 |
| Fine-tuned layer4 | 6,000 | 0.899 | 0.900 |

These results support the conclusion that fine-tuning improved performance on the Fashion-MNIST validation population. They do not establish performance on Fashionpedia or Polyvore garments because the datasets and label spaces differ.

A real-garment ResNet result has not been established. Under the current M1 split, Fashionpedia and Polyvore are validation-only, so real-garment training requires an approved split decision or an explicitly accepted deferral.

Reference: `docs/m2/resnet_category.md`

### CLIP zero-shot category classification

The exploratory CLIP ViT-B/32 experiment evaluated 18 examples: 13 Fashionpedia and 5 Polyvore.

- Accuracy: 66.7%.
- Macro-F1: approximately 0.599 over the observed categories.
- Eight of the eleven taxonomy categories were represented.
- Six errors were recorded, all from Fashionpedia.
- The sample is small, and possible shared-source-image dependence has not been ruled out.
- Three taxonomy categories had no ground-truth examples in this sample.

These results are useful for error inspection but are not directly comparable to the Fashion-MNIST ResNet results.

Reference: `docs/m2/clip_zero_shot_error_analysis.md`

### CLIP visual representation

The pinned model is `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.

The extraction prototype successfully generated 23 normalized, 512-dimensional embeddings. Nine component tests passed.

A nearest-neighbor diagnostic processed 18 real-garment items and produced 54 query-neighbor rows. Mean cosine similarity was 0.7742, and 5 of 54 pairs shared the same raw category-label string. That label-match count is descriptive only: it is not retrieval precision or accuracy, and raw labels have not been validated as shared ground truth.

The diagnostic also showed examples of high-similarity neighbors with different raw garment labels. Therefore, these results do not prove that CLIP embeddings are suitable for outfit compatibility or that CLIP is superior to alternative representations.

References:
- `docs/m2/clip_embeddings.md`
- `docs/m2/clip_retrieval_diagnostic.md`
- `scripts/m2/run_clip_embeddings.py`
- `scripts/m2/check_clip_embedding_retrieval.py`

## 3. Recommended next steps for M3

### Category predictor

Keep the fine-tuned ResNet-50 as a reproducible Fashion-MNIST baseline. Do not designate it the final real-garment category predictor yet.

Before making that selection:

1. Obtain approval for the eligible real-garment evaluation population and label mapping.
2. Establish a valid training strategy for real garments without using the test split.
3. Evaluate the candidates on the same eligible validation items where possible.
4. Report accuracy, macro-F1, per-class support and metrics, confusion matrices, coverage, and compute requirements.
5. Verify the CLIP linear-probe implementation and results if that approach remains a candidate.

If those experiments cannot be completed within M2, record the category-predictor decision as deferred rather than claiming a winner.

### Visual representation

Carry forward CLIP ViT-B/32 as a **provisional integration candidate**, because its pinned extraction implementation and prototype artifacts have been verified.

Before accepting it as the final representation:

1. Confirm the eligible population and required cache schema with the M2 owners.
2. Run extraction over the approved population and reconcile every inventory row with its embedding row.
3. Compare representations using an agreed retrieval evaluation or human relevance judgments.
4. Measure relevant runtime, memory, and storage costs.
5. Record the exact model revision, preprocessing, and reproducible configuration.

This recommendation is a practical next-step choice, not a validated claim that CLIP provides the best outfit compatibility or recommendation performance.

## 4. Blockers and ownership handoff

The following items remain open:

- Lead review of taxonomy v1.0.0 and agreement on cross-dataset label mapping.
- Agreement on eligible validation IDs and leakage-group handling.
- Decision on real-garment ResNet training or formal deferral.
- Verification of any CLIP linear-probe outputs.
- Evaluation of retrieval quality beyond the current exploratory diagnostic.
- Agreement on the embedding cache and schema contract.
- Confirmation of final M3 selections by the team.

Share this decision record and the associated experiment references with Hana for integration planning and Eyad for caching. Share the eligible item IDs, metric definitions, and limitations with Asmaa and Hayat once the evaluation population is agreed.

## 5. Final status

M2 has working prototype components and documented exploratory evidence. It has not yet demonstrated a fair, common-set comparison across all candidate approaches. The category-predictor selection remains open; CLIP is retained as the provisional visual-representation candidate pending validation.

No test-set results were used to make these decisions.
## 6. M2 Closure Status

For the M2 handoff, carry forward CLIP ViT-B/32 as the provisional visual-representation candidate, not as a proven best model. Keep the fine-tuned ResNet-50 as the Fashion-MNIST classification baseline only.

The following items are deferred and must not be presented as completed experiments:

- Real-garment ResNet training and evaluation, pending an approved training-data strategy.
- CLIP linear-probe evaluation, because no implementation or results were found in the searched project files.
- Validated retrieval evaluation on an agreed eligible population.
- Trained attribute prediction and empirical attribute metrics, because the current Fashionpedia data provides no training-supported labels.

The team still needs to review the taxonomy and cross-dataset mapping, agree on eligible validation IDs and grouping, and confirm the final M3 choices. These decisions should be made without using the M1 test split for model selection.
