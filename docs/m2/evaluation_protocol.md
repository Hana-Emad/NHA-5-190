# WARDIQ M2 Evaluation Protocol

**Status:** Draft — pending team agreement
**Purpose:** Define a reproducible evaluation protocol for selecting a category predictor and visual embedding model for M3.

## 1. Objective

Evaluate ResNet-50 and CLIP using consistent eligible examples, labels, metric definitions, and reporting conventions wherever the available implementations permit a valid comparison.

The category predictor and embedding model are separate decisions. The selected models may be different.

## 2. Current inputs and versions

- M1 split manifest: `data/manifests/split_manifest.csv`
- M1 taxonomy: `configs/m1/taxonomy.json`
- Taxonomy version: `1.0.0`
- ResNet baseline configuration: `configs/m2/resnet50_baseline_fmnist.yaml`
- ResNet fine-tuning configuration: `configs/m2/resnet50_finetune_fmnist.yaml`
- CLIP model: `openai/clip-vit-base-patch32`
- CLIP model revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`
- Existing model comparison: `docs/m2/model_comparison.md`
- Existing CLIP error analysis: `docs/m2/clip_zero_shot_error_analysis.md`

The taxonomy is currently marked as finalized by the task owner, pending lead review. Cross-dataset label compatibility must not be assumed until the mapping is approved.

## 3. Evaluation population and eligibility

The primary comparison population must contain item IDs that are eligible for every method being compared.

For each candidate item, record:

- `item_id`
- `source_dataset`
- `target_split`
- `common_category`
- `category_mapping_status`
- `image_path` or the canonical image reference
- `leakage_group_id`, when applicable
- Eligibility or exclusion reason for each method

Use validation examples for model selection. Exclude test examples from training, prompt selection, threshold selection, and model selection.

The existing M1 split assigns all Fashionpedia and Polyvore items to validation. It does not provide training examples for those datasets. A real-garment ResNet experiment therefore requires an explicitly approved training-split decision or a documented deferral.

The existing Fashion-MNIST ResNet results must be reported separately from real-garment results. They cannot establish comparative performance against CLIP on Fashionpedia and Polyvore.

**Common evaluation IDs:** Pending a reproducible intersection of the approved eligible manifests and the available model outputs. The existing 18-item CLIP zero-shot sample is exploratory and is not yet an approved common comparison set.

## 4. Category mapping

Use the versioned M1 taxonomy in `configs/m1/taxonomy.json`.

Record the taxonomy version and mapping status with each result. Do not silently remap labels after examining model errors. Any taxonomy changes require an explicit review and a new recorded version.

Items with unsupported, ambiguous, or excluded labels must be counted and reported. Do not treat absent categories as having zero recall or zero F1 without explicitly defining that convention.

## 5. Metrics and reporting conventions

For every model and evaluation population, report:

- Accuracy
- Macro-F1
- Per-class precision, recall, F1, and support
- Confusion matrix
- Number of eligible, evaluated, and excluded items
- Coverage by dataset, split, and category
- Runtime and execution device, where measured
- Embedding extraction cost and memory observations, where measured
- Model identifier, revision, configuration, seed, and environment
- Known reproducibility limitations

Accuracy is the fraction of evaluated items classified correctly. Macro-F1 is the unweighted mean of per-class F1 over the explicitly stated set of included classes.

Report the class set used for macro-F1, particularly when some categories have no eligible examples. Include per-class support so small or missing categories remain visible.

Do not compare metrics from different datasets as if they came from the same evaluation population.

## 6. Model-specific evaluation

### ResNet-50

The existing Fashion-MNIST baseline and fine-tuning runs are pipeline results. Their validation population contains 6,000 Fashion-MNIST items.

A real-garment ResNet category result is not established in the current documentation. Do not present the Fashion-MNIST scores as real-garment classification performance.

If real-garment training is unavailable, record the blocker and obtain an explicit decision about deferral or an approved split.

### CLIP zero-shot

The existing experiment evaluated 18 exploratory Fashionpedia and Polyvore examples using the pinned CLIP model revision.

Record its results separately until a valid shared evaluation population is established. Do not tune prompts on the final test set.

### CLIP linear probe

Record this experiment as unavailable until a runnable implementation and actual evaluation outputs are verified. Do not infer its performance from zero-shot results.

### CLIP image embeddings

The current embedding extraction prototype contains 23 items. It does not establish full accepted-population coverage or final M3 integration.

Verify embedding-to-item alignment, normalized vector shape, pinned model revision, artifact metadata, and reproducible extraction before claiming full-scale acceptance.

## 7. Error analysis and embedding usefulness

Inspect representative classification errors by class and source. Record item IDs, true labels, predictions, and relevant image references.

Where feasible, perform a documented retrieval or nearest-neighbor check using the extracted embeddings. Record the query items, candidate population, similarity method, and qualitative or quantitative findings.

Do not claim final outfit compatibility or recommendation performance from these M2 proxy evaluations.

## 8. Model selection and M3 recommendation

Select the category predictor and embedding model separately.

For each decision, record:

- Selected model and exact revision
- Validation evidence and evaluation population
- Accuracy, macro-F1, and per-class evidence where applicable
- Coverage and compute tradeoffs
- Known limitations
- Unavailable or deferred experiments
- Runnable configuration and artifact references

If the available evidence cannot support a fair comparison, explicitly state that no comparative winner has been established and identify the evidence required to decide.

Keep test data untouched until model choices are frozen.

## 9. Reproduction and handoff

For each completed experiment, record the exact command, configuration, environment versions, random seed, input manifest, output paths, and model revision.

Keep large model weights and generated embedding caches outside Git unless the repository's artifact policy explicitly permits them.

Provide the final decision record and artifact references to Hana for integration and Eyad for caching.

## 10. Current blockers and next actions

1. Obtain team agreement on the taxonomy and the eligible evaluation population.
2. Publish the actual eligible item IDs and reporting conventions to Asmaa and Hayat.
3. Determine whether a valid shared validation set can be evaluated by both category-prediction approaches.
4. Verify the status of the ResNet real-garment baseline and CLIP linear-probe experiment.
5. Complete the retrieval check where feasible.
6. Consolidate measured compute, coverage, errors, and reproducibility details.
7. Make and document the two M3 model recommendations, or record why the evidence is insufficient.

**Current status:** Protocol draft prepared. Common-set comparison and final M3 recommendations remain pending.