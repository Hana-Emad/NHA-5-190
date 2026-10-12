# CLIP Zero-Shot Error Analysis

## 1. Purpose

This report documents a visual review of the misclassifications from the WARDIQ Milestone 2 CLIP zero-shot experiment. The purpose is to investigate possible sources of error without modifying the original predictions, evaluation metrics, or taxonomy.

## 2. Experiment Details

- Model: `openai/clip-vit-base-patch32`
- Evaluation dataset: Fashionpedia and Polyvore Outfits
- Evaluated items: 18
- Correct predictions: 12
- Incorrect predictions: 6
- Accuracy: 66.7%
- Macro-F1: 0.599 across observed categories
- Evaluation status: Exploratory only

The sample is small and should not be treated as a reliable estimate of general model performance.

## 3. Misclassified Items

| Item ID | Expected Category | Predicted Category | Mapping |
|---|---|---|---|
| fashionpedia_val_ann_008929 | footwear | accessory_other | direct |
| fashionpedia_val_ann_008930 | footwear | accessory_other | direct |
| fashionpedia_val_ann_008932 | accessory_other | headwear | approximate |
| fashionpedia_val_ann_002555 | top | accessory_other | direct |
| fashionpedia_val_ann_005698 | outerwear | eyewear | approximate |
| fashionpedia_val_ann_005700 | jewellery | bottom | approximate |

All six errors came from Fashionpedia. The experiment included five Polyvore items, but none of those five was misclassified.

## 4. Visual Inspection

The six corresponding image crops were inspected.

Observations:

- The garments or items were visible in the inspected crops.
- Some crops may contain two items, which could introduce competing visual information.
- The expected categories appeared consistent with the visual review.
- The category mappings appeared clear to the reviewer; no immediate taxonomy change was justified by this inspection.

All six image paths and output paths existed. Each item used the `bbox_crop` crop method.

## 5. Interpretation and Limitations

The visual review did not identify an obvious missing-image problem or a clear labeling issue. Crops containing multiple items may contribute to classification errors, but this has not been established as their cause.

The three approximate mappings should remain documented for future review. They should not be changed solely because CLIP predicted a different category.

The evaluation has several limitations:

- Only 18 items were evaluated.
- Only eight of the eleven taxonomy categories were represented.
- Fashionpedia crops may share source images, so observations may not be independent.
- The current experiment does not establish how CLIP will perform on a broader garment dataset.
- The visual review was qualitative and did not experimentally isolate the causes of individual errors.

## 6. Decisions

1. Preserve the existing `predictions.csv` and `summary.json`.
2. Do not change the taxonomy based solely on these six errors.
3. Retain the error records for further analysis.
4. Treat the CLIP results as an exploratory baseline.
5. Continue the Milestone 2 comparison while clearly documenting differences in datasets and evaluation conditions.

## 7. Reproducibility

Relevant artifacts:

- `scripts/m2/run_clip_zero_shot.py`
- `artifacts/m2/clip_zero_shot/predictions.csv`
- `artifacts/m2/clip_zero_shot/summary.json`
- `artifacts/m2/clip_zero_shot/error_analysis.csv`
- `artifacts/m2/clip_zero_shot/error_contact_sheet.jpg`
- `artifacts/m1/garment_manifest.csv`

The experiment uses the pinned model revision recorded in the CLIP summary. The error analysis is based on the saved predictions and a visual review of the generated crops.

Report status: Exploratory error analysis completed; broader evaluation remains necessary.