# M2 CLIP Embedding Retrieval Diagnostic

## Status

Completed exploratory diagnostic. Not a validated retrieval benchmark.

## Purpose

Check whether the generated CLIP embeddings can retrieve potentially relevant
garment items using cosine similarity. This is a small diagnostic, not a final
evaluation of retrieval quality.

## Configuration

- Model: `openai/clip-vit-base-patch32`
- Model revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`
- Embedding dimension: 512
- Similarity measure: cosine similarity, calculated as the dot product of
  L2-normalized embeddings
- Total embedded items: 23
- Real-garment items: 18 (13 Fashionpedia, 5 Polyvore)
- Queries with at least one eligible neighbor: 18
- Neighbors returned per query: 3
- Total query-neighbor rows: 54

Fashion-MNIST items were excluded from the real-garment retrieval check.
Candidates from the same source-image group within the same dataset were
excluded to reduce trivial same-image retrieval.

## Results

| Metric | Result |
|---|---:|
| Query-neighbor pairs | 54 |
| Mean cosine similarity | 0.7742 |
| Minimum cosine similarity | 0.5522 |
| Maximum cosine similarity | 0.9106 |
| Pairs with identical raw category-label strings | 5/54 (9.3%) |

The raw-label match count is descriptive only. Source datasets use different
label vocabularies, and raw category labels have not been validated as a
common cross-dataset ground truth.

## Qualitative examples

For query `fashionpedia_val_ann_008929` (raw category: `shoe`), the top
neighbors were:

1. `fashionpedia_val_ann_002554` — `glasses`, similarity 0.8192
2. `fashionpedia_val_ann_002555` — `top, t-shirt, sweatshirt`, similarity 0.8170
3. `fashionpedia_val_ann_002557` — `shirt, blouse`, similarity 0.7795

For query `fashionpedia_val_ann_008930` (raw category: `shoe`), the top
neighbors were:

1. `fashionpedia_val_ann_002555` — `top, t-shirt, sweatshirt`, similarity 0.8935
2. `fashionpedia_val_ann_002554` — `glasses`, similarity 0.8711
3. `fashionpedia_val_ann_005700` — `watch`, similarity 0.8606

For query `fashionpedia_val_ann_008931` (raw category: `pants`), the top
neighbors were:

1. `fashionpedia_val_ann_002557` — `shirt, blouse`, similarity 0.8270
2. `fashionpedia_val_ann_002554` — `glasses`, similarity 0.8230
3. `fashionpedia_val_ann_005698` — `vest`, similarity 0.7680

These examples show that high similarity scores do not guarantee matching
garment categories. They do not establish whether the neighbors are useful
for other retrieval goals, such as style or outfit compatibility.

## Limitations

- The real-garment evaluation set is very small: 18 items.
- The raw category labels are not yet validated against the shared taxonomy.
- No human relevance judgments or approved retrieval ground truth were used.
- The raw-label match count is not retrieval precision, recall, or accuracy.
- Results may change with a larger and more representative candidate pool.
- This diagnostic does not establish whether CLIP is the best representation
  for outfit compatibility or recommendation.

## Artifacts

- Retrieval script: `scripts/m2/check_clip_embedding_retrieval.py`
- Neighbor results: `reports/m2/clip_retrieval/nearest_neighbors.csv`
- Embedding documentation: `docs/m2/clip_embeddings.md`
- Embedding prototype output: `artifacts/m2/clip_embeddings_prototype/`
  (local generated artifacts; do not commit unless the team explicitly requires them)

## Conclusion

The embedding extraction and nearest-neighbor pipeline ran successfully on
the current prototype. The retrieval results are exploratory and do not
yet validate retrieval quality. A stronger evaluation requires an agreed
eligible population, a validated common taxonomy or human relevance labels,
and a larger evaluation set.