
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts" / "m2" / "clip_embeddings_prototype"
EMBEDDING_PATH = ARTIFACT_DIR / "clip_embeddings.npy"
INVENTORY_PATH = ARTIFACT_DIR / "clip_inventory.csv"
OUTPUT_DIR = ROOT / "reports" / "m2" / "clip_retrieval"
OUTPUT_PATH = OUTPUT_DIR / "nearest_neighbors.csv"

TOP_K = 3


def main():
    embeddings = np.load(EMBEDDING_PATH)
    inventory = pd.read_csv(INVENTORY_PATH)

    if len(inventory) != len(embeddings):
        raise ValueError(
            f"Inventory has {len(inventory)} rows but "
            f"embeddings contain {len(embeddings)} rows."
        )

    if "embedding_row_index" not in inventory.columns:
        raise ValueError("Missing embedding_row_index in inventory.")

    if "item_id" not in inventory.columns:
        raise ValueError("Missing item_id in inventory.")

    inventory["embedding_row_index"] = pd.to_numeric(
        inventory["embedding_row_index"], errors="raise"
    ).astype(int)

    expected_indices = np.arange(len(embeddings))
    actual_indices = inventory["embedding_row_index"].to_numpy()

    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError(
            "Inventory rows are not aligned with embedding row indices."
        )

    if inventory["item_id"].duplicated().any():
        raise ValueError("Duplicate item IDs found in inventory.")

    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values.")

    # Embeddings are documented as L2-normalized, so dot product
    # gives cosine similarity, subject to floating-point tolerance.
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Embeddings are not L2-normalized.")

    # Restrict this diagnostic to real-garment examples.
    # Fashion-MNIST items are smoke-test data, not real-garment evidence.
    real_mask = (
        inventory["source_dataset"].astype(str).str.lower()
        != "fashion_mnist"
    )
    real = inventory.loc[real_mask].copy().reset_index(drop=True)

    real_rows = real["embedding_row_index"].to_numpy()
    real_embeddings = embeddings[real_rows]

    similarities = real_embeddings @ real_embeddings.T
    results = []

    for query_pos, query in real.iterrows():

        query_dataset = str(query["source_dataset"])
        query_group = str(query.get("leakage_group_id", ""))

        candidates = []

        for candidate_pos, candidate in real.iterrows():
            if candidate_pos == query_pos:
                continue

            candidate_dataset = str(candidate["source_dataset"])
            candidate_group = str(candidate.get("leakage_group_id", ""))

            # Exclude crops from the same source image.
            if (
                query_group not in ("", "nan", "None")
                and candidate_group == query_group
                and candidate_dataset == query_dataset
            ):
                continue

            score = float(similarities[query_pos, candidate_pos])
            candidates.append((score, candidate_pos))

        candidates.sort(reverse=True)

        for rank, (score, candidate_pos) in enumerate(
            candidates[:TOP_K], start=1
        ):
            candidate = real.iloc[candidate_pos]

            results.append({
                "query_item_id": query["item_id"],
                "query_dataset": query_dataset,
                "query_category": query.get("category_label", ""),
                "query_source_group": query_group,
                "neighbor_rank": rank,
                "neighbor_item_id": candidate["item_id"],
                "neighbor_dataset": candidate["source_dataset"],
                "neighbor_category": candidate.get("category_label", ""),
                "neighbor_source_group": candidate.get(
                    "leakage_group_id", ""
                ),
                "cosine_similarity": score,
                "same_raw_category_label": (
                    str(query.get("category_label", ""))
                    == str(candidate.get("category_label", ""))
                ),
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    print(f"Total embedding items: {len(embeddings)}")
    print(f"Real-garment items: {len(real)}")
    print(f"Queries with at least one eligible neighbor: "
          f"{len(set(r['query_item_id'] for r in results))}")
    print(f"Neighbor rows written: {len(results)}")
    print(f"Output: {OUTPUT_PATH}")
    print(
        "Note: raw category-label matches are descriptive only; "
        "they do not establish cross-dataset taxonomy agreement."
    )


if __name__ == "__main__":
    main()
