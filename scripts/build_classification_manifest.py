"""Build the single-label classification manifest used by the ResNet-50 baseline/fine-tune.

One row = one image with exactly ONE garment category. Labels are resolved BY SOURCE ID through
apply_taxonomy.map_category (never by zipping category_ids with category_labels).

Usage (repo root):
    python scripts/build_classification_manifest.py \
        --manifest data/manifests/clean_manifest.csv \
        --splits data/manifests/split_manifest.csv \
        --out data/manifests/classification_manifest.csv
Add --derive-split ONLY if the team has approved a derived split (see docs/m2/resnet_category.md).
Fashion-MNIST with Omar's M1 split unchanged (needs no raw image files):
    python scripts/build_classification_manifest.py --datasets "Fashion-MNIST" --label-space source_label \
        --out data/manifests/classification_manifest_fmnist.csv
"""
import argparse, hashlib, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from apply_taxonomy import DATASET_KEYS, load_taxonomy, map_category  # noqa: E402


def group_id_of(row):
    gid = str(row.get("group_id", "") or "")
    return gid.split(":")[0] if gid else row["item_id"]  # Polyvore group_id = outfit_id:position


def derive_split(df, seed, ratios=(0.70, 0.15, 0.15)):
    """Deterministic group-aware split: all items of one group land in the same split."""
    def bucket(g):
        h = int(hashlib.sha256(f"{seed}:{g}".encode()).hexdigest(), 16) % 10_000 / 10_000
        return "train" if h < ratios[0] else "validation" if h < ratios[0] + ratios[1] else "test"
    return df["group"].map(bucket)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/clean_manifest.csv")
    ap.add_argument("--splits", default="data/manifests/split_manifest.csv")
    ap.add_argument("--taxonomy", default="configs/m1/taxonomy.json")
    ap.add_argument("--out", default="data/manifests/classification_manifest.csv")
    ap.add_argument("--label-space", choices=["common_category", "source_label"], default="common_category")
    ap.add_argument("--allowed-status", default="direct", help="comma list of mapping_status values to keep")
    ap.add_argument("--datasets", default="Fashionpedia,Polyvore Outfits", help="comma list of source_dataset values to include")
    ap.add_argument("--include-fashion-mnist", action="store_true", help="add Fashion-MNIST to --datasets (pipeline checks only)")
    ap.add_argument("--derive-split", action="store_true", help="derived group-aware split (needs team approval)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tax = load_taxonomy(a.taxonomy)
    allowed = set(a.allowed_status.split(","))
    wanted = {d.strip() for d in a.datasets.split(",")} | ({"Fashion-MNIST"} if a.include_fashion_mnist else set())
    man = pd.read_csv(a.manifest, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "cleaning_status" in man.columns:
        man = man[man.cleaning_status == "retained"]
    spl = pd.read_csv(a.splits, dtype=str, keep_default_na=False, encoding="utf-8-sig")[["item_id", "target_split", "group_id"]]
    man = man.merge(spl, on="item_id", how="left", validate="one_to_one")

    rows, dropped = [], {"no_split": 0, "no_garment": 0, "multi_garment": 0, "status_filtered": 0}
    for r in man.to_dict("records"):
        ds = r["source_dataset"]
        if ds not in wanted:
            dropped["other_dataset"] = dropped.get("other_dataset", 0) + 1; continue
        if not r.get("target_split"):
            dropped["no_split"] += 1; continue
        key = DATASET_KEYS[ds]
        ids = sorted({int(i) for i in r["category_ids"].split(";") if i})
        maps = [(i, map_category(tax, key, i)) for i in ids]
        garments = [(i, m) for i, m in maps if m and m["mapping_status"] != "not_a_garment"]
        if not garments:
            dropped["no_garment"] += 1; continue
        if len({m["common_category"] for _, m in garments}) != 1 and a.label_space == "common_category" or \
           (a.label_space == "source_label" and len(garments) != 1):
            dropped["multi_garment"] += 1; continue  # multi-garment images need Hana's per-garment crops
        cid, m = garments[0]
        if m["mapping_status"] not in allowed:
            dropped["status_filtered"] += 1; continue
        rows.append({"item_id": r["item_id"], "source_dataset": ds, "image_ref": (f"fashion_mnist://{r['source_split']}/{r['source_index']}" if ds == "Fashion-MNIST" else r["original_image_reference"]),
                     "source_category_id": cid, "source_label": m["source_label"],
                     "common_category": m["common_category"], "mapping_status": m["mapping_status"],
                     "label": m[a.label_space], "split": r["target_split"], "split_source": "m1_split_manifest",
                     "group": group_id_of(r)})
    out = pd.DataFrame(rows)
    if out.empty:
        sys.exit(f"No eligible items. Dropped: {dropped}")
    if a.derive_split:
        out["split"] = derive_split(out, a.seed)
        out["split_source"] = f"derived_seed{a.seed}_PENDING_TEAM_APPROVAL"
    for need in ("train", "validation"):
        if (out.split == need).sum() == 0:
            sys.exit(f"No '{need}' items. M1 splits put these datasets in validation only. "
                     f"Get a train split from Omar/Eyad or re-run with --derive-split after the team approves it.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print("dropped:", dropped)
    print(out.groupby(["split", "label"]).size().unstack(0).fillna(0).astype(int).to_string())
    print(f"written {len(out)} rows -> {a.out}")


if __name__ == "__main__":
    main()
