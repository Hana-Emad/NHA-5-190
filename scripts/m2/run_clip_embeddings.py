"""Extract versioned CLIP embeddings for WARDIQ M2."""

from __future__ import annotations

import argparse
from pathlib import Path

from wardiq.representation.clip_embeddings import extract_clip_embeddings

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data/manifests/garment_manifest.csv"
OUTPUT_DIR = ROOT / "artifacts/m2/clip_embeddings_prototype"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CLIP image embeddings from the garment manifest."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    result = extract_clip_embeddings(
        manifest_path=MANIFEST_PATH,
        output_dir=OUTPUT_DIR,
        image_root=ROOT,
        batch_size=args.batch_size,
    )

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
