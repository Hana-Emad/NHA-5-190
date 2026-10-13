# M2 CLIP Image Embeddings

## Purpose and status

This document describes WARDIQ's CLIP image-embedding extraction component, implemented in `src/wardiq/representation/clip_embeddings.py`.

**Status: prototype implemented and successfully executed; full M2 acceptance is not established.**

The component generates normalized CLIP image embeddings and an inventory mapping each item to its embedding row. The successful prototype run verifies that the extraction path works on the current sample manifest. It does not establish full accepted-dataset coverage, a completed model comparison, or final M2 representation selection.

## Implementation

The extraction module provides:

- `load_usable_manifest()` — loads and validates the input manifest, filters usable derivatives, checks item identifiers, and resolves image paths.
- `normalize_embeddings()` — validates and normalizes embedding vectors.
- `validate_embedding_artifact()` — checks expected embedding artifact properties.
- `extract_clip_embeddings()` — extracts CLIP image embeddings and writes prototype artifacts.

The model identifier is `openai/clip-vit-base-patch32`.

| Property | Value |
|---|---|
| Model revision | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| Embedding dimension | 512 |
| Output dtype | `float32` |
| Normalization | L2-normalized |
| Processor image size | 224 |
| Execution device | CPU |

The model revision is pinned in the implementation. Metadata records the model revision, input manifest path and hash, Git commit, item count, and output filenames to support reproducibility.

## Environment and reproduction

Run the commands from the repository root using a Python environment with the required ML dependencies installed.

The project dependency declarations should be used to configure the environment. The successful prototype run used Python 3.14, PyTorch 2.14.1+cpu, Transformers 5.19.0, and Pillow 12.3.0.

The extraction entry point is:

`scripts/m2/run_clip_embeddings.py`

The following PowerShell commands reproduce the prototype invocation in the environment used for verification:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path "src"

& "C:\Users\omara\AppData\Local\Programs\Python\Python314\python.exe" scripts\m2\run_clip_embeddings.py --batch-size 4
```

This invocation uses the repository's `data/manifests/garment_manifest.csv` input and writes to `artifacts/m2/clip_embeddings_prototype/`.

The absolute Python path above is specific to the verification machine. Other environments should use their configured Python executable.

## Prototype artifacts

The verified output directory is:

`artifacts/m2/clip_embeddings_prototype/`

| File | Purpose |
|---|---|
| `clip_embeddings.npy` | Numerical embedding matrix |
| `clip_inventory.csv` | Item metadata and embedding-row mapping |
| `metadata.json` | Model and extraction metadata |

The generated output directory is local and is not intended to be committed to Git.

## Verified prototype results

The successful extraction reported:

| Metric | Result |
|---|---:|
| Items processed | 23 |
| Embedding shape | `(23, 512)` |
| Embedding dtype | `float32` |
| Execution device | CPU |

The prototype item population is:

| Dataset | Split | Items |
|---|---|---:|
| Fashion-MNIST | Train smoke test | 5 |
| Fashionpedia | Validation | 13 |
| Polyvore Outfits | Validation | 5 |
| **Total** | | **23** |

The metadata identifies the source manifest as `data/manifests/garment_manifest.csv` and records its SHA-256:

`b6e1899566fb7a1b302c1ceb6dbf001e4a09b24f9e8ea020de4aacea878c4466`

It also records the model revision, embedding version, normalization, output filenames, and Git commit associated with the run.

The output files were confirmed to exist locally. The extraction completed without a reported error.

## Tests

The following test command was run:

```powershell
& "C:\Users\omara\AppData\Local\Programs\Python\Python314\python.exe" -m pytest tests\test_clip_embeddings.py -q
```

**Result: 9 passed.**

These tests cover the component's tested validation, normalization, manifest-handling, and path-handling behavior. Passing unit tests do not by themselves prove full-scale extraction correctness or accepted-population coverage.

## Limitations and reproducibility requirements

The 23-item output is a prototype, not evidence that the entire accepted M1 population has been embedded.

Before claiming full-scale reproducible extraction, the team must verify:

1. The canonical input manifest and approved item population.
2. The intended inclusion and exclusion rules for each source dataset.
3. The model revision, preprocessing configuration, and environment requirements.
4. The expected item count, embedding shape, and inventory schema.
5. The correspondence between every inventory row and its embedding row.
6. The dataset and split provenance of every included item.
7. That the evaluation population and test-set isolation rules remain intact.

The Fashion-MNIST rows are a train-split smoke test; the Fashionpedia and Polyvore rows are validation examples. Their presence in one prototype artifact does not authorize training or evaluation across those sources without an approved split and evaluation protocol.

Generated embeddings and cache files should remain outside version control unless the team explicitly decides otherwise.

## Remaining M2 work

The following items are not established as complete by this prototype:

- Full accepted-population extraction and inventory reconciliation.
- Confirmation of the final approved evaluation population and split policy.
- Fair comparison of candidate representations using agreed metrics and eligible examples.
- End-to-end validation of saved artifacts against the accepted item inventory.
- Final representation selection and integration.
- Agreement on the cache and schema contract with the relevant M2 owners.

The successful run establishes that the current prototype can generate and save CLIP embeddings for 23 items. It should not be treated as proof that all M2 acceptance criteria have been met.

## Retrieval diagnostic

A small nearest-neighbor diagnostic was run on the 18 available real-garment
embeddings, returning three eligible neighbors per query. The results and
limitations are documented in
[`clip_retrieval_diagnostic.md`](clip_retrieval_diagnostic.md).

The diagnostic verifies that cosine-similarity retrieval runs, but it does
not validate retrieval quality or establish that CLIP is the final
representation choice.