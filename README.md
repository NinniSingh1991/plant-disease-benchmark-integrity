# Plant disease benchmark integrity and integrity-aware adaptive fusion

Code, clean split lists, duplicate and conflict lists, results and test predictions for the article

> **Leakage-free and label-conflict-robust accuracy in plant disease recognition using integrity-aware adaptive fusion of fine-tuned convolutional and foundation models**
> Ninni Singh and Vinit Kumar Gunjan (manuscript under review).

## Contents

| Folder | Content |
|---|---|
| `audit/` | `audit.py`: verified duplicate audit (64-bit difference hash, exact multi-index search within Hamming radius 5, 32 × 32 thumbnail correlation ≥ 0.90, union-find groups, conflicting groups, copies between benchmarks). `conflict_categories.py`: conflict types and label agreement of shared copies. `montage.py`: visual review of candidate pairs. |
| `experiments/` | `clean_balance_runs.py` (audit, clean splits, cleaning × rebalancing grid, leakage test), `fusion_part.py` (frozen DINOv2-L / CLIP ViT-L features, hybrid head, fixed fusion), `adaptive_part.py` (validation-gated adaptive fusion, conflict-aware soft labels, temperature scaling), `build_nb.py` and the six Kaggle notebooks A–F that produced every result. |
| `analysis/` | Significance tests (paired bootstrap over test images, exact McNemar) recomputed from the saved predictions. |
| `splits/` | One CSV per trained benchmark (PlantDoc, Paddy Doctor, Cassava 2020, Rice Leaf, PlantWild): image, label, main split (test/train), membership of the cleaned training set, fusion split (test/train90/validation), duplicate group and conflict flag. |
| `duplicates/` | Verified pairs and duplicate groups of all eight audited benchmarks (incl. PlantVillage, Plant Pathology 2021, Corn/Maize), and the verified copies shared between benchmarks. |
| `results/` | Audit results, conflict categories, model cost, and the result tables and summaries of every experiment. |
| `predictions/` | Per-seed test (and validation) probabilities of every run, one compressed file per notebook, with the split indices. |
| `figures/` | Script that draws the figures of the article from the files above. |

## Image identifiers

Paths in the CSV files are relative to each benchmark's own release:
PlantDoc `train/<class>/<file>` or `test/<class>/<file>` (GitHub release); Paddy Doctor and Cassava 2020 `train_images/...` (Kaggle competitions);
PlantWild `images/<class>/<file>` (`plantwild.zip`, v1); Rice Leaf `agml_row_<n>` = row *n* of the AgML parquet files read in sorted file order.
The local audit (`duplicates/`) uses the identifiers recorded by `audit.py` (folder-relative paths, or `<parquet file>#<row group>#<row>`).

## Reproducing

1. Install the packages in `requirements.txt` (Python 3.10 or later).
2. Audit: download the benchmarks, set `PLANTVILLAGE_DIR` if needed, then `python audit/audit.py hash <names>` and `python audit/audit.py report`.
3. Experiments: import a notebook from `experiments/notebooks/` into Kaggle (GPU T4, Internet on; add the Paddy and Cassava competitions as inputs) and run it; seeds 42, 43 and 44.
4. Analysis: place the notebook outputs in `out_A` … `out_F` and run the scripts in `analysis/`.

## Data

The benchmarks are not redistributed here. Each remains under its own licence: PlantVillage, PlantDoc, Paddy Doctor (Kaggle competition), Cassava Leaf Disease Classification (Kaggle competition), Rice Disease Dataset and Corn or Maize Leaf Disease Dataset (Kaggle, through AgML), PlantWild, Plant Pathology 2021.

## Contact

Ninni Singh, Department of Computer Science and Engineering, Madan Mohan Malaviya University of Technology, Gorakhpur, India (nscse@mmmut.ac.in).
