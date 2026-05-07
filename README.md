# GraphAge-Seq

A unified sequence–graph integration framework for epigenetic age prediction using co-methylation graphs and DNA sequence context.

---

## Overview

This repository contains code for **GraphAge-Seq**, which jointly models co-methylation graph structure and site-specific DNA sequence context for epigenetic age prediction on Illumina HumanMethylation27 (GPL8490) blood methylation data.

**Key results (test set, n=756):**

| Model | MAE | MSE | R² |
|---|---|---|---|
| Horvath | 5.823 | 78.104 | 0.8865 |
| AltumAge | 3.710 | 47.675 | 0.9307 |
| DeepMAge | 3.222 | 29.574 | 0.9570 |
| ResnetAge | 5.552 | 58.481 | 0.9150 |
| GraphAge | 3.611 | 29.966 | 0.9564 |
| PNA-GNN (baseline) | 3.845 | 35.035 | 0.9491 |
| PNA-GNN + CNN | 3.271 | 26.512 | 0.9615 |
| **PNA-GNN + Stat. Features (Ours)** | **3.149** | **26.230** | **0.9619** |

---

## File Structure

```
graphage_seq/
│
│  ── Training ──
├── graphagewithseq_no_cnn.py          # Main model: PNA-GNN + Statistical Features (best)
├── graphagewithseq_cnn.py             # Ablation: PNA-GNN + CNN
│
│  ── Baseline training ──
├── train_deepmage.py                  # Train DeepMAge reimplementation
├── train_resnetage.py                 # Train ResnetAge reimplementation
│
│  ── Inference: healthy test set ──
├── inference_and_plot.py              # Main model inference + scatter plot + age group bar chart
├── inference_cnn.py                   # CNN model inference + scatter plot
├── inference_original_graphage.py     # PNA-GNN baseline inference + scatter plot
├── inference_baselines.py             # Horvath + AltumAge inference (requires tf env)
│
│  ── Inference: unhealthy cohorts ──
├── inference_unhealthy.py             # All GNN models + Horvath on unhealthy dataset
├── inference_unhealthy_altumage.py    # AltumAge on unhealthy dataset (requires tf env)
├── inference_unhealthy_mlp.py         # DeepMAge + ResnetAge on unhealthy dataset
├── inference_unhealthy_cnn.py         # PNA-GNN + CNN on unhealthy dataset
│
│  ── Interpretability ──
├── explainer_node_attr_no_cnn.py      # GNNExplainer: sequence feature importance (Fig. 4)
├── explainer_node_object_no_cnn.py    # GNNExplainer: CpG site importance (Appendix C)
│
│  ── Utilities ──
├── extract_dataset_stats.py           # Dataset statistics extraction
├── analyze_unhealthy_age_groups.py    # Age acceleration analysis (unhealthy cohorts)
│
│  ── Data directories ──
├── cpgsite-info/
│   └── GPL8490_HumanMethylation27_270596_v.1.2.csv   # GPL8490 manifest (download separately)
├── graph-age/
│   └── example_dependencies/
│       └── multi_platform_cpgs.pkl    # AltumAge CpG list (from GraphAge repo)
├── all-organs4/
│   └── all_organs/                    # Methylation data: 37 .pkl files (download separately)
├── unhealthy-dataset/
│   └── Unhealthy Normalized/          # Unhealthy cohort .pkl files (download separately)
│
│  ── Output directories (created at runtime) ──
├── checkpoint_seq_no_CNN/             # PNA-GNN + Stat. Features checkpoints + predictions
├── checkpoint_cnn/                    # PNA-GNN + CNN checkpoints + predictions
├── checkpoint/                        # PNA-GNN baseline checkpoint + predictions
└── baselines/                         # Baseline model weights + predictions
    ├── coefficients.csv               # Horvath 2013 CpG coefficients (download separately)
    ├── scaler.pkl                     # AltumAge RobustScaler (from AltumAge repo)
    └── AltumAge.h5                    # AltumAge trained model (from AltumAge repo)
```

---

## Data

### Methylation Data

The 37 blood methylation datasets (3,707 samples) are sourced from GEO and EMBL-EBI ArrayExpress. Each dataset is stored as a `.pkl` file under `all-organs4/all_organs/`. Each file is a pandas DataFrame with columns: CpG site IDs, `age`, `gender`, `dataset`, `tissue_type`.

**To obtain the data:**
1. Download datasets from [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/) and [EMBL-EBI ArrayExpress](https://www.ebi.ac.uk/arrayexpress/)
2. Filter for blood tissue samples on the GPL8490 platform
3. Intersect CpG sites with the AltumAge multi-platform CpG list (20,318 sites)
4. Save each dataset as a `.pkl` file

### Unhealthy Cohort Data

Three disease groups are used for age acceleration analysis. Download from GEO and place `.pkl` files under `unhealthy-dataset/Unhealthy Normalized/`:

| GEO Accession | Disease |
|---|---|
| GSE19711 | Ovarian Cancer |
| GSE41037 | Schizophrenia |
| GSE27044, GSE99624, E-GEOD-44763, GSE77241 | Osteoporosis |

### Baseline Model Files

Place the following in `baselines/`:

**Horvath (2013):** Download `coefficients.csv` from Horvath 2013 (Genome Biology) supplementary or the [online methylation age calculator](https://dnamage.genetics.ucla.edu/).

**AltumAge:** Clone the [AltumAge repository](https://github.com/rsinghlab/AltumAge) and copy `AltumAge.h5` and `scaler.pkl` to `baselines/`.

### GPL8490 Manifest

Download the GPL8490 annotation file from NCBI GEO:
```
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL8490
```
Place it at: `cpgsite-info/GPL8490_HumanMethylation27_270596_v.1.2.csv`

### AltumAge CpG List

Clone the [GraphAge repository](https://github.com/rsinghlab/graphage) and copy:
```
graphage/example_dependencies/multi_platform_cpgs.pkl
```
Place it at: `graph-age/example_dependencies/multi_platform_cpgs.pkl`

---

## Setup

### Environment

```bash
conda create -n graphage_env python=3.9
conda activate graphage_env

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install scikit-learn pandas numpy matplotlib seaborn scipy tqdm
pip install graphviz  # optional, for MRN subnetwork visualisation
```

Tested on: Python 3.9, PyTorch 2.x, PyG 2.x, CUDA 11.8 (H100 / L40S GPU).

### Path Configuration

All scripts read `BASE_DIR` from the environment variable `GRAPHAGE_BASE_DIR`. Set it before running:

```bash
export GRAPHAGE_BASE_DIR=/path/to/your/graphage_seq
```

Alternatively, edit the `BASE_DIR` line near the top of each script directly:

```python
BASE_DIR = '/path/to/your/graphage_seq'
```

---

## Running

### 1. Check dataset statistics

```bash
python extract_dataset_stats.py
```

Prints per-dataset and overall summary (sample counts, age distribution, gender breakdown).

### 2. Train the main model (PNA-GNN + Statistical Features)

```bash
python graphagewithseq_no_cnn.py
```

- Trains fold 2 of a 5-fold cross-validation (matching the reported result)
- Supports resume from checkpoint: just re-run the same command if interrupted
- Checkpoints saved to `checkpoint_seq_no_CNN/`
- Best model saved as `checkpoint_seq_no_CNN/cv_fold2_best_test_model.pth`

**Key hyperparameters** (in `__main__` block):
```python
NUM_EPOCHS   = 150
K_FOLDS      = 5
DESIRED_FOLD = 2        # 0-indexed; fold 2 = third fold
lr           = 8e-5
weight_decay = 6e-4
patience     = 4
factor       = 0.4
```

### 3. Train the CNN ablation

```bash
python graphagewithseq_cnn.py
```

Same interface as above. Uses one-hot encoded DNA sequences and a 3-layer CNN encoder instead of statistical features.

### 4. GNNExplainer — sequence feature importance

```bash
python explainer_node_attr_no_cnn.py
```

Requires the trained checkpoint at `checkpoint_seq_no_CNN/cv_fold2_best_test_model.pth`. Runs GNNExplainer on all 756 test samples and saves importance scores and plots to `checkpoint_seq_no_CNN/explainer/`.

**Outputs:**
- `seq_age_group_mean.npy` — [8, 8] mean importance per age group
- `seq_mean_overall.npy` — [8] overall mean importance
- `seq_all_per_sample.npy` — [756, 8] per-sample importance
- `all_ages.npy` — [756] chronological ages
- Publication-quality bar charts and temporal analysis plots

### 5. GNNExplainer — CpG site importance

```bash
python explainer_node_object_no_cnn.py
```

Runs node-level GNNExplainer. Saves per-CpG importance, temporal slope analysis, top-10 trending CpG sites, and optional MRN subnetwork visualisations (requires `graphviz`).

**Outputs:**
- `node_importance.npy` — [756, 20318] per-sample per-node importance
- `node_importance_slopes.npy` — [20318] linear regression slope per CpG
- `top10_upward_cpg.csv`, `top10_downward_cpg.csv`
- `top100_node_summary.csv`
- MRN subnetwork visualisations (Graphviz, optional)

### 6. Unhealthy cohort age acceleration analysis

```bash
python analyze_unhealthy_age_groups.py
```

Reads `baselines/unhealthy_predictions.csv` (generated separately by running baseline models on unhealthy cohorts) and computes mean age acceleration per disease per age group.

---

### 7. Inference on healthy test set (main model)

```bash
python inference_and_plot.py
```

Outputs to `checkpoint_seq_no_CNN/`: `test_predictions.csv`, `predicted_vs_true_age.png`, `mae_by_age_group.png`.

### 8. Inference on healthy test set (CNN ablation)

```bash
python inference_cnn.py
```

Outputs to `checkpoint_cnn/`: `test_predictions_cnn.csv`, `predicted_vs_true_age_cnn.png`.

### 9. Inference on healthy test set (PNA-GNN baseline)

```bash
python inference_original_graphage.py
```

Outputs to `checkpoint/`: `test_predictions_original.csv`, `predicted_vs_true_age_original.png`.

### 10. Horvath + AltumAge baseline inference

AltumAge requires TensorFlow. Run in the TF environment:

```bash
python inference_baselines.py
```

Requires `baselines/coefficients.csv`, `baselines/AltumAge.h5`, `baselines/scaler.pkl`. Outputs `baselines/baseline_results.csv`, `baselines/horvath_predictions.csv`, `baselines/altumage_predictions.csv`.

### 11. Train baseline models (DeepMAge and ResnetAge)

```bash
python train_deepmage.py
python train_resnetage.py
```

Both use the same 5-fold split (fold 2) as the main model. Outputs checkpoints and predictions to `baselines/`.

### 12. Unhealthy cohort age acceleration analysis

Run these in order (each merges its columns into `unhealthy_predictions.csv`):

```bash
python inference_unhealthy.py            # Horvath + PNA-GNN models
python inference_unhealthy_altumage.py   # AltumAge (tf env)
python inference_unhealthy_mlp.py        # DeepMAge + ResnetAge
python inference_unhealthy_cnn.py        # PNA-GNN + CNN
python analyze_unhealthy_age_groups.py   # Summarise by age group
```

Final outputs in `baselines/`: `unhealthy_predictions.csv`, `unhealthy_acceleration.csv`, `unhealthy_age_group_acceleration.csv`.

---

## Recommended Run Order

```
# Step 1 – Train main models
python graphagewithseq_no_cnn.py
python graphagewithseq_cnn.py

# Step 2 – Train baseline MLPs
python train_deepmage.py
python train_resnetage.py

# Step 3 – Baseline inference (requires tf env for AltumAge)
python inference_baselines.py

# Step 4 – Inference on healthy test set
python inference_and_plot.py
python inference_cnn.py
python inference_original_graphage.py

# Step 5 – Unhealthy cohort analysis
python inference_unhealthy.py
python inference_unhealthy_altumage.py   # tf env
python inference_unhealthy_mlp.py
python inference_unhealthy_cnn.py
python analyze_unhealthy_age_groups.py

# Step 6 – Interpretability
python explainer_node_attr_no_cnn.py
python explainer_node_object_no_cnn.py
```

---

## Reproducing the Main Result

The exact test MAE of 3.149 years corresponds to:

- **Script:** `graphagewithseq_no_cnn.py`
- **Checkpoint:** `checkpoint_seq_no_CNN/cv_fold2_best_test_model.pth`
- **Fold:** 2 (0-indexed), 5-fold CV, `random_state=42`
- **Random seed:** 0
- **Dataset split:** per-dataset 80/20 train/test, `random_state=42`
- **Graph construction:** built from training fold only (no leakage); three-tier edge filtering with thresholds 0.70 / 0.68 / 0.66 and distance threshold 1e5 bp

---

## Graph Construction Details

Edges are included if **any** of the following conditions hold:

1. **Primary:** |Pearson r| > 0.70 (any chromosome pair)
2. **Secondary:** same chromosome AND |r| > 0.68
3. **Tertiary:** same chromosome AND genomic distance < 1e5 bp AND |r| > 0.66

Self-loops are excluded. Edge attributes: [Pearson r, same-chromosome indicator, same-gene indicator].

---

## Model Architecture

```
DNA Sequence (122 bp)
       ↓
Statistical Feature Extraction (8-dim: GC content, CpG density, ...)
       ↓
Importance Gate (8 → 16 → 1, Sigmoid)     Seq Projection (8 → 8 → 2)
       ↓                                           ↓
Modulated Methylation (m × gate)    Positional Features (9-dim)    Seq Extra (2-dim)
       └──────────────────── Concat ──────────────────────────────────┘
                              12-dim fused node features
                                      ↓
                        PNA Convolution (4 aggregators, 3 scalers)
                                      ↓
                              MLP (20318 → 1024 → ... → 1)
                                      ↓
                              Predicted Age (years)
```

---

## Citation

Anonymous submission. Please do not cite prior to author notification.
