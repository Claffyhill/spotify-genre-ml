# Spotify Genre Classification

> Multiclass supervised classification of Spotify tracks by genre using numerical audio features and scikit-learn.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Dataset](https://img.shields.io/badge/Kaggle-Spotify%20Tracks%20Genre-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/thedevastator/spotify-tracks-genre-dataset)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Requirements](#requirements)
- [Setup & Usage](#setup--usage)
- [Notebooks](#notebooks)

---

## Overview

This project frames music genre tagging as a **multiclass supervised classification problem**: given the numerical audio features of a Spotify track, predict which of **114 genres** it belongs to.

The task is considerably more challenging than binary classification — with 114 balanced classes, a random classifier achieves approximately **0.88% accuracy**. The project benchmarks four scikit-learn classifiers against this baseline, evaluates them on a held-out test set, and analyses per-genre performance through precision, recall, F1-score, and a confusion matrix.

**Key result:** Random Forest achieved **29.18% accuracy** on the test set — the best-performing model, outperforming all other evaluated classifiers.

---

## Dataset

**Source:** [Spotify Tracks Genre Dataset](https://www.kaggle.com/datasets/thedevastator/spotify-tracks-genre-dataset) (Kaggle)

| Property | Value |
|---|---|
| Total tracks | 114,000 |
| Columns | 21 |
| Genre classes | 114 (balanced, ~1,000 tracks per genre) |
| Missing values | None |
| Duplicate rows | None |

### Input features (X) — 14 numerical audio variables

| Feature | Description |
|---|---|
| `popularity` | Track popularity score (0–100) |
| `duration_ms` | Track duration in milliseconds |
| `danceability` | Suitability for dancing (0.0–1.0) |
| `energy` | Perceptual intensity and activity (0.0–1.0) |
| `key` | Estimated overall key (0–11) |
| `loudness` | Overall loudness in dB |
| `mode` | Modality: major (1) or minor (0) |
| `speechiness` | Presence of spoken words (0.0–1.0) |
| `acousticness` | Likelihood of being acoustic (0.0–1.0) |
| `instrumentalness` | Likelihood of containing no vocals (0.0–1.0) |
| `liveness` | Presence of a live audience (0.0–1.0) |
| `valence` | Musical positiveness (0.0–1.0) |
| `tempo` | Estimated tempo in BPM |
| `time_signature` | Estimated beats per bar |

**Target variable (Y):** `track_genre` — one of 114 mutually exclusive genre labels.

Text columns (`track_name`, `artists`, `album_name`, `track_id`) were **excluded** from model input, as they do not represent audio signal and would not generalise to unseen tracks.

### Downloading the dataset

The notebooks support two methods:

**Option 1 — automatic download via `kagglehub`** (default):

```python
import kagglehub

path = kagglehub.dataset_download("thedevastator/spotify-tracks-genre-dataset")
print("Path to dataset files:", path)
```

**Option 2 — manual download:**

Download the CSV from Kaggle and place it at:

```
data/spotify_genre_classification.csv
```

If this local file exists, the notebooks use it directly; otherwise they fall back to `kagglehub`.

---

## Project Structure

```
spotify-genre-classification/
│
├── data/                          # Local dataset (optional, not tracked by git)
│   └── spotify_genre_classification.csv
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   └── 02_modeling.ipynb          # Preprocessing, training, and evaluation
│
├── presentation/
│   └── Spotify_Genre_Classification.pptx   # 10-slide project presentation
│
├── requirements.txt
└── README.md
```

---

## Methodology

### 1 · Exploratory Data Analysis

- Verified dataset integrity: shape, types, missing values, and duplicate rows
- Confirmed balanced class distribution (~1,000 tracks per genre)
- Identified the 14 numerical features to use as input
- Examined inter-genre feature separability through distribution plots

**Key finding:** Some genres exhibit visually distinguishable feature profiles (e.g., acoustic vs. electronic tracks), but a substantial number of genre pairs overlap considerably in feature space. This motivated the comparison of both linear and non-linear classifiers.

### 2 · Preprocessing Pipeline

| Step | Detail |
|---|---|
| **Stratified subsampling** | 20,000 tracks drawn from 114,000, preserving genre proportions |
| **Train / test split** | 75% train (15,000) / 25% test (5,000), stratified by genre |
| **Feature scaling** | `StandardScaler` applied to Logistic Regression and KNN — both depend on feature magnitude (gradient optimisation and Euclidean distance respectively). Random Forest is scale-invariant and was trained on unscaled data. |

All steps use a fixed `random_state` to ensure reproducibility.

### 3 · Feature Selection

All 14 numerical audio features were used directly — no dimensionality reduction was applied. With only 14 input variables the feature space is already low-dimensional, and Random Forest is additionally robust to less-informative features through its internal split criterion.

### 4 · Model Selection

Four classifiers were evaluated, chosen to cover different learning paradigms:

| Model | Paradigm | Why selected |
|---|---|---|
| `DummyClassifier` | Baseline | Establishes the performance lower bound (~0.88% for 114 balanced classes) |
| `LogisticRegression` | Linear | Tests whether a linear decision boundary is sufficient to separate genres |
| `KNeighborsClassifier (k=5)` | Distance-based | Captures local similarity in the audio feature space |
| `RandomForestClassifier (n=100)` | Ensemble | Captures non-linear relationships; expected to outperform linear models given the overlapping class structure found during EDA |

---

## Results

### Model comparison (test set accuracy)

| Model | Accuracy |
|---|---|
| Dummy Classifier (baseline) | 0.88% |
| K-Nearest Neighbors (k=5) | 14.42% |
| Logistic Regression | 18.80% |
| **Random Forest (100 trees)** | **29.18%** ✓ |

Random Forest achieved the highest accuracy (29.18%), outperforming all other evaluated classifiers.

### Per-genre performance (selected examples)

| Genre | Precision | Recall | F1-score | Notes |
|---|---|---|---|---|
| comedy | 0.89 | 0.75 | **0.81** | Acoustically distinctive |
| grindcore | 0.75 | 0.86 | **0.80** | Acoustically distinctive |
| honky-tonk | 0.78 | 0.70 | **0.74** | Acoustically distinctive |
| emo | 0.00 | 0.00 | **0.00** | Overlaps with adjacent styles |
| electronic | 0.06 | 0.02 | **0.03** | Overlaps with adjacent styles |

**Macro-average:** Precision 0.28 · Recall 0.29 · F1 0.28  
**Weighted average:** F1 ≈ 0.28

### Interpretation

Although 29.18% may appear low in absolute terms, the task involves distinguishing among **114 balanced classes** — a considerably more demanding setting than conventional binary classification. All trained classifiers substantially outperform random prediction.

Classification errors are concentrated within acoustically similar genre families (e.g., house / deep-house / techno), rather than randomly distributed across all classes. This indicates that the performance ceiling is partly determined by the inherent acoustic similarity among subgenres, and is not solely a consequence of model limitations.

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
kagglehub
jupyter
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup & Usage

**1. Clone the repository**

```bash
git https://github.com/Claffyhill/spotify-genre-ml.git
cd spotify-genre-classification
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the notebooks in order**

```bash
jupyter notebook
```

Open and run:

1. `notebooks/01_eda.ipynb` — Exploratory Data Analysis
2. `notebooks/02_modeling.ipynb` — Preprocessing, model training, and evaluation

The dataset is downloaded automatically on first run. Alternatively, place the CSV at `data/spotify_genre_classification.csv` before running.

---

## Notebooks

| Notebook | Description |
|---|---|
| [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Data integrity checks, class distribution, feature distributions across genres, separability analysis |
| [`02_modeling.ipynb`](notebooks/02_modeling.ipynb) | Stratified sampling, train/test split, feature scaling, model training, accuracy comparison, classification report, confusion matrix |

---

*University Machine Learning Project · [Machine Learning] · [2025/2026]*