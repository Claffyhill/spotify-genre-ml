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
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)

---

## Overview

This project addresses music genre classification as a **multiclass supervised classification** problem. Given the numerical audio features of a Spotify track, the objective is to predict its genre from a set of 114 balanced classes.

The dataset consists of 114,000 tracks sourced from the [Spotify Tracks Genre Dataset](https://www.kaggle.com/datasets/thedevastator/spotify-tracks-genre-dataset) (Kaggle). Four scikit-learn classifiers are trained and compared on a stratified held-out test set. Performance is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

**Best result:** Random Forest — **29.18% accuracy** on the test set.

---

## Dataset

**Source:** [Spotify Tracks Genre Dataset](https://www.kaggle.com/datasets/thedevastator/spotify-tracks-genre-dataset) — Kaggle

| Property | Value |
|---|---|
| Total tracks | 114,000 |
| Total columns | 21 |
| Target classes | 114 genres, balanced (~1,000 tracks per class) |
| Missing values | None |
| Duplicate rows | None |

### Input features — 14 numerical audio variables

| Feature | Description |
|---|---|
| `popularity` | Track popularity score (0–100) |
| `duration_ms` | Duration in milliseconds |
| `danceability` | Suitability for dancing (0.0–1.0) |
| `energy` | Perceptual intensity and activity (0.0–1.0) |
| `key` | Estimated musical key (0–11) |
| `loudness` | Overall loudness in dB |
| `mode` | Modality: major (1) or minor (0) |
| `speechiness` | Presence of spoken words (0.0–1.0) |
| `acousticness` | Likelihood of being acoustic (0.0–1.0) |
| `instrumentalness` | Likelihood of containing no vocals (0.0–1.0) |
| `liveness` | Presence of a live audience (0.0–1.0) |
| `valence` | Musical positiveness (0.0–1.0) |
| `tempo` | Estimated tempo in BPM |
| `time_signature` | Estimated beats per bar |

**Target variable:** `track_genre` — one of 114 mutually exclusive genre labels.

Text columns (`track_name`, `artists`, `album_name`, `track_id`) were excluded from model input, as they do not represent audio signal and would not generalise to unseen tracks.

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

If this local file is present, the notebooks use it directly; otherwise they fall back to `kagglehub`.

---

## Repository Structure

```
spotify-genre-classification/
│
├── data/                        # Local dataset directory (not tracked by git)
│   └── spotify_genre_classification.csv
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   └── 02_modeling.ipynb        # Preprocessing, model training, and evaluation
│
├── presentation/
│   └── Spotify_Genre_Classification.pptx  # 10-slide project presentation
│
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Exploratory Data Analysis

The EDA notebook verifies dataset integrity (shape, data types, missing values, duplicate rows) and examines the distribution of all 14 numerical features across the top genres. The class distribution was confirmed to be balanced, with approximately 1,000 tracks per genre.

A key observation from the EDA is that while some genres exhibit visually distinguishable feature profiles — for example, acoustic tracks tend toward high acousticness and low energy, whereas electronic genres show the inverse — a substantial proportion of genre pairs overlap considerably in feature space. This finding motivated the evaluation of both linear and non-linear classifiers.

### 2. Stratified Subsampling

To keep training computationally manageable, a stratified sample of **20,000 tracks** was drawn from the full 114,000-track dataset. Stratified sampling preserves the original genre proportions, ensuring no class is over- or under-represented in the working subset.

### 3. Train / Test Split

The sample was partitioned into a **training set** (75%, 15,000 tracks) and a **test set** (25%, 5,000 tracks), using stratified splitting to maintain class balance in both partitions. A fixed `random_state` was applied throughout to ensure reproducibility.

### 4. Feature Scaling

`StandardScaler` was applied to the features used by Logistic Regression and K-Nearest Neighbors:

- **Logistic Regression** relies on gradient-based optimisation, which is sensitive to differences in feature magnitude.
- **KNN** computes Euclidean distances, which are distorted when features operate on different scales.

Random Forest is scale-invariant — its splits are based on rank thresholds rather than absolute values — and was therefore trained on unscaled data.

### 5. Model Training and Evaluation

All four classifiers were trained on the same training set and evaluated on the same held-out test set. Evaluation metrics include overall accuracy, per-class precision, recall and F1-score, and a confusion matrix computed over the 10 most frequent genres.

---

## Models

Four classifiers were selected to cover different learning paradigms:

| Model | Paradigm | Rationale |
|---|---|---|
| `DummyClassifier` | Baseline | Predicts the most frequent class regardless of input. Establishes the performance lower bound for the task (~0.88% for 114 balanced classes). |
| `LogisticRegression` | Linear | Simple linear classifier. Included to assess whether a linear decision boundary is sufficient to separate genres in the 14-dimensional feature space. |
| `KNeighborsClassifier (k=5)` | Distance-based | Non-parametric classifier that assigns labels based on nearest neighbours. Captures local similarity in the audio feature space. |
| `RandomForestClassifier (n_estimators=100)` | Ensemble | Ensemble of decision trees capable of learning non-linear relationships between features. Expected to outperform linear models given the complex, overlapping class structure observed during EDA. |

---

## Results

### Model comparison — test set accuracy

| Model | Accuracy |
|---|---|
| Dummy Classifier (baseline) | 0.88% |
| K-Nearest Neighbors (k = 5) | 14.42% |
| Logistic Regression | 18.80% |
| **Random Forest (100 trees)** | **29.18%** ✓ |


### Per-class performance — selected examples

| Genre | Precision | Recall | F1-score |
|---|---|---|---|
| comedy | 0.89 | 0.75 | **0.81** |
| grindcore | 0.75 | 0.86 | **0.80** |
| honky-tonk | 0.78 | 0.70 | **0.74** |
| emo | 0.00 | 0.00 | **0.00** |
| electronic | 0.06 | 0.02 | **0.03** |

**Macro-average:** Precision 0.28 · Recall 0.29 · F1 0.28  
**Weighted average:** F1 ≈ 0.28

### Interpretation

Random Forest achieved the highest accuracy (29.18%), outperforming all other evaluated classifiers. Its advantage over Logistic Regression (18.80%) and KNN (14.42%) is consistent with the overlapping class structure identified during EDA: the relationship between audio features and genre labels is non-linear, favouring an ensemble approach over a linear decision boundary.

Although 29.18% may appear modest, the task involves distinguishing among **114 balanced classes** — a considerably more demanding setting than binary classification, where chance-level performance is approximately 0.88%. All trained models substantially outperform random prediction.

Per-class analysis reveals that genres with acoustically distinctive profiles — such as comedy, grindcore, and honky-tonk — are classified with substantially higher F1-scores. Genres that overlap with adjacent styles in feature space (e.g., emo, electronic) produce lower scores. The confusion matrix confirms that misclassifications are concentrated within acoustically similar genre families rather than randomly distributed, suggesting that the primary performance ceiling is the inherent similarity among subgenres.

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/[username]/spotify-genre-classification.git
cd spotify-genre-classification
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**Dependencies:**

```
pandas
numpy
scikit-learn
matplotlib
seaborn
kagglehub
jupyter
```

---

## Usage

Launch Jupyter and execute the notebooks **in order**:

```bash
jupyter notebook
```

| Step | Notebook | Description |
|---|---|---|
| 1 | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Data integrity checks, class distribution, feature distributions, separability analysis |
| 2 | [`notebooks/02_modeling.ipynb`](notebooks/02_modeling.ipynb) | Stratified subsampling, train/test split, feature scaling, model training, accuracy comparison, classification report, confusion matrix |

The dataset is downloaded automatically on first run via `kagglehub`. To use a local copy instead, place the CSV at `data/spotify_genre_classification.csv` before running the notebooks.

---

## Future Work

The following extensions could be explored in subsequent iterations of this project:

- **Hyperparameter tuning:** systematic search over Random Forest parameters (`n_estimators`, `max_depth`, `min_samples_leaf`) to improve classification performance.
- **Feature engineering:** construction of interaction features or the incorporation of aggregated metadata to enrich the audio-based representation.
- **Additional ensemble methods:** evaluation of Gradient Boosting variants on the same train/test partition.
- **Neural network classifiers:** exploration of shallow feed-forward architectures to assess whether increased model capacity improves performance on the harder-to-classify genres.

---

*University of Cagliari · Machine Learning · Academic Year 2025/2026*