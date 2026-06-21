# Spotify Genre Classification

Machine learning project to classify Spotify tracks by genre using metadata and audio features from the Kaggle dataset:

https://www.kaggle.com/datasets/thedevastator/spotify-tracks-genre-dataset

## Problem

The goal is to predict the track genre (`track_genre`) from Spotify track features such as popularity, duration, danceability, energy, loudness, acousticness, instrumentalness, valence, tempo, and related metadata.

## Dataset

The notebooks download the dataset using `kagglehub`:

```python
import kagglehub

path = kagglehub.dataset_download("thedevastator/spotify-tracks-genre-dataset")
print("Path to dataset files:", path)
```

You can also download the dataset manually from Kaggle and place the CSV file inside `data/`.

Optional local path:

```text
data/spotify_genre_classification.csv
```

If this local file exists, the notebooks use it. Otherwise, they download the dataset with `kagglehub`.

## Project Plan

1. Exploratory Data Analysis.
2. Data cleaning and preprocessing.
3. Train/test split using a manageable subset if the dataset is larger than 20,000 samples.
4. Baseline model.
5. Train and compare scikit-learn classifiers.
6. Evaluate the final model with accuracy, F1-score, and confusion matrix.
7. Prepare a 10-slide presentation explaining decisions and results.
