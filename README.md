# 🎵 Content-Based Music Recommendation System

**EncoderX Remote Internship Batch 02 — AI/ML Track**
**Task 3: Recommendation System Development (Week 03)**

## 1. Dataset

**Spotify Dataset** — a Spotify-style track dataset with audio features per track:
`danceability, energy, valence, acousticness, instrumentalness, liveness, speechiness, tempo, popularity`, plus `track_name`, `artist`, `genre`.

- File: `data/spotify_tracks.csv` (300 tracks after cleaning, 10 genres, 20 artists)
- Generated with `data/generate_sample_data.py` for demonstration purposes.
  **To use a real dataset:** download a Spotify audio-features dataset from Kaggle
  (e.g. "Spotify Tracks Dataset") and save it as `data/spotify_tracks.csv` with the
  same column names — the rest of the pipeline works unchanged.

## 2. Data Preparation

Handled in `src/data_preparation.py`:
- **Missing values** — numeric audio features filled with the column median
- **Duplicates** — removed by `track_id`
- **Irrelevant records** — rows missing essential identifying info dropped
- **Feature preparation** — audio features scaled to [0, 1] with `MinMaxScaler`, forming the item feature matrix used by the recommender

## 3. Recommendation Engine

**Approach: Content-Based Filtering** (`src/recommender.py`)

**Why this approach:** the dataset describes each track by its own audio
characteristics rather than by many users' listening histories, so
content-based filtering fits naturally. It also avoids the cold-start
problem of collaborative filtering and produces recommendations that are
easy to explain.

**How it works:**
1. Scale each track's audio features to [0, 1]
2. Compute cosine similarity between every pair of tracks
3. For a selected track, return the top-N most similar tracks (excluding itself)

## 4. Evaluation

Metric: **Precision@K** and **Recall@K**, using genre match as the relevance proxy (a recommended track is "relevant" if it shares the query track's genre).

| K  | Precision@K | Recall@K |
|----|-------------|----------|
| 3  | 0.733       | 0.077    |
| 5  | 0.684       | 0.119    |
| 10 | 0.646       | 0.226    |

**Interpretation:** Precision stays high (65–73%) across all K, meaning most recommended tracks genuinely share the query track's genre/style — the audio-feature similarity is capturing real genre structure. Recall rises with K, as expected, since a larger K can capture more of each genre's tracks.

Run it yourself: `python src/evaluate.py`

## 5. Demonstration Interface

A Streamlit app (`app.py`): pick a track from a dropdown, choose how many
recommendations to see, and view the results with a short explanation of
which audio features drove each match.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
spotify-recommender/
├── data/
│   ├── generate_sample_data.py
│   └── spotify_tracks.csv
├── src/
│   ├── data_preparation.py
│   ├── recommender.py
│   └── evaluate.py
├── app.py
├── requirements.txt
└── README.md
```

## Core Skills Demonstrated

Data Preprocessing, Content-Based Filtering, Cosine Similarity, Model Evaluation (Precision@K, Recall@K), Streamlit UI, Documentation.
