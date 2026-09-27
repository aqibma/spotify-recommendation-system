"""
Generates a sample Spotify-style audio-features dataset for Task 3.

NOTE: This creates a realistic SAMPLE dataset (track name, artist, genre +
Spotify-style audio features: danceability, energy, valence, tempo,
acousticness, instrumentalness, liveness, speechiness, popularity).

For the actual EncoderX submission, you can replace this file's output
(spotify_tracks.csv) with a real Spotify dataset downloaded from Kaggle
(e.g. "Spotify Tracks Dataset" by maharshipandya) — the rest of the
pipeline (data_preparation.py, recommender.py, evaluate.py, app.py)
works with any CSV that has the same column names.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

genres = ["Pop", "Rock", "Hip-Hop", "Jazz", "Classical", "EDM", "R&B", "Country", "Indie", "Reggae"]
artists = [f"Artist {chr(65+i)}" for i in range(20)]  # Artist A ... Artist T

n_tracks = 300
rows = []
for i in range(n_tracks):
    genre = np.random.choice(genres)

    # Give each genre a rough "center" in feature space so genres cluster,
    # which makes content-based similarity and evaluation meaningful.
    genre_profile = {
        "Pop":       dict(dance=0.75, energy=0.70, valence=0.65, acoustic=0.15, instr=0.02, live=0.15, speech=0.06, tempo=120),
        "Rock":      dict(dance=0.45, energy=0.85, valence=0.55, acoustic=0.10, instr=0.10, live=0.25, speech=0.05, tempo=130),
        "Hip-Hop":   dict(dance=0.80, energy=0.65, valence=0.50, acoustic=0.08, instr=0.01, live=0.15, speech=0.25, tempo=95),
        "Jazz":      dict(dance=0.40, energy=0.35, valence=0.45, acoustic=0.70, instr=0.35, live=0.20, speech=0.05, tempo=110),
        "Classical": dict(dance=0.20, energy=0.20, valence=0.35, acoustic=0.92, instr=0.85, live=0.15, speech=0.03, tempo=90),
        "EDM":       dict(dance=0.85, energy=0.90, valence=0.60, acoustic=0.03, instr=0.20, live=0.30, speech=0.06, tempo=128),
        "R&B":       dict(dance=0.65, energy=0.50, valence=0.45, acoustic=0.25, instr=0.02, live=0.15, speech=0.08, tempo=100),
        "Country":   dict(dance=0.55, energy=0.55, valence=0.60, acoustic=0.45, instr=0.02, live=0.20, speech=0.04, tempo=115),
        "Indie":     dict(dance=0.55, energy=0.55, valence=0.50, acoustic=0.40, instr=0.10, live=0.18, speech=0.05, tempo=118),
        "Reggae":    dict(dance=0.70, energy=0.60, valence=0.70, acoustic=0.30, instr=0.05, live=0.20, speech=0.06, tempo=90),
    }[genre]

    def jitter(val, spread=0.08, lo=0.0, hi=1.0):
        return float(np.clip(np.random.normal(val, spread), lo, hi))

    rows.append({
        "track_id": f"T{i:04d}",
        "track_name": f"Track {i}",
        "artist": np.random.choice(artists),
        "genre": genre,
        "danceability": jitter(genre_profile["dance"]),
        "energy": jitter(genre_profile["energy"]),
        "valence": jitter(genre_profile["valence"]),
        "acousticness": jitter(genre_profile["acoustic"]),
        "instrumentalness": jitter(genre_profile["instr"]),
        "liveness": jitter(genre_profile["live"]),
        "speechiness": jitter(genre_profile["speech"]),
        "tempo": float(np.clip(np.random.normal(genre_profile["tempo"], 8), 60, 200)),
        "popularity": int(np.clip(np.random.normal(55, 20), 0, 100)),
    })

df = pd.DataFrame(rows)

# Inject a few realistic data-quality issues so data_preparation.py has
# something genuine to clean (Task Step 2: Data Preparation).
dup_rows = df.sample(5, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)  # duplicates
missing_idx = df.sample(10, random_state=2).index
df.loc[missing_idx, "tempo"] = np.nan  # missing values

df.to_csv("spotify_tracks.csv", index=False)
print(f"Wrote spotify_tracks.csv with {len(df)} rows (incl. injected duplicates/missing values).")
