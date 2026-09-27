"""
Step 2: Data Preparation (as required by the EncoderX Task 3 booklet)

- Handles missing values
- Removes duplicate / irrelevant records
- Prepares item (track) information
- Builds the scaled feature matrix used by the recommendation engine
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLUMNS = [
    "danceability", "energy", "valence", "acousticness",
    "instrumentalness", "liveness", "speechiness", "tempo",
]


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load the raw dataset and clean it."""
    df = pd.read_csv(csv_path)

    before = len(df)

    # 1. Remove duplicate records (by track_id, keep first occurrence)
    df = df.drop_duplicates(subset="track_id", keep="first")

    # 2. Handle missing values: numeric audio features -> fill with column median
    for col in FEATURE_COLUMNS + ["popularity"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # 3. Drop rows still missing essential identifying info (irrelevant records)
    df = df.dropna(subset=["track_id", "track_name", "artist", "genre"])

    df = df.reset_index(drop=True)
    after = len(df)
    print(f"Data preparation: {before} rows -> {after} rows "
          f"(removed {before - after} duplicate/invalid records).")
    return df


def build_feature_matrix(df: pd.DataFrame):
    """Scale audio features to [0,1] so no single feature dominates similarity."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURE_COLUMNS])
    feature_df = pd.DataFrame(scaled, columns=FEATURE_COLUMNS, index=df.index)
    return feature_df, scaler


if __name__ == "__main__":
    df = load_and_clean("../data/spotify_tracks.csv")
    features, _ = build_feature_matrix(df)
    print(df.head())
    print(features.head())
