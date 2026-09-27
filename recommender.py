"""
Step 3: Recommendation Engine Development

Approach chosen: CONTENT-BASED FILTERING.

Why: The Spotify dataset describes each track by its own audio features
(danceability, energy, valence, acousticness, instrumentalness, liveness,
speechiness, tempo) rather than by many users' listening histories.
Content-based filtering uses exactly this kind of item information, so it
fits the dataset naturally, works well with a small number of users
(no cold-start problem), and produces recommendations that are easy to
explain ("this track was recommended because it has similar energy and
danceability to the one you picked").

How it generates recommendations:
1. Every track's audio features are scaled to [0, 1].
2. Cosine similarity is computed between every pair of tracks based on
   these scaled features.
3. Given a track the user likes, the engine returns the N most similar
   tracks (highest cosine similarity), excluding the track itself.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from data_preparation import load_and_clean, build_feature_matrix, FEATURE_COLUMNS


class ContentBasedRecommender:
    def __init__(self, csv_path: str):
        self.df = load_and_clean(csv_path)
        self.features, self.scaler = build_feature_matrix(self.df)
        self.similarity_matrix = cosine_similarity(self.features)

    def _track_index(self, track_name: str) -> int:
        matches = self.df.index[self.df["track_name"] == track_name].tolist()
        if not matches:
            raise ValueError(f"Track '{track_name}' not found in dataset.")
        return matches[0]

    def recommend(self, track_name: str, top_n: int = 5) -> pd.DataFrame:
        """Return the top_n tracks most similar to the given track."""
        idx = self._track_index(track_name)
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = [s for s in scores if s[0] != idx]
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_n]

        result = self.df.iloc[[i for i, _ in top]][
            ["track_name", "artist", "genre"]
        ].copy()
        result["similarity_score"] = [round(s, 3) for _, s in top]
        return result.reset_index(drop=True)

    def explain(self, track_name: str, recommended_track_name: str) -> str:
        """Return a short explanation of which audio features drove the match."""
        i = self._track_index(track_name)
        j = self._track_index(recommended_track_name)
        diffs = (self.features.iloc[i] - self.features.iloc[j]).abs().sort_values()
        closest_features = diffs.index[:3].tolist()
        return f"Most similar in: {', '.join(closest_features)}"

    def track_names(self):
        return sorted(self.df["track_name"].unique().tolist())


if __name__ == "__main__":
    rec = ContentBasedRecommender("../data/spotify_tracks.csv")
    sample_track = rec.df.iloc[0]["track_name"]
    print(f"Recommendations for '{sample_track}':")
    print(rec.recommend(sample_track, top_n=5))
