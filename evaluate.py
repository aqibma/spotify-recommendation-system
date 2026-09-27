"""
Step 4: Evaluation

Metrics used: Precision@K and Recall@K.

Ground truth / relevance definition: since this dataset has no explicit
user ratings, a recommended track is treated as "relevant" to a query
track if it belongs to the SAME GENRE as the query track. This is a
standard proxy used for content-based recommender evaluation when
explicit relevance labels aren't available.
"""

import numpy as np
from recommender import ContentBasedRecommender


def precision_recall_at_k(recommender: ContentBasedRecommender, k: int = 5, n_queries: int = 50):
    df = recommender.df
    sample = df.sample(n=min(n_queries, len(df)), random_state=7)

    precisions, recalls = [], []

    for _, row in sample.iterrows():
        query_track = row["track_name"]
        query_genre = row["genre"]

        recs = recommender.recommend(query_track, top_n=k)
        relevant_in_recs = (recs["genre"] == query_genre).sum()

        total_relevant_in_dataset = (df["genre"] == query_genre).sum() - 1  # exclude query itself

        precision = relevant_in_recs / k
        recall = relevant_in_recs / total_relevant_in_dataset if total_relevant_in_dataset > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    return {
        "k": k,
        "n_queries": len(sample),
        "precision_at_k": round(float(np.mean(precisions)), 3),
        "recall_at_k": round(float(np.mean(recalls)), 3),
    }


if __name__ == "__main__":
    rec = ContentBasedRecommender("../data/spotify_tracks.csv")
    for k in [3, 5, 10]:
        result = precision_recall_at_k(rec, k=k, n_queries=50)
        print(result)
