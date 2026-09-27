"""
Step 5: Demonstration Interface

Simple Streamlit app: user selects a track, gets top-N similar tracks
with a short explanation of why each was recommended.

Run with:
    streamlit run app.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from recommender import ContentBasedRecommender

st.set_page_config(page_title="Spotify-style Recommender", page_icon="🎵")

st.title("🎵 Content-Based Music Recommender")
st.caption("EncoderX Task 3 — Recommendation System Development")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "spotify_tracks.csv")


@st.cache_resource
def get_recommender():
    return ContentBasedRecommender(DATA_PATH)


rec = get_recommender()

track = st.selectbox("Pick a track you like:", rec.track_names())
top_n = st.slider("How many recommendations?", min_value=3, max_value=15, value=5)

if st.button("Get Recommendations"):
    results = rec.recommend(track, top_n=top_n)
    st.subheader(f"Because you liked '{track}':")
    for _, row in results.iterrows():
        why = rec.explain(track, row["track_name"])
        st.markdown(
            f"**{row['track_name']}** — {row['artist']} ({row['genre']})  \n"
            f"Similarity: `{row['similarity_score']}`  •  {why}"
        )
