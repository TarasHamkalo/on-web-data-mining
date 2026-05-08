import numpy as np
import pandas as pd
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer


class ContentBasedRec:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

        self.movies_data = None
        self.ratings = None
        self.user_profiles = None

        self.rating_like_threshold = 3.5

    def train(self, movies_data: pd.DataFrame, ratings: pd.DataFrame):
        self.movies_data = movies_data.copy()
        self.ratings = ratings.copy()
        texts = (
            self.movies_data["title"].fillna("") +
            " ; " +
            self.movies_data["genres"]
            .fillna("")
            .str.split("|")
            .apply(lambda genres: " ".join(genres))
        ).tolist()

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False
        )

        self.movies_data["emb"] = list(embeddings)

        movie_emb_map = dict(
            zip(self.movies_data["movieId"], self.movies_data["emb"])
        )

        user_profiles = []

        for user_id in self.ratings["userId"].unique():
            user_ratings = self.ratings[self.ratings["userId"] == user_id]
            rated_movie_ids = user_ratings["movieId"].tolist()

            liked = user_ratings[user_ratings["rating"] >= self.rating_like_threshold]
            liked_movie_ids = liked["movieId"].tolist()

            liked_embs = [
                movie_emb_map[mid]
                for mid in liked_movie_ids
                if mid in movie_emb_map
            ]

            if len(liked_embs) > 0:
                user_profile_emb = np.mean(liked_embs, axis=0)
            else:
                user_profile_emb = np.zeros_like(embeddings[0])

            user_profiles.append({
                "userId": user_id,
                "rated_movie_ids": rated_movie_ids,
                "emb": user_profile_emb
            })

        self.user_profiles = pd.DataFrame(user_profiles)

    def cosine(self, A, B):
        denom = norm(A) * norm(B)

        if denom == 0:
            return 0.0

        return np.dot(A, B) / denom

    def recommend(self, user_id, top_k=10):
        profile_row = self.user_profiles[self.user_profiles["userId"] == user_id]

        if len(profile_row) == 0:
            return pd.DataFrame()

        profile = profile_row.iloc[0]

        rated_set = set(profile["rated_movie_ids"])
        profile_emb = profile["emb"]

        candidates = self.movies_data[
            ~self.movies_data["movieId"].isin(rated_set)
        ].copy()

        candidates["score"] = candidates["emb"].apply(
            lambda emb: self.cosine(profile_emb, emb)
        )

        top = candidates.sort_values(by="score", ascending=False)

        return top.head(top_k)[
            ["movieId", "title", "genres", "score"]
        ]
