import numpy as np
import pandas as pd
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer


class ContentBasedRec:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.movies_data = None
        self.ratings = None

        self.rating_like_threshold = 2.5

    def train(self, movies_data: pd.DataFrame, ratings: pd.DataFrame):
        self.movies_data = movies_data.copy()
        self.ratings = ratings.copy()

        texts = (
            self.movies_data["title"] +
            " ; " +
            self.movies_data["genres"]
        ).tolist()

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False
        )

        self.movies_data["emb"] = list(embeddings)

        user_ids = list(ratings["userId"].unique())
        user_profiles = []
        for user_id in user_ids:
            user_ratings = self.ratings[self.ratings["userId"] == user_id]
            liked_movie_ids = []
            rated_movie_ids = []

            user_profile_emb = np.zeros_like(embeddings[0])

            for user_rating in user_ratings.iterrows():
                movie_id = user_rating["movieId"].getValue()
                rated_movie_ids.append(movie_id)

                if user_rating["rating"].getValue() > self.rating_like_threshold:
                    liked_movie_ids.append(movie_id)

                    liked_movie_mask = [self.movies_data["movie_id"] ==  movie_id]

                    user_profile_emb += self.movies_data["emb"][liked_movie_mask]

            if len(liked_movie_ids) > 0:
                # TODO: many users gonna have 0?
                #  maybe we should account also those that wer not liked and at least recommend anything else
                user_profile_emb /= len(liked_movie_ids)

            user_profiles.append({
                "userId": user_id,
                "rated_movie_ids": rated_movie_ids,
                "emb": user_profile_emb
            })


        self.user_profiles = pd.DataFrame(user_profiles)

    def cosine(self, A, B):
        return np.dot(A, B) / (norm(A) * norm(B))

    def recommend(self, user_id, top_k=10):
        profile = self.user_profiles[self.user_profiles["userId"] == user_id]

        not_rated_mask = self.movies_data["movieId"].apply(
            lambda x: x not in profile["rated_movie_ids"], axis=1
        )

        not_rated_movie_ids = self.movies_data[not_rated_mask == True]["movieId"].copy()

        not_rated_movie_ids["score"] = not_rated_movie_ids.apply(
           lambda x: self.cosine(profile["emb"], self.movies_data[x["movieId"].getValue()]["emb"]),
           axis=1,
        )

        top = not_rated_movie_ids.sort_values(by="score", ascending=False)
        return top[:top_k]