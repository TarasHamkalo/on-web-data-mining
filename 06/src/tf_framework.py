import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

from recommender import Recommender


# Model definition
class RetrievalModel(tfrs.models.Model):
    def __init__(self, user_model, movie_model, task):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        self.task = task

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["userId"])
        movie_embeddings = self.movie_model(features["movieId"])

        return self.task(user_embeddings, movie_embeddings)

class TfFrameworkRec(Recommender):

    def __init__(
        self,
        rating_like_threshold=3.5,
        embedding_dim=32,
        epochs=5,
        batch_size=1024,
        learning_rate=0.05,
        verbose=0,
    ):
        self.rating_like_threshold = rating_like_threshold

        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.movies_data = None
        self.ratings = None

        self.user_ids = None
        self.movie_ids = None

        self.model = None
        self.user_model = None
        self.movie_model = None

        self.user_lookup = None
        self.movie_lookup = None

        self.movie_embeddings = None

        # cache: userId -> set(movieIds already rated)
        self.user_rated_movies = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        movies_data: pd.DataFrame,
        ratings: pd.DataFrame
    ):
        self.movies_data = movies_data.copy()
        self.ratings = ratings.copy()
        self.ratings = ratings[ratings["rating"] > self.rating_like_threshold]


        # Convert IDs to strings for TensorFlow
        self.movies_data["movieIdStr"] = (
            self.movies_data["movieId"].astype(str)
        )
        self.ratings["userIdStr"] = (
            self.ratings["userId"].astype(str)
        )
        self.ratings["movieIdStr"] = (
            self.ratings["movieId"].astype(str)
        )

        # Unique IDs
        self.user_ids = (
            self.ratings["userIdStr"]
            .drop_duplicates()
            .tolist()
        )

        self.movie_ids = (
            self.movies_data["movieIdStr"]
            .drop_duplicates()
            .tolist()
        )

        # Embedding models
        self.user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=self.user_ids, mask_token=None),
            tf.keras.layers.Embedding(
                input_dim=len(self.user_ids) + 1,
                output_dim=self.embedding_dim
            )
        ])

        self.movie_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=self.movie_ids, mask_token=None),
            tf.keras.layers.Embedding(
                input_dim=len(self.movie_ids) + 1,
                output_dim=self.embedding_dim
            )
        ])

        candidate_dataset = tf.data.Dataset.from_tensor_slices(
            self.movie_ids
        ).batch(1024).map(self.movie_model)

        # Retrieval task
        task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_dataset
            )
        )

        self.model = RetrievalModel(self.user_model, self.movie_model, task)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adagrad(
                learning_rate=self.learning_rate
            )
        )

        # Training dataset
        train_ds = tf.data.Dataset.from_tensor_slices({
            "userId": self.ratings["userIdStr"].values,
            "movieId": self.ratings["movieIdStr"].values,
        })

        train_ds = (
            train_ds
            .shuffle(len(self.ratings), seed=42)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE) # parallelism level
        )

        self.model.fit(train_ds, epochs=self.epochs, verbose=self.verbose)

        # precompute movie embeddings
        movie_emb = self.movie_model(tf.constant(self.movie_ids)).numpy()

        self.movie_embeddings = pd.DataFrame({"movieIdStr": self.movie_ids, "emb": list(movie_emb)})
        # rated movies cache
        self.user_rated_movies = (
            self.ratings
            .groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

    def _get_user_embedding(self, user_id):
        user_id_str = str(user_id)

        if user_id_str not in self.user_ids:
            return None

        emb = self.user_model(
            tf.constant([user_id_str])
        ).numpy()[0]

        return emb

    def _score_movies(self, user_embedding, movie_ids):
        movie_ids_str = [str(mid) for mid in movie_ids]

        movie_embeddings = self.movie_model(tf.constant(movie_ids_str)).numpy()

        scores = np.dot(movie_embeddings, user_embedding)
        return scores

    def recommend(
        self,
        user_id,
        top_k=10
    ):
        user_embedding = self._get_user_embedding(user_id)

        if user_embedding is None:
            return pd.DataFrame(columns=["movieId", "title", "genres", "score"])

        rated_movies = self.user_rated_movies.get(user_id, set())

        candidates = self.movies_data[
            ~self.movies_data["movieId"].isin(rated_movies)
        ].copy()

        if candidates.empty:
            return pd.DataFrame(columns=["movieId", "title", "genres", "score"])

        scores = self._score_movies(user_embedding, candidates["movieId"].tolist())
        candidates["score"] = scores
        candidates = candidates.sort_values(by="score", ascending=False)
        return candidates.head(top_k)[["movieId", "title", "genres", "score"]]


    def rank(self, user_id, movies_data: pd.DataFrame):
        user_embedding = self._get_user_embedding(user_id)
        movies_data = movies_data.copy()

        if user_embedding is None or movies_data.empty:
            movies_data["tfrsScore"] = 0.0
            return movies_data

        scores = self._score_movies(user_embedding, movies_data["movieId"].tolist())
        movies_data["tfrsScore"] = scores
        return movies_data