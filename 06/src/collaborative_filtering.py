import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

from recommender import Recommender


class CollabFilteringRec(Recommender):

    def __init__(
        self,
        top_users_to_consider=10,
        use_pearson=False
    ):
        self.top_users_to_consider = top_users_to_consider

        self.user_movie_matrix = None
        self.sparse_matrix = None
        self.user_similarity = None

        self.movies_data = None
        self.ratings = None

        self.user_ids = None
        self.use_pearson = use_pearson

    def train(
        self,
        movies_data: pd.DataFrame,
        ratings: pd.DataFrame
    ):
        self.movies_data = movies_data.copy()
        self.ratings = ratings.copy()

        user_movie_matrix = ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            fill_value=0
        )

        self.user_movie_matrix = user_movie_matrix
        self.user_ids = list(user_movie_matrix.index)
        self.sparse_matrix = csr_matrix(user_movie_matrix.values)

        if not self.use_pearson:
            self.user_similarity = cosine_similarity(self.sparse_matrix)
            return

        self.user_similarity = np.zeros(
            (len(self.user_ids), len(self.user_ids))
        )

        for i in range(len(self.user_ids)):
            for j in range(i, len(self.user_ids)):
                ratings_i = self.user_movie_matrix.iloc[i].values
                ratings_j = self.user_movie_matrix.iloc[j].values

                common_mask = ((ratings_i > 0) & (ratings_j > 0))

                if np.sum(common_mask) < 2:
                    sim = 0
                else:
                    sim, _ = pearsonr(ratings_i[common_mask], ratings_j[common_mask])
                    if np.isnan(sim):
                        sim = 0

                self.user_similarity[i, j] = sim
                self.user_similarity[j, i] = sim

    def recommend(
        self,
        user_id,
        top_k=10
    ):
        """
            Pretty weird mapping of user ids to indexes in matrix and vice versa, tried to comment some
        """
        if user_id not in self.user_movie_matrix.index:
            return pd.DataFrame(
                columns=["movieId", "title", "genres", "score"]
            )

        user_idx = self.user_ids.index(user_id)  # get index of column

        # filter self.top_users_to_consider similar to users to given one (just internal indices)
        similarities = self.user_similarity[user_idx]
        similar_user_indices = np.argsort(similarities)[::-1]  # reversing array

        similar_user_indices = [
            idx
            for idx in similar_user_indices
            if idx != user_idx and similarities[idx] > 0  # filter those having negative similarity
        ][:self.top_users_to_consider]

        user_ratings = self.user_movie_matrix.loc[user_id]

        rated_movies = set(user_ratings[user_ratings > 0].index)

        movie_scores = {}
        # iterate of internal indices (remapping to ids and similarity)
        for similar_idx in similar_user_indices:

            similar_user_id = self.user_ids[similar_idx]
            similarity = similarities[similar_idx]

            similar_user_ratings = (self.user_movie_matrix.loc[similar_user_id])

            for movie_id, rating in similar_user_ratings.items():
                if rating <= 0:
                    continue

                if movie_id in rated_movies:
                    continue

                if movie_id not in movie_scores:
                    movie_scores[movie_id] = []

                movie_scores[movie_id].append((similarity, rating))  # add weighted score

        recommendations = []

        for movie_id, scores in movie_scores.items():
            similarity_sum = np.sum([s for s, _ in scores])
            weighted_sum = np.sum([s * r for s, r in scores])
            if similarity_sum == 0:
                score = 0
            else:
                score = weighted_sum / similarity_sum

            movie_row = self.movies_data[self.movies_data["movieId"] == movie_id].iloc[0]

            recommendations.append({
                "movieId": movie_id,
                "title": movie_row["title"],
                "genres": movie_row["genres"],
                "score": score
            })

        recommendations = sorted(
            recommendations,
            key=lambda x: x["score"],
            reverse=True
        )

        return pd.DataFrame(recommendations[:top_k])

    def rank(self, user_id, movies_data: pd.DataFrame):
        if user_id not in self.user_ids:
            movies_data = movies_data.copy()
            movies_data["cfrScore"] = 0.0
            return movies_data

        user_idx = self.user_ids.index(user_id)
        user_user_similarity = self.user_similarity[user_idx]

        user_id_to_similarity_map = {}
        movie_scores = {}
        for _, movie_data in movies_data.iterrows():
            movie_id = movie_data["movieId"]
            if movie_id not in self.user_movie_matrix.columns:
                movie_scores[movie_id] = 0.0
                continue

            user_ratings = self.user_movie_matrix.loc[:, movie_id]

            movie_user_ratings = []
            for other_user_id, rating in user_ratings.items():
                if rating <= 0:
                    continue

                if other_user_id == user_id:
                    continue

                if other_user_id not in user_id_to_similarity_map:
                    target_user_idx = self.user_ids.index(other_user_id)
                    similarity = user_user_similarity[target_user_idx]
                    user_id_to_similarity_map[other_user_id] = similarity
                else:
                    similarity = user_id_to_similarity_map[other_user_id]

                movie_user_ratings.append((similarity, rating))

            movie_user_ratings = sorted(
                movie_user_ratings,
                key=lambda x: x[0],
                reverse=True
            )

            movie_user_ratings = movie_user_ratings[:self.top_users_to_consider]
            if len(movie_user_ratings) == 0:
                movie_scores[movie_id] = 0.0
                continue

            weighted_sum = np.sum(s * r for s, r in movie_user_ratings)
            similarity_sum = np.sum([s for s, _ in movie_user_ratings])
            if similarity_sum == 0:
                movie_scores[movie_id] = 0.0
            else:
                movie_scores[movie_id] = (weighted_sum / similarity_sum)

        movies_data = movies_data.copy()

        movies_data["cfrScore"] = movies_data["movieId"].apply(
            lambda movie_id: movie_scores.get(movie_id, 0.0)
        )

        return movies_data
