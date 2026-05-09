import pandas as pd


class Recommender:
    def train(self, movies_data: pd.DataFrame, ratings: pd.DataFrame):
        pass

    def recommend(self, user_id, top_k=10):
        """
        :return: ["movieId", "title", "genres", "score"]
        """
        pass

    def rank(self, user_id, movies_data: pd.DataFrame):
        pass