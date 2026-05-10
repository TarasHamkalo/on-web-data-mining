import pandas as pd


class Recommender:
    """
        Interface for all recommender types, used to unify evaluation.
    """

    def train(self, movies_data: pd.DataFrame, ratings: pd.DataFrame):
        pass

    def recommend(self, user_id, top_k=10):
        """
        :return: pd.DataFrame(["movieId", "title", "genres", "score"])
        """
        pass

    def rank(self, user_id, movies_data: pd.DataFrame):
        """
        :return modified movie_data data frame containing another score column 
        """
        pass