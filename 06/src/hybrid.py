import pandas as pd

from collaborative_filtering import CollabFilteringRec
from content_based import ContentBasedRec
from recommender import Recommender


class HybridRec(Recommender):

    def __init__(
        self,
        cbr: ContentBasedRec,
        cfr: CollabFilteringRec,
        use_content_first=True,
        enlarge_candidate_pool_times=3.5
    ):
        self.cbr = cbr
        self.cfr = cfr

        # True  -> Content-Based -> Collaborative re-ranking
        # False -> Collaborative -> Content-Based re-ranking
        self.use_content_first = use_content_first

        self.enlarge_candidate_pool_times = enlarge_candidate_pool_times

    def train(self, movies_data: pd.DataFrame, ratings: pd.DataFrame):
        self.cbr.train(movies_data, ratings)
        self.cfr.train(movies_data, ratings)

    def recommend(self, user_id, top_k=10):
        if self.use_content_first:
            return self._recommend_pipeline(
                user_id=user_id,
                top_k=top_k,
                first_recommender=self.cbr,
                second_recommender=self.cfr,
                second_score_column="cfrScore"
            )

        return self._recommend_pipeline(
            user_id=user_id,
            top_k=top_k,
            first_recommender=self.cfr,
            second_recommender=self.cbr,
            second_score_column="cbrScore"
        )

    def _recommend_pipeline(
        self,
        user_id,
        top_k,
        first_recommender,
        second_recommender,
        second_score_column
    ):
        candidate_top_k = self._compute_candidate_pool_size(top_k)

        candidates = first_recommender.recommend(user_id=user_id, top_k=candidate_top_k)
        if candidates.empty:
            return candidates

        candidates = second_recommender.rank(user_id=user_id, movies_data=candidates)

        candidates = self._normalize_column(candidates, "score", "score_norm")

        candidates = self._normalize_column(
            candidates,
            second_score_column,
            "rerank_score_norm"
        )

        candidates["hybridScore"] = (candidates["score_norm"] + candidates["rerank_score_norm"]) / 2.0
        candidates = candidates.sort_values(by="hybridScore", ascending=False)

        return candidates.head(top_k)

    def _compute_candidate_pool_size(self, top_k):
        if self.enlarge_candidate_pool_times is None:
            return top_k

        return max(top_k, int(top_k * self.enlarge_candidate_pool_times))

    def _normalize_column(self, data: pd.DataFrame, source_column, target_column):
        """
        Normalize given column to range from 0 to 1
        """
        data = data.copy()
        min_value = data[source_column].min()
        max_value = data[source_column].max()

        if max_value == min_value:
            data[target_column] = 0.0
            return data

        data[target_column] = (data[source_column] - min_value) / (max_value - min_value)
        return data
