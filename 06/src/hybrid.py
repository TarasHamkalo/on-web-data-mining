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
        enlarge_candidate_pool_times=5,
        max_candidate_pool=100
    ):
        self.cbr = cbr
        self.cfr = cfr

        # True  -> Content-Based -> Collaborative re-ranking
        # False -> Collaborative -> Content-Based re-ranking
        self.use_content_first = use_content_first

        self.enlarge_candidate_pool_times = enlarge_candidate_pool_times
        self.max_candidate_pool = max_candidate_pool

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

        candidates = candidates.sort_values(by=second_score_column, ascending=False)
        return candidates.head(top_k)

    def _compute_candidate_pool_size(
        self,
        top_k
    ):
        candidate_top_k = top_k
        if ((self.max_candidate_pool is not None) and
            (candidate_top_k >= self.max_candidate_pool)):
            return candidate_top_k

        if self.enlarge_candidate_pool_times is not None:
            candidate_top_k *= self.enlarge_candidate_pool_times

        if self.max_candidate_pool is not None:
            candidate_top_k = min(
                candidate_top_k,
                self.max_candidate_pool
            )

        return candidate_top_k
