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
        self.use_content_first = use_content_first
        self.enlarge_candidate_pool_times = enlarge_candidate_pool_times
        self.max_candidate_pool = max_candidate_pool

    def train(self, movies_data: pd.DataFrame, ratings: pd.DataFrame):
        self.cbr.train(movies_data, ratings)
        self.cfr.train(movies_data, ratings)

    def recommend(self, user_id, top_k=10):
        if self.use_content_first:
            r = self.recommend_cb_cf(user_id, top_k)
        else:
            r = self.recommend_cf_cb(user_id, top_k)
        return r

    # TODO: unify scoring
    def recommend_cb_cf(self, user_id, top_k=10):
        candidate_top_k = top_k

        if self.enlarge_candidate_pool_times is not None:
            candidate_top_k *= self.enlarge_candidate_pool_times
            candidate_top_k = min(candidate_top_k, self.max_candidate_pool)

        recommendations = self.cbr.recommend(user_id=user_id, top_k=candidate_top_k)
        recommendations = self.cfr.rank(user_id=user_id, movies_data=recommendations)

        recommendations = recommendations.sort_values(by="cfrScore", ascending=False)

        return recommendations.head(top_k)

    def recommend_cf_cb(self, user_id, top_k=10):
        candidate_top_k = top_k

        if self.enlarge_candidate_pool_times is not None:
            candidate_top_k *= self.enlarge_candidate_pool_times
            candidate_top_k = min(candidate_top_k, self.max_candidate_pool)

        recommendations = self.cfr.recommend(user_id=user_id, top_k=candidate_top_k)
        recommendations = self.cbr.rank(user_id=user_id, movies_data=recommendations)

        recommendations = recommendations.sort_values(by="cbrScore", ascending=False)

        return recommendations.head(top_k)