import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from recommender import Recommender

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]

    relevant_retrieved = sum(
        1 for doc_id in retrieved_k
        if doc_id in relevant
    )

    return relevant_retrieved / k


def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(
        1 for doc_id in retrieved_k
        if doc_id in relevant
    )

    return relevant_retrieved / min(k, len(retrieved_k))


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0

    return 2 * precision * recall / (precision + recall)


# find position of first relevant document in results
def reciprocal_rank(retrieved, relevant):
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1 / rank

    return 0


# for each of ks compute avg precision, recall, f1 across all queries
# def evaluate_similarity_matrix(
#     document_ids,
#     similarity_matrix,
#     larger_is_better=True,
#     ks=frozenset([1, 3, 5, 10, 15, 25, 35, 50, 100, 1000]),
# ):
#     precision_scores = {k: [] for k in ks}
#     recall_scores = {k: [] for k in ks}
#     f1_scores = {k: [] for k in ks}
#
#     reciprocal_ranks = []
#
#     for q_idx, (query_id, _, relevant_docs) in enumerate(queries):
#         relevant_docs = set(relevant_docs)
#         scores = similarity_matrix[q_idx]
#
#         if larger_is_better:
#             ranked_indices = np.argsort(scores)[::-1]
#         else:
#             ranked_indices = np.argsort(scores)
#
#         ranked_doc_ids = [
#             document_ids[idx]
#             for idx in ranked_indices
#         ]
#
#         for k in ks:
#             precision = precision_at_k(
#                 ranked_doc_ids,
#                 relevant_docs,
#                 k
#             )
#
#             recall = recall_at_k(
#                 ranked_doc_ids,
#                 relevant_docs,
#                 k
#             )
#
#             f1 = f1_score(precision, recall)
#
#             precision_scores[k].append(precision)
#             recall_scores[k].append(recall)
#             f1_scores[k].append(f1)
#
#         reciprocal_ranks.append(
#             reciprocal_rank(
#                 ranked_doc_ids,
#                 relevant_docs
#             )
#         )
#
#     avg_precision = [
#         np.mean(precision_scores[k])
#         for k in ks
#     ]
#
#     avg_recall = [
#         np.mean(recall_scores[k])
#         for k in ks
#     ]
#
#     avg_f1 = [
#         np.mean(f1_scores[k])
#         for k in ks
#     ]
#
#     mrr = np.mean(reciprocal_ranks)
#
#     return {
#         "ks": ks,
#         "precision": avg_precision,
#         "recall": avg_recall,
#         "f1": avg_f1,
#         "mrr": mrr
#     }



import matplotlib.pyplot as plt


def plot_results(results, title, save_path=None):
    ks = results["ks"]
    plt.figure(figsize=(8, 5))
    # [float(p) * 100 for p in results["precision"]],

    plt.plot(
        ks,
        results["precision"],
        marker="o",
        label="Precision@k"
    )

    plt.plot(
        ks,
        results["recall"],
        marker="o",
        label="Recall@k"
    )
    # [float(p) * 100 for p in results["recall"]],

    plt.plot(
        ks,
        results["f1"],
        marker="o",
        label="F1@k"
    )
    # [float(p) * 100  for p in results["f1"]],

    plt.xlabel("k")
    plt.ylabel("Score")

    plt.title(
        f"{title}"#\nMRR={results['mrr']:.4f}"
    )
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def train_test_split_ratings(ratings_data: DataFrame, test_size_per_user=0.3, seed=42):
    train_parts = []
    test_parts = []

    for user_id, user_ratings in ratings_data.groupby("userId"):
        train, test = train_test_split(
            user_ratings,
            test_size=test_size_per_user,
            random_state=seed
        )

        train_parts.append(train)
        test_parts.append(test)

    train_ratings = pd.concat(train_parts).reset_index(drop=True)
    test_ratings = pd.concat(test_parts).reset_index(drop=True)
    return train_ratings, test_ratings

def evaluate(
    recommender: Recommender,
    movies_data: pd.DataFrame,
    ratings_data: pd.DataFrame,
    relevance_threshold=3.5,
    ks=[1, 3, 5, 10, 15, 25, 35, 50, 100, 1000],
    test_size_per_user=0.3,
    seed=42
):
    """
    :param: relevance_threshold min ground truth rating to consider recommendation relevant
    :return: ["movieId", "title", "genres", "score"]
    """
    precision_scores = {k: [] for k in ks}
    recall_scores = {k: [] for k in ks}
    f1_scores = {k: [] for k in ks}
    query_times = {k: [] for k in ks}

    train_ratings, test_ratings = train_test_split_ratings(
        ratings_data, test_size_per_user=test_size_per_user, seed=seed
    )

    train_start_time = datetime.datetime.now()
    recommender.train(movies_data=movies_data, ratings=train_ratings)

    train_duration = datetime.datetime.now() - train_start_time

    user_ids = list(ratings_data["userId"].unique())

    for user_id in user_ids:
        user_test = test_ratings[test_ratings["userId"] == user_id]
        user_test = user_test[user_test["rating"] >= relevance_threshold].sort_values(by="rating")

        relevant_movies_ordered = list(user_test["movieId"])
        if len(relevant_movies_ordered) == 0:
            print(f"User {user_id} is dropped, as it has relevant movies len {len(relevant_movies_ordered)}")
            continue

        for top_k in ks:
            query_time_start = datetime.datetime.now()
            recommendations = recommender.recommend(user_id, top_k=top_k)
            query_time = datetime.datetime.now() - query_time_start


            recommended_movies = list(recommendations["movieId"])

            precision = precision_at_k(recommended_movies, relevant_movies_ordered, k=top_k)
            recall = recall_at_k(recommended_movies, relevant_movies_ordered, k=top_k)
            f1 = f1_score(precision, recall)

            precision_scores[top_k].append(precision)
            recall_scores[top_k].append(recall)
            f1_scores[top_k].append(f1)

            query_times[top_k].append(query_time)

    avg_precision = [
        np.mean(precision_scores[k])
        for k in ks
    ]

    avg_recall = [
        np.mean(recall_scores[k])
        for k in ks
    ]

    avg_f1 = [
        np.mean(f1_scores[k])
        for k in ks
    ]

    avg_query_time = [
        np.mean(query_times[k])
        for k in ks
    ]


    # mrr = np.mean(reciprocal_ranks)

    return {
        "train_size": len(train_ratings),
        "test_size": len(test_ratings),

        "train_duration": train_duration,

        "ks": ks,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "query_time": avg_query_time,
    }