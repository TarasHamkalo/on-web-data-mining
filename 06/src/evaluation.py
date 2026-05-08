import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

    return relevant_retrieved / len(relevant)


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0

    return 2 * precision * recall / (precision + recall)


def reciprocal_rank(retrieved, relevant):
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1 / rank

    return 0


def dcg_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]

    dcg = 0

    for idx, doc_id in enumerate(retrieved_k):
        if doc_id in relevant:
            dcg += 1 / np.log2(idx + 2)

    return dcg


def ndcg_at_k(retrieved, relevant, k):
    dcg = dcg_at_k(retrieved, relevant, k)

    ideal_hits = min(len(relevant), k)

    if ideal_hits == 0:
        return 0

    idcg = sum(
        1 / np.log2(i + 2)
        for i in range(ideal_hits)
    )

    return dcg / idcg

def diversity_scores_all_ks(
    recommended_movies,
    movies_data,
    ks
):
    emb_map = dict(
        zip(
            movies_data["movieId"],
            movies_data["emb"]
        )
    )

    embeddings = np.array([
        emb_map[mid]
        for mid in recommended_movies
        if mid in emb_map
    ])

    if len(embeddings) <= 1:
        return {
            k: 0
            for k in ks
        }

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    normalized = embeddings / norms
    similarity_matrix = normalized @ normalized.T
    diversities = {}
    for k in ks:
        k = min(k, len(similarity_matrix))
        sims = similarity_matrix[:k, :k]
        upper = sims[
            np.triu_indices(k, k=1)
        ]

        if len(upper) == 0:
            diversities[k] = 0
        else:
            diversities[k] = 1 - np.mean(upper)

    return diversities

def plot_results(results, title, save_path=None):
    ks = results["ks"]
    plt.figure(figsize=(8, 5))

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

    plt.plot(
        ks,
        results["f1"],
        marker="o",
        label="F1@k"
    )

    plt.plot(
        ks,
        results["ndcg"],
        marker="o",
        label="nDCG@k"
    )

    plt.xlabel("k")
    plt.ylabel("Score")

    plt.title(
        f"{title}\nMRR={results['mrr']:.4f}"
    )

    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path)

    plt.show()

from matplotlib.ticker import ScalarFormatter


def plot_results_2(results, title, save_path=None):
    ks = results["ks"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5)
    )

    axes[0].plot(
        ks,
        results["precision"],
        marker="o",
        label="Precision@k"
    )

    axes[0].plot(
        ks,
        results["recall"],
        marker="o",
        label="Recall@k"
    )

    axes[0].plot(
        ks,
        results["f1"],
        marker="o",
        label="F1@k"
    )

    axes[0].set_xscale("log", base=10)
    axes[0].set_yscale("log", base=10)

    axes[0].set_xticks(ks)

    axes[0].set_yticks([
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3
    ])

    axes[0].xaxis.set_major_formatter(
        ScalarFormatter()
    )

    axes[0].yaxis.set_major_formatter(
        ScalarFormatter()
    )

    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Score")

    axes[0].set_title(
        "Precision / Recall / F1"
    )

    axes[0].grid(True, which="both")
    axes[0].legend()

    axes[1].plot(
        ks,
        results["ndcg"],
        marker="o",
        label="nDCG@k"
    )

    axes[1].axhline(
        results["mrr"],
        linestyle="--",
        label=f"MRR={results['mrr']:.4f}"
    )

    axes[1].set_xscale("log", base=10)
    # axes[1].set_yscale("log", base=10)

    axes[1].set_xticks(ks)

    axes[1].xaxis.set_major_formatter(
        ScalarFormatter()
    )

    # axes[1].yaxis.set_major_formatter(
    #     ScalarFormatter()
    # )

    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Score")

    axes[1].set_title(
        "Ranking Quality"
    )

    axes[1].grid(True, which="both")
    axes[1].legend()

    axes[2].plot(
        ks,
        results["diversity"],
        marker="o",
        label="Diversity@k"
    )

    axes[2].set_xscale("log", base=10)
    # axes[2].set_yscale("log", base=10)

    axes[2].set_xticks(ks)

    axes[2].xaxis.set_major_formatter(
        ScalarFormatter()
    )
    #
    # axes[2].yaxis.set_major_formatter(
    #     ScalarFormatter()
    # )

    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Score")

    axes[2].set_title(
        "Recommendation Diversity"
    )

    axes[2].grid(True, which="both")
    axes[2].legend()

    fig.suptitle(title)

    plt.tight_layout()

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
    ks=[1, 3, 5, 10, 15, 25, 35, 50, 100, 500],
    test_size_per_user=0.3,
    seed=42
):
    """
    :warn expects movies_data to have "emb" field with embeddings computed
    """
    precision_scores = {k: [] for k in ks}
    recall_scores = {k: [] for k in ks}
    f1_scores = {k: [] for k in ks}
    ndcg_scores = {k: [] for k in ks}
    diversity_scores = {k: [] for k in ks}

    reciprocal_ranks = []

    train_ratings, test_ratings = train_test_split_ratings(
        ratings_data,
        test_size_per_user=test_size_per_user,
        seed=seed
    )

    train_start_time = datetime.datetime.now()
    recommender.train(movies_data=movies_data, ratings=train_ratings)

    train_duration = datetime.datetime.now() - train_start_time

    user_ids = list(ratings_data["userId"].unique())
    max_k = max(ks)

    query_times_1000 = []

    dropped_users = []



    for user_id in user_ids:
        user_test = test_ratings[test_ratings["userId"] == user_id]
        user_test = user_test[user_test["rating"] >= relevance_threshold]

        relevant_movies_ordered = list(user_test["movieId"])
        if len(relevant_movies_ordered) == 0:
            dropped_users.append(user_id)
            continue

        query_time_start = datetime.datetime.now()

        recommendations = recommender.recommend(user_id, top_k=max_k)

        query_time = datetime.datetime.now() - query_time_start

        query_times_1000.append(query_time)

        recommended_movies = list(recommendations["movieId"])

        reciprocal_ranks.append(
            reciprocal_rank(
                recommended_movies,
                relevant_movies_ordered
            )
        )

        diversities = diversity_scores_all_ks(
            recommended_movies,
            movies_data,
            ks
        )


        for top_k in ks:
            recommended_movies_k = recommended_movies[:top_k]

            precision = precision_at_k(
                recommended_movies_k,
                relevant_movies_ordered,
                k=top_k
            )

            recall = recall_at_k(
                recommended_movies_k,
                relevant_movies_ordered,
                k=top_k
            )

            f1 = f1_score(
                precision,
                recall
            )

            ndcg = ndcg_at_k(
                recommended_movies_k,
                relevant_movies_ordered,
                k=top_k
            )

            precision_scores[top_k].append(precision)
            recall_scores[top_k].append(recall)
            f1_scores[top_k].append(f1)
            ndcg_scores[top_k].append(ndcg)
            diversity_scores[top_k].append(
                diversities[top_k]
            )

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

    avg_ndcg = [
        np.mean(ndcg_scores[k])
        for k in ks
    ]

    avg_diversity = [
        np.mean(diversity_scores[k])
        for k in ks
    ]

    mrr = np.mean(reciprocal_ranks)

    avg_query_time_1000 = np.mean(query_times_1000)

    return {
        "train_size": len(train_ratings),
        "test_size": len(test_ratings),
        "dropped_user_ids": dropped_users,

        "train_duration": train_duration,
        "query_time_1000": avg_query_time_1000,

        "ks": ks,

        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "ndcg": avg_ndcg,
        "diversity": avg_diversity,

        "mrr": mrr,
    }
