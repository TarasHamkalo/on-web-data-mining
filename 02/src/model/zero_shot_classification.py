from collections import Counter
import mlflow
from transformers import pipeline
import numpy as np

def run_zeroshot(dataset):
    pipe = pipeline(model="facebook/bart-large-mnli")

    candidate_labels = [
        "film", "television", "music", "theatre",
        "books", "celebrity", "awards", "art"
    ]

    counts = Counter()
    total = 0

    for article in dataset.articles:
        for sentence in article.document.sents:
            result = pipe(sentence.text, candidate_labels=candidate_labels)

            scores = result["scores"]
            labels = result["labels"]

            label = labels[np.argmax(scores)]

            counts[label] += 1
            total += 1

    for label in candidate_labels:
        mlflow.log_metric(
            f"zeroshot_count_{label}",
            counts[label]
        )
