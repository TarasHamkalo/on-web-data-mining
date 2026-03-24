import tempfile
from collections import Counter
import mlflow
from transformers import pipeline
import numpy as np
import pandas as pd

def run_zeroshot(dataset):
    pipe = pipeline(model="facebook/bart-large-mnli", device=0)

    candidate_labels = [
        "film", "television", "music", "theatre",
        "books", "celebrity", "awards", "art"
    ]

    counts = Counter()
    total = 0

    classifications = []
    for article in dataset.articles:
        for sentence in article.document.sents:
            result = pipe(sentence.text, candidate_labels=candidate_labels)

            scores = result["scores"]
            labels = result["labels"]

            highest = np.argmax(scores)
            label = labels[highest]
            classifications.append({
                "sentence": sentence.text,
                "label": label,
                "score": scores[highest],
            })

            counts[label] += 1
            total += 1

    for label in candidate_labels:
        mlflow.log_metric(
            f"zeroshot_count_{label}",
            counts[label]
        )

    classified_sentences_df = pd.DataFrame(classifications)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        classified_sentences_df.to_csv(f, index=False)
        temp_path = f.name

    mlflow.log_artifact(temp_path, artifact_path="zero_shot_classification")

