import tempfile
from collections import Counter
import mlflow
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from src.util import readable_id


def run_sentiment(dataset):
    vader_analyzer = SentimentIntensityAnalyzer()
    counts = Counter()

    classified_sentences = []
    for article in dataset.articles:
        for sentence in article.document.sents:
            scores = vader_analyzer.polarity_scores(sentence.text)

            # argmax over neg/neu/pos
            if scores['compound'] >= 0.5:
                label = "pos"
            elif scores['compound'] <= -0.5:
                label = "neg"
            else:
                label = "neu"

            classified_sentences.append(
                {
                    "sentence_id": readable_id(sentence.text),
                    "sentence": sentence.text,
                    "label": label,
                    "score (compound)": scores['compound']
                }
            )
            counts[label] += 1

    # metrics
    mlflow.log_metric("nltk_sentiment_positive", counts["pos"])
    mlflow.log_metric("nltk_sentiment_negative", counts["neg"])
    mlflow.log_metric("nltk_sentiment_neutral", counts["neu"])

    classified_sentences_df = pd.DataFrame(classified_sentences)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        classified_sentences_df.to_csv(f, index=False)
        temp_path = f.name

    mlflow.log_artifact(temp_path, artifact_path="nltk_sentiment")

