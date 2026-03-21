from collections import Counter
import mlflow
from nltk.sentiment import SentimentIntensityAnalyzer

def run_sentiment(dataset):
    vader_analyzer = SentimentIntensityAnalyzer()
    counts = Counter()

    for article in dataset.articles:
        for sentence in article.document.sents:
            scores = vader_analyzer.polarity_scores(sentence.text)

            # argmax over neg/neu/pos
            label = max(
                ("neg", scores["neg"]),
                ("neu", scores["neu"]),
                ("pos", scores["pos"]),
                key=lambda x: x[1]
            )[0]

            counts[label] += 1

    # metrics
    mlflow.log_metric("nltk_sentiment_positive", counts["pos"])
    mlflow.log_metric("nltk_sentiment_negative", counts["neg"])
    mlflow.log_metric("nltk_sentiment_neutral", counts["neu"])
