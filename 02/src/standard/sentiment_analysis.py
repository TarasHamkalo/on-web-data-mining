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
            if scores['compound'] >= 0.5:
                label = "pos"
            elif scores['compound'] <= -0.5:
                label = "neg"
            else:
                label = "neu"
            counts[label] += 1

    # metrics
    mlflow.log_metric("nltk_sentiment_positive", counts["pos"])
    mlflow.log_metric("nltk_sentiment_negative", counts["neg"])
    mlflow.log_metric("nltk_sentiment_neutral", counts["neu"])
