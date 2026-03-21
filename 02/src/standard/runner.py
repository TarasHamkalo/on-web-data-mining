from datetime import datetime

from standard.sentiment_analysis import run_sentiment


def run_standard(dataset):
    # run_ner(dataset)
    # run_nec(dataset)
    run_sentiment(dataset)

