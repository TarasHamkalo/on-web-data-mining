from standard.named_entities import run_ner, run_nec
from standard.sentiment_analysis import run_sentiment


def run_standard(dataset):
    run_ner(dataset)
    run_nec(dataset)
    run_sentiment(dataset)
