from src.model.sentiment_analysis import run_sentiment_model
from src.model.named_entities import run_ner_model, run_nec_model

def run_model_based(dataset):
    run_ner_model(dataset)
    run_nec_model(dataset)
    run_sentiment_model(dataset)
