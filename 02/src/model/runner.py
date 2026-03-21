from model.zero_shot_classification import run_zeroshot
from src.model.sentiment_analysis import run_sentiment_model
from src.model.named_entities import run_ner_model, run_nec_model

from src.model.summarization import run_summarization
def run_model_based(dataset):
    run_ner_model(dataset)
    run_nec_model(dataset)
    run_sentiment_model(dataset)

def run_zeroshot_model(dataset):
    run_zeroshot(dataset)

def run_summarization_model(dataset):
    run_summarization(dataset)