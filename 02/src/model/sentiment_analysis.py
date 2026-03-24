import tempfile
from collections import Counter
import mlflow
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
from src.util import readable_id

def run_sentiment_model(dataset):
    task = "sentiment"
    model_name = f"cardiffnlp/twitter-roberta-base-{task}"


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    counts = Counter()
    classified_sentences = []
    for article in dataset.articles:
        for sentence in article.document.sents:
            encoded_input = tokenizer(sentence.text, return_tensors="pt", truncation=True)
            output = model(**encoded_input)

            scores = output[0][0].detach().numpy()
            scores = softmax(scores)

            label_idx = scores.argmax()
            label = ["neg", "neu", "pos"][label_idx]

            classified_sentences.append({
                "sentence_id": readable_id(sentence.text),
                "sentence": sentence.text,
                "label": label,
                "score": scores[label_idx]
            })

            counts[label] += 1

    # metrics
    mlflow.log_metric("model_sentiment_positive", counts["pos"])
    mlflow.log_metric("model_sentiment_negative", counts["neg"])
    mlflow.log_metric("model_sentiment_neutral", counts["neu"])

    classified_sentences_df = pd.DataFrame(classified_sentences)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        classified_sentences_df.to_csv(f, index=False)
        temp_path = f.name

    mlflow.log_artifact(temp_path, artifact_path="model_sentiment.csv")
