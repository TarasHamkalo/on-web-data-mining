import numpy as np
import pandas as pd

import mlflow
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def run_summarization(dataset):
    model_name = "facebook/bart-large-cnn"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    per_article_ratios = []
    for article in dataset.articles:
        inputs = tokenizer(
            article.content,
            return_tensors="pt",
        )

        summary_ids = model.generate(
            **inputs,
            max_length=130,
            min_length=30,
            do_sample=False
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # 👇 compression ratio (summary / original)
        per_article_ratios.append(len(summary) / len(article.content))

    mlflow.log_metric(
        "summarization_compression_ratio_mean",
        np.mean(per_article_ratios)
    )

    mlflow.log_metric("summarization_articles_processed", len(dataset.articles))