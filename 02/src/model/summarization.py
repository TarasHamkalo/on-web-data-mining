import tempfile

import numpy as np
import pandas as pd

import mlflow
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def run_summarization(dataset):
    model_name = "facebook/bart-large-cnn"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    per_article_ratios = []
    summaries = []
    for article in dataset.articles:
        inputs = tokenizer(
            article.content,
            return_tensors="pt",
        ).to(device)

        summary_ids = model.generate(
            **inputs,
            max_length=130,
            min_length=30,
            do_sample=False
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # 👇 compression ratio (summary / original)
        per_article_ratios.append(len(summary) / len(article.content))
        summaries.append({
            "title": article.title,
            "summary": summary,
            "content": article.content,
            "compression_ratio": len(summary) / len(article.content)
        })

    mlflow.log_metric(
        "summarization_compression_ratio_mean",
        np.mean(per_article_ratios)
    )

    mlflow.log_metric("summarization_articles_processed", len(dataset.articles))

    summaries_df = pd.DataFrame(summaries)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        summaries_df.to_csv(f, index=False)
        temp_path = f.name

    mlflow.log_artifact(temp_path, artifact_path="summaries")
