import mlflow

from src.standard.runner import run_standard
from src.model.runner import run_model_based, run_zeroshot_model, run_summarization_model
from src.articles_dataset import ArticlesDataset
from datetime import datetime
if __name__ == "__main__":
    articles_dataset_path = \
        "./resources/bcc-news-entertainment.csv"
    dataset = ArticlesDataset(articles_dataset_path)
    dataset.load_articles(50)
    dataset.preprocess()

    sentences_count = 0
    for article in dataset.articles:
        sentences_count += len(list(article.document.sents))


    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("hw-02-nlp")

    with mlflow.start_run():
        mlflow.log_param("data_source", "bbc")
        mlflow.log_param("dataset_url", "https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive")
        mlflow.log_param("dataset_news_category", "entertainment")
        mlflow.log_param("sentences_count", sentences_count)

 
        start_time = datetime.now()
        mlflow.log_param("start_time", start_time.isoformat() + "Z")

        run_standard(dataset)
        mlflow.log_metric("nltk_duration", (datetime.now() - start_time).total_seconds())

        model_start_time = datetime.now()
        run_model_based(dataset)
        mlflow.log_metric("model_duration", (datetime.now() - model_start_time).total_seconds())

        classification_start_time = datetime.now()
        run_zeroshot_model(dataset)
        mlflow.log_metric(
            "zero_shot_classification_duration",
            (datetime.now() - classification_start_time).total_seconds()
        )

        summarization_start_time = datetime.now()
        run_summarization_model(dataset)
        mlflow.log_metric(
            "summarization_duration",
            (datetime.now() - summarization_start_time).total_seconds()
        )

        mlflow.log_metric("total_run_duration", (datetime.now() - start_time).total_seconds())
