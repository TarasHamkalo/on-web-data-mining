import mlflow
from src.standard.runner import run_standard
from src.model.runner import run_model_based
from src.articles_dataset import ArticlesDataset
from datetime import datetime
if __name__ == "__main__":
    articles_dataset_path = \
        "./resources/bcc-news-entertainment.csv"
    dataset = ArticlesDataset(articles_dataset_path)
    dataset.load_articles(1)
    dataset.preprocess()

    sents_count = 0
    for article in dataset.articles:
        sents_count += len(list(article.document.sents))


    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("hw-02-nlp")

    with mlflow.start_run():
        start_time = datetime.now()
        mlflow.log_param("start_time", start_time.isoformat() + "Z")

        run_standard(dataset)
        mlflow.log_metric("nltk_duration", (datetime.now() - start_time).total_seconds())

        run_model_based(dataset)
        mlflow.log_metric("model_duration", (datetime.now() - start_time).total_seconds())