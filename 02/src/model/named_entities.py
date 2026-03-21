import tempfile
from collections import Counter
import mlflow
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def run_ner_model(dataset):
    model_name = "Jean-Baptiste/roberta-large-ner-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    nlp = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )

    recognized_entities_count = 0
    unique_entities = set()

    for article in dataset.articles:
        for sentence in article.document.sents:
            raw_ne = nlp(sentence.text)

            entities = [ne["word"] for ne in raw_ne]

            recognized_entities_count += len(entities)
            unique_entities.update(entities)

    # metrics
    mlflow.log_metric("model_recognized_entities_count", recognized_entities_count)
    mlflow.log_metric("model_unique_entities_count", len(unique_entities))

    # artifact
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        for e in sorted(unique_entities):
            f.write(e + "\n")
        temp_path = f.name

    mlflow.log_artifact(temp_path, artifact_path="model_unique_entities")


def run_nec_model(dataset):
    model_name = "Jean-Baptiste/roberta-large-ner-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    nlp = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )

    unique_entities = set()
    recognized_classes = set()
    all_classes = []

    for article in dataset.articles:
        for sentence in article.document.sents:
            raw_ne = nlp(sentence.text)

            classified = [
                (ne["word"], ne["entity_group"])
                for ne in raw_ne
                if ne["entity_group"] is not None
            ]

            recognized_classes.update([c for _, c in classified])
            unique_entities.update([f"{e}:{c}" for e, c in classified])
            all_classes.extend([c for _, c in classified])

    # metrics
    mlflow.log_metric("model_unique_entity_class_pairs", len(unique_entities))
    mlflow.log_metric("model_recognized_classes_count", len(recognized_classes))

    # class distribution
    class_counts = Counter(all_classes)
    for cls, count in class_counts.items():
        mlflow.log_metric(f"model_class_occurrences_{cls}", count)

    # artifact
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        for e in sorted(unique_entities):
            f.write(e + "\n")
        temp_path = f.name

    mlflow.log_artifact(temp_path, artifact_path="model_entity_class_pairs")