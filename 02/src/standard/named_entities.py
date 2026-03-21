import tempfile
from collections import Counter

import mlflow
import nltk
from nltk.tree import Tree

def get_continuous_chunks(chunked):
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))

        if current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                if hasattr(i, 'label') and not i.label() == 'NE':
                    continuous_chunk.append((named_entity, i.label()))
                else:
                    continuous_chunk.append((named_entity, None))
                current_chunk = []
        else:
            continue

    return continuous_chunk

def run_ner(dataset):
    recognized_entities_count = 0
    unique_entities = set()
    for article in dataset.articles:
        for sentence in article.document.sents:
            # tokens = nltk.word_tokenize(sentence.text)
            tokens = [t.text for t in sentence]
            tagged = nltk.pos_tag(tokens)
            ne_chunked = nltk.ne_chunk(tagged, binary=True)
            continuous_chunk = get_continuous_chunks(ne_chunked)
            recognized_entities_count += len(continuous_chunk)
            unique_entities.update([e[0] for e in continuous_chunk])

    mlflow.log_metric("nltk_recognized_entities_count", recognized_entities_count)
    mlflow.log_metric("nltk_unique_entities_count", len(unique_entities))

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        for e in unique_entities:
            f.write(e + "\n")
        temp_path = f.name

    mlflow.log_artifact(temp_path, artifact_path="nltk_unique_entities_binary")

def run_nec(dataset):
    unique_entities = set()
    recognized_classes = set()
    all_classes = []

    for article in dataset.articles:
        for sentence in article.document.sents:
            # tokens = nltk.word_tokenize(sentence.text)
            tokens = [t.text for t in sentence]
            tagged = nltk.pos_tag(tokens)
            ne_chunked = nltk.ne_chunk(tagged, binary=False)
            continuous_chunk = get_continuous_chunks(ne_chunked)
            # print(continuous_chunk)
            classified = [(e[0], e[1]) for e in continuous_chunk if e[1] is not None]

            recognized_classes.update([c for _, c in classified])
            unique_entities.update([f"{e}:{c}" for e, c in classified])
            all_classes.extend([c for _, c in classified])

    mlflow.log_metric("nltk_unique_entity_class_pairs", len(unique_entities))
    mlflow.log_metric("nltk_recognized_classes_count", len(recognized_classes))

    class_counts = Counter(all_classes)
    for cls, count in class_counts.items():
        mlflow.log_metric(f"nltk_class_occurrences_{cls}", count)

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        for e in unique_entities:
            f.write(e + "\n")
        temp_path = f.name

    mlflow.log_artifact(temp_path, artifact_path="nltk_unique_entity_class_pairs")