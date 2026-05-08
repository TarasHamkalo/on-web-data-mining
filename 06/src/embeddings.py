import pandas as pd
from sentence_transformers import SentenceTransformer


def compute_movie_embeddings(
    movies_data: pd.DataFrame,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    model = SentenceTransformer(model_name)
    texts = (
        movies_data["title"].fillna("") +
        " ; " +
        movies_data["genres"]
            .fillna("")
            .str.split("|")
            .apply(lambda genres: " ".join(genres))
    ).tolist()

    embeddings = model.encode(texts, show_progress_bar=False)

    movies_data["emb"] = list(embeddings)

    return movies_data
