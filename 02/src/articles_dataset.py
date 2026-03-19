import spacy
import csv
from typing import List

from article import Article

class ArticlesDataset:

    _nlp = spacy.load("en_core_web_sm")

    def __init__(self, path):
        self.path = path

        self.articles: List[Article] = []

    def load_articles(self, count=3):

        with open(self.path) as f:
            csv_reader = csv.reader(f, delimiter=',')
            next(csv_reader) # skip header

            for i in range(0, count):
                line = next(csv_reader)
                if not line:
                   break

                self.articles.append(
                    Article(
                        category=line[0],
                        title=line[2],
                        content=line[3],
                    )
                )


    def preprocess(self):
        for article in self.articles:
            article.document = self._nlp(article.content)
