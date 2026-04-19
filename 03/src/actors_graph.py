from typing import List, Any
import re

from bs4 import BeautifulSoup
import networkx as nx

NAME_REGEX = re.compile(r"^[^\W_ .,'`\-]+([ .,'`\-][^\W_]+)*$", re.UNICODE)
REJECT_REGEX = re.compile(r"^s\s?a$", re.UNICODE) # unknown actor

class ActorsGraph:
    def __init__(self, path="../resources/movie/data/casts.html", tables_count=None):
        self.tables_count = tables_count
        self.path = path
        self.g = nx.Graph()
        self.html_content: BeautifulSoup | None = None

    def build(self):
        with open(self.path, "rb") as f:
            casts_html_content = f.read()
            self.html_content = BeautifulSoup(casts_html_content, "html5lib")

        casts_tables = self.html_content.find_all("table")
        casts_tables = casts_tables[1:]
        if self.tables_count is not None:
            casts_tables = casts_tables[:self.tables_count]


        movie_to_actors_map = {}

        for table in casts_tables:
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue

                movie = cells[1].text.strip()
                actor_name = cells[2].text.strip()
                id = actor_name.lower()


                if not NAME_REGEX.match(id) or REJECT_REGEX.match(id):
                    continue

                if not self.g.has_node(id):
                   self.g.add_node(id, name=actor_name)

                if movie not in movie_to_actors_map:
                    movie_to_actors_map[movie] = []

                for coactor in movie_to_actors_map[movie]:
                    if coactor == id:
                        # actor played few roles in same film, remove such
                        continue

                    if self.g.has_edge(id, coactor):
                        data = self.g.get_edge_data(id, coactor)
                        data["weight"] += 1
                    else:
                        self.g.add_edge(id, coactor, weight=1)

                movie_to_actors_map[movie].append(id)

    def component_stats(self):
        components = list(nx.connected_components(self.g))
        sizes = [len(c) for c in components]
        sizes.sort(reverse=True)

        total_nodes = self.g.number_of_nodes()

        largest_size = sizes[0]

        singleton_count = sum(1 for s in sizes if s == 1)

        small = sum(1 for s in sizes if 2 <= s <= 10)
        medium = sum(1 for s in sizes if 11 <= s <= 100)
        large = sum(1 for s in sizes if s > 100 and s != largest_size)

        return {
            "num_components": len(components),
            "largest_size": largest_size,
            "largest_ratio": largest_size / total_nodes,
            "singletons": singleton_count,
            "below_10_nodes": small,
            "below_100_nodes": medium,
            "above_100_nodes": large
        }

    def stats(self):
        return {
            "nodes_count": len(self.g.nodes),
            "edges_count": len(self.g.edges),
            "density": nx.density(self.g),
            "avg_clustering": nx.average_clustering(self.g),
            "bridges": list(nx.bridges(self.g)),
            "local_bridges": list(nx.local_bridges(self.g)),
            "degree_histogram": nx.degree_histogram(self.g),

        }
