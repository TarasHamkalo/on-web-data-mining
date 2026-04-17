#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mlflow
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("hw-03-sna")

mlflow.start_run()
mlflow.log_param("data_source",  "movie_dataset")


# In[2]:


import gzip
import networkx as nx

from actors_graph import ActorsGraph

actors_graph = ActorsGraph("../resources/movie/data/casts.html")
actors_graph.build()

with gzip.open("../results/actor_graph.gexf.gz", "wb") as f:
    nx.write_gexf(actors_graph.g, f)

mlflow.log_artifact("../results/actor_graph.gexf.gz")


# ## Log stats and comopnent stats

# In[3]:


stats = actors_graph.stats()

mlflow.log_metric("nodes", stats["nodes_count"])
mlflow.log_metric("edges", stats["edges_count"])
mlflow.log_metric("density", stats["density"])
mlflow.log_metric("clustering", stats["clustering_coefficient"])


bridges = [
    (a, b) for (a, b) in stats["bridges"]
    if actors_graph.g.degree(a) > 1 and actors_graph.g.degree(b) > 1
]

local_bridges = [
    (a, b) for (a, b, _) in stats["local_bridges"]
    if actors_graph.g.degree(a) > 1 and actors_graph.g.degree(b) > 1
]

mlflow.log_metric("num_bridges_high_deg", len(bridges))
mlflow.log_metric("num_local_bridges_high_deg", len(local_bridges))

comp = actors_graph.component_stats()
for k, v in comp.items():
    mlflow.log_metric(k, v)


# ## Log demo bridge

# In[6]:


import networkx as nx
from matplotlib import pyplot as plt

u, v = ('francisco rabal', 'silvia pinal')
# nodes_u = nx.single_source_shortest_path_length(actors_graph.g, u, cutoff=2).keys()
nodes_u = nx.neighbors(actors_graph.g, u)
nodes_v = nx.neighbors(actors_graph.g, v)
# nodes_v = nx.single_source_shortest_path_length(actors_graph.g, v, cutoff=2).keys()
nodes = set(nodes_u) | set(nodes_v)

subG = actors_graph.g.subgraph(nodes).copy()
pos = nx.kamada_kawai_layout(subG)
scale = 3
pos = {k: (x * scale, y * scale) for k, (x, y) in pos.items()}

# base graph
nx.draw(
    subG,
    pos,
    node_size=40,
    node_color="lightblue",
    edge_color="gray",
    with_labels=True,
    alpha=0.7
)

# highlight nodes
nx.draw_networkx_nodes(
    subG,
    pos,
    nodelist=[u, v],
    node_color="red",
    node_size=120
)

# highlight edge
nx.draw_networkx_edges(
    subG,
    pos,
    edgelist=[(u, v)],
    edge_color="red",
    width=2.5
)

plt.savefig("../outputs/demo_bridge.png", dpi=100, bbox_inches="tight")
plt.close()
mlflow.log_artifact("../outputs/demo_bridge.png", artifact_path="visualizations")


# ## Degree histogram

# In[11]:


from plotter import Plotter
top_by_degree = sorted(actors_graph.g.degree, key=lambda x: x[1], reverse=True)[:5]
Plotter().plot_histogram_bin_count(
    sequence=stats["degree_histogram"],
    bins_count=len(stats["degree_histogram"]),
    log_y_scale=False,
    title="Degree Distribution",
    x_label="Degree",
    y_label="Frequency",
    label=f"Max degree ({top_by_degree[0][1]})",
    x_majors=30,
    output_file="../outputs/degree_histogram.png"
)


mlflow.log_artifact("../outputs/degree_histogram.png", artifact_path="visualizations")


# ## Communities

# In[12]:


from networkx.algorithms import community
communities = list(community.louvain_communities(actors_graph.g))

mlflow.log_metric("modularity", community.modularity(actors_graph.g, communities))
sizes = [len(c) for c in communities]

mlflow.log_metric("num_communities", len(communities))
mlflow.log_metric("singleton_communities", sum(1 for s in sizes if s == 1))
mlflow.log_metric("communities_gt_10", sum(1 for s in sizes if s > 10))
mlflow.log_metric("communities_10_100", sum(1 for s in sizes if 10 < s <= 100))
mlflow.log_metric("communities_100_1000", sum(1 for s in sizes if 100 < s <= 1000))
mlflow.log_metric("communities_gt_1000", sum(1 for s in sizes if s > 1000))


# In[19]:


middle_size_comms = [comm for comm in communities if 10 < len(comm) <= 100]
comm_size_95_i = [len(comm) for comm in middle_size_comms].index(95)
comm_size_95 = middle_size_comms[comm_size_95_i]
print("Size of chosen community is", len(comm_size_95))


# In[20]:


subgraph_g = nx.subgraph(actors_graph.g, comm_size_95)
pos = nx.kamada_kawai_layout(subgraph_g)
scale = 2
pos = {k: (x * scale, y * scale) for k, (x, y) in pos.items()}
plt.figure(1, figsize=(20, 20))
nx.draw(
    subgraph_g,
    pos,
    node_size=200,
    node_color="lightblue",
    edge_color="gray",
    with_labels=True,
    alpha=0.7
)
plt.savefig("../outputs/demo_community.png", dpi=100, bbox_inches="tight")
plt.close()
mlflow.log_artifact(
    "../outputs/demo_community.png",
    artifact_path="visualizations"
)


# In[ ]:





# ### Kevin bacon

# In[22]:


import numpy as np

node_distances = []
for node in actors_graph.g.nodes:
    try:
        dist = nx.shortest_path_length(actors_graph.g, source=node, target="kevin bacon")
    except:
        dist = np.inf
    node_distances.append({
        "node": node,
        "distance": dist
    })

import pandas as pd
df = pd.DataFrame(node_distances, columns=["node", "distance"])
df = df[df["distance"] != np.inf]


# In[25]:


from plotter import Plotter

sequence = df["distance"].to_numpy()
Plotter().plot_distance_distribution(
    sequence,
    label="Kevin Bacon distances",
    log_y_scale=True,
    output_file="../outputs/kevin-distance-distribution.png"
)
mlflow.log_artifact(
    "../outputs/kevin-distance-distribution.png",
    artifact_path="visualizations"
)

mlflow.log_metric("kevin_bacon_dist_mean", sequence.mean())


# In[28]:


kevin_ego_g = nx.ego_graph(actors_graph.g, "kevin bacon", radius=1)
print("Direct neigboors: ", len(kevin_ego_g.nodes))
pos = nx.kamada_kawai_layout(kevin_ego_g)
scale = 1
pos = {k: (x * scale, y * scale) for k, (x, y) in pos.items()}
plt.figure(1, figsize=(12, 12))
nx.draw(
    kevin_ego_g,
    pos,
    node_size=80,
    node_color="lightblue",
    edge_color="gray",
    with_labels=True,
    alpha=0.7
)
nx.draw_networkx_nodes(
    kevin_ego_g,
    pos,
    nodelist=["kevin bacon"],
    node_color="red",
    node_size=120
)
plt.savefig("../outputs/kevin-ego-graph.png", dpi=100, bbox_inches="tight")
plt.close()
mlflow.log_artifact(
    "../outputs/kevin-ego-graph.png",
    artifact_path="visualizations"
)


# ## Embeddings

# In[31]:


from node2vec import Node2Vec
from gensim.models import Word2Vec
train = False
# node2vec
if train:
    node2vec = Node2Vec(actors_graph.g, dimensions=50, walk_length=15, num_walks=200, workers=12)
    model = node2vec.fit(window=5, min_count=1, batch_words=4)
    model.save("../results/node2vec.model")
else:
    model = Word2Vec.load("../results/node2vec.model")
mlflow.log_param("embedding_method", "node2vec")
mlflow.log_artifact(
    "../results/node2vec.model",
    artifact_path="models"
)


# In[32]:


import json
import mlflow

similar_to_kevin_bacon = model.wv.similar_by_key("kevin bacon", topn=53)

output_path = "../outputs/kevin_bacon_similar.json"

with open(output_path, "w") as f:
    json.dump(
        [{"name": name, "score": float(score)} for name, score in similar_to_kevin_bacon],
        f,
        indent=2
    )

mlflow.log_artifact(output_path, artifact_path="similarities")


# In[34]:


kevin_ego_nodes_set = set(kevin_ego_g.nodes)

similar_to_kevin_bacon_set = set([k for k, v in similar_to_kevin_bacon[:10]])

kevin_intersection = similar_to_kevin_bacon_set & kevin_ego_nodes_set
mlflow.log_metric("top10_similar_neighbors_overlap", len(kevin_intersection))


# In[35]:


top_k = 53

similar_set = set([k for k, v in similar_to_kevin_bacon[:top_k]])
intersection = similar_set & kevin_ego_nodes_set

mlflow.log_metric("top53_similar_neighbors_overlap", len(intersection))


# In[36]:


import networkx as nx
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# visualize ego graph with radius 1
from adjustText import adjust_text

similar = [k for k in kevin_ego_nodes_set]
kevin_index = similar.index("kevin bacon")
vectors = [
    model.wv.get_vector(k) for k in similar
]

pca = PCA(n_components=2)
pca = pca.fit(model.wv.vectors)

pca_comps = pca.transform(vectors)

plt.figure(figsize=(22, 12))
plt.scatter(pca_comps[:, 0], pca_comps[:, 1], c='blue', s=30)
texts = []
for name, x, y in zip(similar, pca_comps[:, 0], pca_comps[:, 1]):
    texts.append(plt.text(x, y, name, fontsize=8))

plt.scatter(pca_comps[kevin_index, 0], pca_comps[kevin_index, 1], c='red', s=100)

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

plt.savefig("../outputs/kevin-neighbors-pca.png", dpi=100, bbox_inches="tight")
plt.close()
mlflow.log_artifact(
    "../outputs/kevin-neighbors-pca.png",
    artifact_path="visualizations"
)



# In[37]:


mlflow.end_run()


# In[ ]:




