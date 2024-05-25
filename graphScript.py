import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# Read the file into a DataFrame
file_path = "ProcessedData.txt"
columns = ["source", "target", "timeStamp"]
df = pd.read_csv(file_path, sep=" ", header=None, names=columns)
print(df)

# Construct the directed graph
G = nx.DiGraph()
edges = list(zip(df["source"], df["target"], df["timeStamp"]))
G.add_weighted_edges_from(edges)

#Basic Metrics
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
average_degree = sum(dict(G.degree()).values()) / num_nodes
degree_distribution = [d for n, d in G.degree()]
print("Number of nodes:", num_nodes)
print("Number of edges:", num_edges)
print("Graph density:", nx.density(G))
print("Average degree:", average_degree)
print("Degree distribution:", degree_distribution)

# Clustering Coefficient
global_clustering_coefficient = nx.transitivity(G)
local_clustering_coefficients = nx.clustering(G)
average_local_clustering_coefficient = sum(local_clustering_coefficients.values()) / num_nodes
print("Global clustering coefficient:", global_clustering_coefficient)
print("Average local clustering coefficient:", average_local_clustering_coefficient)

# Compute k-truss
def directed_k_truss(G, k):
    H = G.copy()
    while True:
        to_remove = []
        for u, v in H.edges():
            # Count the number of triangles each edge is part of
            count = 0
            for w in set(H.predecessors(u)).intersection(H.successors(v)):
                if H.has_edge(v, w):
                    count += 1
            if count < k - 2:
                to_remove.append((u, v))
        if not to_remove:
            break
        H.remove_edges_from(to_remove)
    return H

k_values = [3, 4, 5]
k_trusses = {k: directed_k_truss(G, k) for k in k_values}

def analyze_k_truss(k_truss_subgraph):
    # Compute density
    density = nx.density(k_truss_subgraph)
    print("Density of k-truss:", density)

    # Degree distribution
    degrees = [d for n, d in k_truss_subgraph.degree()]
    print("Degree distribution:", degrees)

for k in k_values:
    analyze_k_truss(k_trusses[k])

# Visualize the k-trusses
# pos = nx.spring_layout(G)
# plt.figure(figsize=(30, 10))

# for i, k in enumerate(k_values, 1):
#     plt.subplot(1, len(k_values), i)
#     nx.draw_networkx(G, pos, node_color='lightgray', with_labels=True, edge_color='gray')
#     nx.draw_networkx_edges(k_trusses[k], pos, edge_color='blue', width=2)
#     plt.title(f"k-Truss (k={k})")
# plt.show()
