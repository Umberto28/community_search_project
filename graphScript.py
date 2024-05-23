import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# Read the file into a DataFrame
file_path = "ProcessedData.txt"
columns = ["Nodo_source", "Nodo_target", "Converted_TimeStamp"]
df = pd.read_csv(file_path, sep=" ", header=None, names=columns)

# Construct the directed graph
G = nx.DiGraph()
edges = list(zip(df["Nodo_source"], df["Nodo_target"]))
G.add_edges_from(edges)

# Compute k-trusses for k=3, 4, 5
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

# Measure densities
def density(G):
    n = len(G.nodes())
    m = len(G.edges())
    if n > 1:
        return (2 * m) / (n * (n - 1))
    else:
        return 0

densities = {k: density(k_trusses[k]) for k in k_values}
print("Densities:", densities)

# Visualize the k-trusses
pos = nx.spring_layout(G)
plt.figure(figsize=(15, 5))

for i, k in enumerate(k_values, 1):
    plt.subplot(1, len(k_values), i)
    nx.draw_networkx(G, pos, node_color='lightgray', with_labels=True, edge_color='gray')
    nx.draw_networkx_edges(k_trusses[k], pos, edge_color='blue', width=2)
    plt.title(f"k-Truss (k={k})")

plt.show()
