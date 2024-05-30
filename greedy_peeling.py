import pandas as pd
import networkx as nx
from networkx.algorithms.flow import edmonds_karp

'''
    Find the densest subgraph of a graph G. Using the greedy algorithm.
'''

PATH = "ProcessedCollegeMsg.txt"
COL = ["source", "target", "timeStamp"]

def convert_to_dict(timestamp_str):
    key, value = timestamp_str.split('=', 1)
    return {key: value}

def create_graph(file_path: str, columns: list):

    df = pd.read_csv(file_path, sep=" ", header=None, names=columns)
    df['timeStamp'] = df['timeStamp'].apply(convert_to_dict)

    graph = nx.MultiDiGraph()
    static_graph = nx.DiGraph()
    edges = list(zip(df["source"], df["target"], df["timeStamp"]))
    static_graph.add_edges_from(edges)
    graph.add_edges_from(edges)
    print(f"Number of nodes: {static_graph.number_of_nodes()} ")
    
    
    return static_graph, graph
    
# Greedy Peeling algorithm
def compute_densest_subgraph_greedyPeeling(graph):
    
    H = graph.copy()
    max_density = 0
    best_subgraph = H.copy()
    
    removal_order = []
    
    while len(H) > 0:
        num_edges = H.number_of_edges()
        num_nodes = H.number_of_nodes()
        current_density = num_edges / num_nodes if num_nodes > 0 else 0
        
        if current_density > max_density:
            max_density = current_density
            best_subgraph = H.copy()
        
        # Find the node with the smallest degree
        min_degree_node = min(H.nodes(), key=H.degree)
        
        # Remove this node from the graph
        removal_order.append(min_degree_node)
        H.remove_node(min_degree_node)
        
    return best_subgraph
        
G, dynamic_graph  = create_graph(PATH, COL)    
densest_subgraph_static = compute_densest_subgraph_greedyPeeling(G) 
densest_subgraph_dynamic = compute_densest_subgraph_greedyPeeling(dynamic_graph)

print(f"The densest subgraph in static graph has {densest_subgraph_static.number_of_nodes()} nodes and {densest_subgraph_static.number_of_edges()} edges.")
print(f"The densest subgraph in dynamic graph has {densest_subgraph_dynamic.number_of_nodes()} nodes and {densest_subgraph_dynamic.number_of_edges()} edges.")

# For Static graph: The densest subgraph has 312 nodes and 8014 edges.
# For dynamic graph: The densest subgraph has 4 nodes and 550 edges.


