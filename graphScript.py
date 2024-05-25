import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

PATH = "ProcessedData.txt"
COL = ["source", "target", "timeStamp"]
K = [3, 4, 5]

def create_graph(file_path: str, columns: list):
    # Read the file into a DataFrame
    df = pd.read_csv(file_path, sep=" ", header=None, names=columns)
    print(df)

    # Construct the directed graph
    graph = nx.DiGraph()
    edges = list(zip(df["source"], df["target"], df["timeStamp"]))
    graph.add_weighted_edges_from(edges)
    
    # Print metrics
    calculate_basic_metrics(graph)
    
    return graph

def calculate_basic_metrics(graph: nx.DiGraph):
    #Basic Metrics
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    average_degree = sum(dict(graph.degree()).values()) / num_nodes
    degree_distribution = [d for n, d in graph.degree()]
    print('\n---------------------- BASIC METRICS ----------------------')
    print("Number of nodes:", num_nodes)
    print("Number of edges:", num_edges)
    print("Graph density:", nx.density(graph))
    print("Average degree:", average_degree)
    print("Degree distribution:", degree_distribution)

    # Clustering Coefficient
    global_clustering_coefficient = nx.transitivity(graph)
    local_clustering_coefficients = nx.clustering(graph)
    average_local_clustering_coefficient = sum(local_clustering_coefficients.values()) / num_nodes
    print("Global clustering coefficient:", global_clustering_coefficient)
    print("Average local clustering coefficient:", average_local_clustering_coefficient)

def compute_support(graph: nx.DiGraph):
    support = {edge: 0 for edge in graph.edges()}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                if graph.has_edge(neighbors[i], neighbors[j]):
                    support[(node, neighbors[i])] += 1
                    support[(node, neighbors[j])] += 1
                    support[(neighbors[i], neighbors[j])] += 1
    return support

def decomposition(graph: nx.DiGraph, k: int):
    support = compute_support(graph)
    
    edges_to_remove = [edge for edge, sup in support.items() if sup < (k-2)]
    if not edges_to_remove:
        return graph
    graph.remove_edges_from(edges_to_remove)
    
    nodes_to_remove = [node for node in list(graph.nodes) if graph.degree(node) == 0]
    graph.remove_nodes_from(nodes_to_remove)
    return graph

def truss_decomposition(graph: nx.DiGraph):
    k=3
    k_class = []
    while graph.number_of_edges() > 0:
        k_truss = decomposition(graph.copy(), k)
        if k_truss.number_of_edges() > 0:
            k_class.append(k_truss)
        k += 1
        graph = k_truss

    print('\n---------------------- TRUSS DECOMPOSITION ----------------------')
    for i in range(len(k_class)):
        print(f"{i+1}-class: {k_class[i].number_of_nodes()} nodes; {k_class[i].number_of_edges()} edges;")

def k_trusses(graph: nx.DiGraph, k_list: list):
    graph = graph.to_undirected()
    trusses = []
    
    print('\n---------------------- K-TRUSSES ----------------------')
    for k in k_list:
        truss = nx.k_truss(graph, k)
        average_degree = sum(dict(truss.degree()).values()) / truss.number_of_nodes()
        print(f"{k}-truss: {truss.number_of_nodes()} nodes; {truss.number_of_edges()} edges; density = {nx.density(truss)}; avg degree = {average_degree};")
        trusses.append(truss)
    
    return trusses

G = create_graph(PATH, COL)
truss_decomposition(G.copy())
trusses = k_trusses(G.copy(), K)
