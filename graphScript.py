import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

PATH = "ProcessedData.txt"
COL = ["source", "target", "timeStamp"]
K = [3, 4, 5]

def convert_to_dict(timestamp_str):
    key, value = timestamp_str.split('=', 1)
    return {key: value}

def create_graph(file_path: str, columns: list, analytics_file):
    ''' 
    Function to read the dataset txt into a DataFrame and convert it to a networkx directed graph,
    in order to simple handle data and implement algorithms

    input:
        file_path: string that represent the path of the dataset text file
        column: list of dataset columns names
        analytics_file: txt file where store analytics about graphs and algorithms results
    '''
    df = pd.read_csv(file_path, sep=" ", header=None, names=columns)
    df['timeStamp'] = df['timeStamp'].apply(convert_to_dict)
    analytics_file.write(f'{df}\n')

    graph = nx.MultiDiGraph()
    static_graph = nx.DiGraph()
    # edges = list(zip(df["source"], df["target"], df["timeStamp"]))
    # graph.add_weighted_edges_from(edges)
    edges = list(zip(df["source"], df["target"], df["timeStamp"]))
    graph.add_edges_from(edges)
    static_graph.add_edges_from(edges)
    
    # Print metrics
    calculate_basic_metrics(graph, analytics_file)
    
    return static_graph

def calculate_basic_metrics(graph: nx.MultiDiGraph, analytics_file):
    ''' 
    Function to calculate some basic graph metrics analyzing the structure

    input:
        graph: the considered directed graph
        analytics_file: txt file where store analytics about graphs and algorithms results
    '''
    #Basic Metrics
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    average_degree = sum(dict(graph.degree()).values()) / num_nodes
    degree_distribution = [d for n, d in graph.degree()]
    analytics_file.write('\n---------------------- BASIC METRICS ----------------------\n')
    analytics_file.write(f"Number of nodes: {num_nodes}\n")
    analytics_file.write(f"Number of edges: {num_edges}\n")
    analytics_file.write(f"Graph density: {nx.density(graph)}")
    analytics_file.write(f"Average degree: {average_degree}\n")
    analytics_file.write(f"Degree distribution: {degree_distribution}\n")

def compute_support(graph: nx.MultiDiGraph):
    ''' 
    Function to compute the triangle support for each graph node
    (count how many triangle contains a specific edge),
    useful for the truss decomposition

    input:
        graph: the considered directed graph
    '''
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


def decomposition(graph: nx.MultiDiGraph, k: int):
    ''' 
    Function to execute one iteration of edges and nodes removing for the truss decomposition,
    based on the computed triangle support

    input:
        graph: the considered directed graph
        k: the current iteration k parameter
    '''
    support = compute_support(graph)
    
    edges_to_remove = [edge for edge, sup in support.items() if sup < (k-2)]
    if not edges_to_remove:
        return graph
    graph.remove_edges_from(edges_to_remove)
    
    nodes_to_remove = [node for node in list(graph.nodes) if graph.degree(node) == 0]
    graph.remove_nodes_from(nodes_to_remove)
    return graph

def truss_decomposition(graph: nx.MultiDiGraph, analytics_file):
    ''' 
    Function to execute the truss decomposition of the graph,
    starting with k=3 and increase it after each iteration

    input:
        graph: the considered directed graph
        analytics_file: txt file where store analytics about graphs and algorithms results
    '''
    k=3
    k_class = []
    
    while graph.number_of_edges() > 0:
        k_truss = decomposition(graph.copy(), k)
        if k_truss.number_of_edges() > 0:
            k_class.append(k_truss)
        k += 1
        graph = k_truss

    analytics_file.write('\n---------------------- TRUSS DECOMPOSITION ----------------------\n')
    for i in range(len(k_class)):
        analytics_file.write(f"{i+1}-class: {k_class[i].number_of_nodes()} nodes; {k_class[i].number_of_edges()} edges;\n")
    
    return k_class

def k_trusses(graph: nx.MultiDiGraph, k_list: list, analytics_file):
    ''' 
    Function to execute different k-trusses on the graph, based on a k parameters list

    input:
        graph: the considered directed graph
        k_list: list of k parameters for k-trusses
        analytics_file: txt file where store analytics about graphs and algorithms results
    '''
    graph = graph.to_undirected()
    trusses = []
    
    analytics_file.write('\n---------------------- K-TRUSSES ----------------------\n')
    for k in k_list:
        truss = nx.k_truss(graph, k)
        average_degree = sum(dict(truss.degree()).values()) / truss.number_of_nodes()
        periphereal_comunications = analyze_periphereal_comunication(graph, k, truss)
        analytics_file.write(f"{k}-truss: {truss.number_of_nodes()} nodes; {truss.number_of_edges()} edges (number of internal messages); density = {nx.density(truss)}; avg degree = {average_degree}; number of periphereal comunications = {periphereal_comunications};\n")
        trusses.append(truss)
    
    return trusses
    
def analyze_periphereal_comunication(graph, k, truss):
    ''' 
    Function to compute the periphereal comunications, given by all the edges,
    that have one node in the periphereal nodes set and the other in the central nodes set or viceversa

    input:
        graph: the considered directed graph
        k: current k parameter
        truss: the calculated k-truss
    '''
    k_minus_1_truss = nx.k_truss(graph, k-1)
    peripherial = set(k_minus_1_truss.nodes()) - set(truss.nodes()) 
    peripherial_edges = []
    for edge in graph.edges():
        u,v = edge
        if (u in truss.nodes() and v in peripherial) or (v in truss.nodes() and u in peripherial):
            peripherial_edges.append(edge)

    return len(peripherial_edges)

def visualize_graph(graph: nx.MultiGraph | nx.MultiDiGraph, output_file: str):
    plt.figure(figsize=(20, 20))
    # pos = nx.spring_layout(graph, k=0.1, iterations=50)
    pos = nx.kamada_kawai_layout(graph)
    nx.draw_networkx(graph, pos, node_size=50, node_color='red', edge_color='gray', font_size=0)
    # edge_labels = nx.get_edge_attributes(graph, 'timestamp')
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, font_color='red', bbox=None)
    plt.title(output_file.upper())
    plt.savefig('./graph_plots/' + output_file + '.png', format='PNG')  # Save the figure to a file
    plt.close()

def main():
    with open('graph_analytics.txt', 'w') as file:
        # Graphs creation and algorithms execution
        G = create_graph(PATH, COL, file)
        decom_graph = truss_decomposition(G.copy(), file)
        trusses = k_trusses(G.copy(), K, file)
    
    # Result graphs visualization
    visualize_graph(G, 'original_graph')
    visualize_graph(decom_graph[len(decom_graph) - 1], 'decomposed_graph')
    for i in range(len(trusses)):
        visualize_graph(trusses[i], f'{i+3}-truss')

if __name__ == '__main__':
    main()