import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

PATH = "ProcessedCollegeMsg.txt"
COL = ["source", "target", "timeStamp"]
K = [3, 4, 5, 6]

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

    graph = nx.MultiDiGraph()
    static_graph = nx.DiGraph()
    edges = list(zip(df["source"], df["target"], df["timeStamp"]))
    graph.add_edges_from(edges)
    static_graph.add_edges_from(edges)
    
    # Print metrics
    analytics_file.write('---------------------- ORIGINAL ----------------------')
    calculate_metrics(graph, 0, None, analytics_file)
    analytics_file.write('\n---------------------- STATIC ----------------------')
    calculate_metrics(static_graph, 0, None, analytics_file)
    
    return graph, static_graph

def calculate_metrics(graph: nx.MultiDiGraph | nx.DiGraph | nx.Graph, metrics: int, k: int | None, analytics_file):
    ''' 
    Function to calculate some graph metrics analyzing the structure,
    basing on the programe stage (original graph, decomposed or k-trusses)

    input:
        graph: the considered directed graph
        metrics: define what type of result graph is considered
        k: used for decomposition and k-truss
        analytics_file: txt file where store analytics about graphs and algorithms results
    '''
    #Basic Metrics
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    average_degree = sum(dict(graph.degree()).values()) / num_nodes
    # degree_distribution = [d for n, d in graph.degree()]
    
    if metrics == 0:
        analytics_file.write('\n---------------------- BASIC METRICS ----------------------\n')
    elif metrics == 1:
        analytics_file.write(f"\n------ {k+1}-CLASS ------\n")
    elif metrics == 2:
        analytics_file.write(f"\n------ {k}-TRUSS ------\n")
    
    analytics_file.write(f"Number of Nodes: {num_nodes}\n")
    analytics_file.write(f"Number of Edges: {num_edges}\n")
    analytics_file.write(f"Graph Density: {nx.density(graph)}\n")
    analytics_file.write(f"Average Degree: {average_degree}\n")
    # analytics_file.write(f"Degree Distribution: {degree_distribution}\n")

def compute_support(graph: nx.DiGraph):
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


def decomposition(graph: nx.DiGraph, k: int):
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

def truss_decomposition(graph: nx.DiGraph, analytics_file):
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
        calculate_metrics(k_class[i], 1, i, analytics_file)
    
    return k_class

def k_trusses(graph: nx.DiGraph, k_list: list, analytics_file):
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
        peripheral_comunications = analyze_peripheral_comunication(graph, k, truss)
        peripheral_comunications_nx = analyze_peripheral_comunication_nx(graph, truss)
        
        calculate_metrics(truss, 2, k, analytics_file)
        analytics_file.write(f"Number of Peripheral Comunications (k-1 truss): {peripheral_comunications}\n")
        analytics_file.write(f"Number of Peripheral Comunications (NX): {peripheral_comunications_nx}\n")
        
        trusses.append(truss)
    
    return trusses
    
def analyze_peripheral_comunication(graph: nx.Graph, k: int, truss: nx.Graph):
    ''' 
    Function to compute the peripheral comunications, given by all the edges,
    that have one node in the (k-1)-truss nodes set and the other in the k-truss nodes set or viceversa

    input:
        graph: the considered directed graph
        k: current k parameter
        truss: the calculated k-truss
    '''
    k_minus_1_truss = nx.k_truss(graph, k-1)
    peripherial = set(k_minus_1_truss.nodes()) - set(truss.nodes()) 
    communications = 0
    for edge in graph.edges():
        u,v = edge
        if (u in truss.nodes() and v in peripherial) or (v in truss.nodes() and u in peripherial):
            communications += 1

    return communications

def analyze_peripheral_comunication_nx(graph: nx.Graph, truss: nx.Graph):
    ''' 
    Function to compute the peripheral comunications, given by all the edges
    that have one node in the peripheral nodes set (considered by nx.periphery function)
    and the other in the k-truss nodes set or viceversa

    input:
        graph: the considered directed graph
        truss: the calculated k-truss
    '''
    truss_nodes = set(truss.nodes())
    peripheral_nodes = []
    
    # If the graph isn't connected the periphery function doesn't work, so it's considered the periphery of each component in the same list of nodes
    if not nx.is_connected(graph):
        for component in nx.connected_components(graph):
            subgraph = graph.subgraph(component).copy()
            peripheral_nodes.extend(nx.periphery(subgraph))
    else:
        peripheral_nodes = nx.periphery(graph)
    
    peripheral_nodes = set(peripheral_nodes) - truss_nodes

    communications = 0
    for u, v in graph.edges():
        if (u in truss_nodes and v in peripheral_nodes) or (u in peripheral_nodes and v in truss_nodes):
            communications += 1
    
    return communications

def visualize_graph(graph: nx.Graph | nx.DiGraph | nx.MultiDiGraph, output_file: str):
    plt.figure(figsize=(50, 50))
    
    # pos = nx.spring_layout(graph, k=0.1, iterations=50)
    pos = nx.kamada_kawai_layout(graph)
    
    nx.draw_networkx(graph, pos, node_size=100, node_color='red', edge_color='gray', font_size=0)
    # edge_labels = nx.get_edge_attributes(graph, 'timestamp')
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, font_color='red', bbox=None)
    plt.title(output_file.upper(), fontsize=25)
    plt.savefig('./graph_plots/' + output_file + '.png', format='PNG')  # Save the figure to a file
    plt.close()

def main():
    with open('graph_analytics.txt', 'w') as file:
        # Graphs creation and algorithms execution
        original_G, G  = create_graph(PATH, COL, file)
        decom_graph = truss_decomposition(G.copy(), file)
        trusses = k_trusses(G.copy(), K, file)
    
    # Result graphs visualization
    visualize_graph(G, 'original_graph')
    visualize_graph(original_G, 'static_graph')
    visualize_graph(decom_graph[len(decom_graph) - 1], 'decomposed_graph')
    for i in range(len(trusses)):
        visualize_graph(trusses[i], f'{i+3}-truss')

if __name__ == '__main__':
    main()