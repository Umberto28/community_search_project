import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

PATH = "./dataset/ProcessedCollegeMsg.txt"
COL = ["source", "target", "timeStamp"]
K = [3, 4, 5, 6, 7]

def convert_to_dict(timestamp_str):
    key, value = timestamp_str.split('=', 1)
    return {key: value}

def create_graph(file_path: str, columns: list, analytics_file):
    ''' 
    Function to read the dataset txt into a DataFrame and convert it to a networkx directed graph,
    in order to simple handle data and implement algorithms

    input:
        file_path: string that represent the path of the dataset text file
        columns: list of dataset columns names
        analytics_file: log file where store analytics about graphs and algorithms results
    
    output:
        graph: the networkx representation of the input graph (considering multi edges)
        static_graph: the networkx static representation of the input graph (without multi edges)
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
    calculate_metrics(graph, 'create', None, analytics_file)
    analytics_file.write('\n---------------------- STATIC ----------------------')
    calculate_metrics(static_graph, 'create', None, analytics_file)
    
    return graph, static_graph

def calculate_metrics(graph: nx.MultiDiGraph | nx.DiGraph | nx.Graph, algorithm: str, k: int | None, analytics_file):
    ''' 
    Function to calculate some graph's metrics, analyzing the structure
    based on the current considered algorithm (original graph, decomposed, k-trusses or dsg)

    input:
        graph: the considered graph
        algorithm: the algorithm where the considered graph come from
        k: value used for decomposition and k-truss
        analytics_file: log file where store analytics about graphs and algorithms results
    '''
    #Basic Metrics
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    average_degree = sum(dict(graph.degree()).values()) / num_nodes
    
    if algorithm == 'create':
        analytics_file.write('\n---------------------- BASIC METRICS ----------------------\n')
    elif algorithm == 'decomposition':
        analytics_file.write(f"\n------ {k+1}-CLASS ------\n")
    elif algorithm == 'k-truss':
        analytics_file.write(f"\n------ {k}-TRUSS ------\n")
    elif algorithm == 'densest':
        analytics_file.write('\n------ DENSEST SUBGRAPH ------\n')
    else:
        analytics_file.write(f"\nNo Algorithm selected!\n")
    
    analytics_file.write(f"Number of Nodes: {num_nodes}\n")
    analytics_file.write(f"Number of Edges: {num_edges}\n")
    analytics_file.write(f"Graph Density: {nx.density(graph)}\n")
    analytics_file.write(f"Average Degree: {average_degree}\n")
    if nx.is_directed(graph):
        analytics_file.write(f"Diameter: {nx.diameter(graph) if nx.is_strongly_connected(graph) else 'not a connected graph!'}\n")
    else:
        analytics_file.write(f"Diameter: {nx.diameter(graph) if nx.is_connected(graph) else 'not a connected graph!'}\n")

def compute_support(graph: nx.DiGraph):
    ''' 
    Function to compute the triangle support for each graph's node
    (count how many triangle contains a specific edge), useful for the truss decomposition

    input:
        graph: the considered graph

    output:
        support: dict with all nodes support values
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
        graph: the considered graph
        k: the current iteration k parameter
        
    output:
        graph: the k-decomposed graph
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
    starting with k=3, removing edges with low triangle support (based on k value)
    and increase the k value after each iteration

    input:
        graph: the considered graph
        analytics_file: log file where store analytics about graphs and algorithms results
        
    output:
        k_class: list of decomposed graphs, one for each iteration
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
        calculate_metrics(k_class[i], 'decomposition', i, analytics_file)
    
    return k_class

def k_trusses(graph: nx.DiGraph, k_list: list, analytics_file):
    ''' 
    Function to execute different k-trusses on the graph, based on a k parameters' list

    input:
        graph: the considered graph
        k_list: list of k parameters for k-trusses
        analytics_file: log file where store analytics about graphs and algorithms results
        
    output:
        trusses: list of different k-trusses result graphs
    '''
    graph = graph.to_undirected()
    periphery = find_periphery(graph)
    trusses = []
    
    analytics_file.write('\n---------------------- K-TRUSSES ----------------------\n')
    for i in range(len(k_list)):
        truss = nx.k_truss(graph, k_list[i])
        peripheral_comunications = analyze_peripheral_comunication(graph, k_list[i], truss, k_minus_1_truss=None if i == 0 else trusses[i-1])
        peripheral_comunications_nx = analyze_peripheral_comunication_nx(graph, truss, periphery)
        
        calculate_metrics(truss, 'k-truss', k_list[i], analytics_file)
        analytics_file.write(f"Number of Peripheral Comunications (k-1 truss): {peripheral_comunications}\n")
        analytics_file.write(f"Number of Peripheral Comunications (NX): {peripheral_comunications_nx}\n")
        
        trusses.append(truss)
    
    return trusses
    
def analyze_peripheral_comunication(graph: nx.Graph, k: int, truss: nx.Graph, k_minus_1_truss: nx.Graph | None):
    ''' 
    Function to compute the peripheral comunications, given by all the edges,
    that have one node in the (k-1)-truss nodes set and the other in the k-truss nodes set or viceversa

    input:
        graph: the considered graph
        k: current k parameter
        truss: the calculated k-truss
        k_minus_1_truss: the calculated (k-1)-truss, if available
        
    output:
        communications: the number of messages sent between the k-truss and the periphery (k-1)truss
    '''
    if k_minus_1_truss == None:
        k_minus_1_truss = nx.k_truss(graph, k-1)
    
    peripherial = set(k_minus_1_truss.nodes()) - set(truss.nodes()) 
    communications = 0
    
    for edge in graph.edges():
        u,v = edge
        if (u in truss.nodes() and v in peripherial) or (v in truss.nodes() and u in peripherial):
            communications += 1

    return communications

def find_periphery(graph: nx.Graph):
    ''' 
    Function to find the graph periphery, considering every connected components for not connected graphs

    input:
        graph: the considered graph

    output:
        peripheral_nodes: list of the peripheral nodes
    '''
    peripheral_nodes = []
    
    # If the graph isn't connected the periphery function doesn't work, so it's considered the periphery of each component in the same list of nodes
    if not nx.is_connected(graph):
        for component in nx.connected_components(graph):
            subgraph = graph.subgraph(component).copy()
            peripheral_nodes.extend(nx.periphery(subgraph))
    else:
        peripheral_nodes = nx.periphery(graph)
    
    return peripheral_nodes

def analyze_peripheral_comunication_nx(graph: nx.Graph, truss: nx.Graph, peripheral_nodes):
    ''' 
    Function to compute the peripheral comunications, given by all the edges
    that have one node in the peripheral nodes set (considered by nx.periphery function)
    and the other in the k-truss nodes set or viceversa

    input:
        graph: the considered graph
        truss: the calculated k-truss
        peripheral_nodes: list of the peripheral nodes, obtained by the nx function

    output:
        communications: the number of messages sent between the k-truss and the periphery (k-1)truss
    '''
    truss_nodes = set(truss.nodes())    
    peripheral_nodes = set(peripheral_nodes) - truss_nodes

    communications = 0
    for u, v in graph.edges():
        if (u in truss_nodes and v in peripheral_nodes) or (u in peripheral_nodes and v in truss_nodes):
            communications += 1
    
    return communications

def visualize_graph(graph: nx.Graph | nx.DiGraph | nx.MultiDiGraph, output_file: str):
    ''' 
    Function to create graphs' plots with some information charts and save them into images

    input:
        graph: the considered graph
        output_file: the final image path where to save thw plot
    '''
    fig = plt.figure(figsize=(55, 70))
    
    # Create a gridspec with 2 rows and 2 columns, width ratio of 2:1 for the columns
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    
    ax_main = fig.add_subplot(gs[0, :])
    pos = nx.spring_layout(graph, k=0.1, iterations=40)
    nx.draw_networkx(graph, pos, node_size=90, node_color='red', ax=ax_main)
    ax_main.set_title(output_file.upper(), fontsize=50)

    # First additional subplot: Node Degree Distribution
    ax_deg = fig.add_subplot(gs[1, 0])
    degrees = [graph.degree(n) for n in graph.nodes()]
    ax_deg.bar(*np.unique(degrees, return_counts=True))
    ax_deg.set_title('Node Degree Distribution', fontsize=50)
    ax_deg.set_xlabel('Degree', fontsize=40)
    ax_deg.set_ylabel('# of Nodes', fontsize=40)

    # Second additional subplot: Edge Betweenness Centrality
    ax_bet = fig.add_subplot(gs[1, 1])
    edge_betweenness = nx.edge_betweenness_centrality(graph)
    betweenness_values = list(edge_betweenness.values())
    ax_deg.bar(*np.unique(betweenness_values, return_counts=True))
    ax_bet.set_title('Edge Betweenness Centrality Distribution', fontsize=50)
    ax_bet.set_xlabel('Betweenness Centrality', fontsize=40)
    ax_bet.set_ylabel('Frequency', fontsize=40)

    plt.tight_layout()
    plt.savefig('./graph_plots/' + output_file + '.png', format='PNG')
    plt.close()

def main():
    with open('./output_log/community_search.txt', 'w') as file:
        start_time = time.time()
        
        original_G, static_G  = create_graph(PATH, COL, file)
        decom_graph = truss_decomposition(static_G.copy(), file)
        trusses = k_trusses(static_G.copy(), K, file)
        
        end_time = time.time()
        execution_time = end_time - start_time
        file.write(f'\nEXECUTION TIME: {execution_time:.2f} s')

    visualize_graph(original_G, 'original_graph')
    visualize_graph(static_G, 'static_graph')
    visualize_graph(decom_graph[len(decom_graph) - 1], 'truss/decomposed_graph')
    for i in range(len(trusses)):
        visualize_graph(trusses[i], f'truss/{i+3}-truss')

if __name__ == '__main__':
    main()