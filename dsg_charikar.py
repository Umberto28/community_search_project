import pandas as pd
import networkx as nx
import copy
from community_search import convert_to_dict, calculate_metrics, visualize_graph

PATH = "./dataset/ProcessedCollegeMsg.txt"
COL = ["source", "target", "timeStamp"]

def create_graph(file_path: str, columns: list):
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

    dynamic_graph = nx.MultiDiGraph()
    static_graph = nx.DiGraph()
    edges = list(zip(df["source"], df["target"], df["timeStamp"]))
    static_graph.add_edges_from(edges)
    dynamic_graph.add_edges_from(edges)
    
    return dynamic_graph, static_graph

def charikar_algorithm(graph: nx.DiGraph):
    """
    Finds the maximum density subgraph using Charikar's greedy algorithm.

    Args:
        G: A NetworkX graph.

    Returns:
        A subgraph with the highest density.
    """
    subgraph = graph.copy()
    max_density = nx.density(graph)

    while len(graph.nodes()) > 0:
        # Find the node with the lowest degree
        node_to_remove = min(graph.nodes(), key=graph.degree)

        # Remove the node and its edges from the graph
        graph.remove_node(node_to_remove)

        # Calculate the density of the remaining graph
        current_density = nx.density(graph)
        print(current_density)

        # If the density increases, update the subgraph
        if current_density > max_density:
            max_density = current_density
            subgraph = graph.copy()

    return subgraph

def charikar_peeling(H, load):
        density = 0
        best_subgraph = None
        while len(H) > 0:
            min_node = min(H, key=lambda u: load[u] + H.degree[u])
            load[min_node] += H.degree[min_node]
            H.remove_node(min_node)
            current_density = nx.density(H)
            if current_density > density:
                density = current_density
                best_subgraph = H.copy()
        return best_subgraph, load

def greedy_plus_plus(G, T=10):
    Gdensest = G.copy()
    load = {node: 0 for node in G.nodes()}
    for _ in range(T):
        H = G.copy()
        subgraph, load = charikar_peeling(H, load)
        if nx.density(subgraph) > nx.density(Gdensest):
            Gdensest = subgraph
    return Gdensest

def main():
    DG, SG  = create_graph(PATH, COL)
    # densest_subgraph_d = charikar_algorithm(DG)
    # densest_subgraph_s = charikar_algorithm(SG)
    densest = greedy_plus_plus(SG.to_undirected(), 10)
    densest_flowless_d = DG.subgraph(densest)
    with open('./output_log/dsg_charikar.txt', 'w') as file:
        calculate_metrics(densest_flowless_d, 0, 'densest', None, file)
        # calculate_metrics(densest_subgraph_d, 0, 'densest', None, file)
        # calculate_metrics(densest_subgraph_s, 1, 'densest', None, file)
    visualize_graph(densest_flowless_d, 'densest_d')
    # visualize_graph(densest_subgraph_d, 'densest_d')
    # visualize_graph(densest_subgraph_s, 'densest_s')

if __name__ == '__main__':
    main()