import time
import pandas as pd
import networkx as nx
from community_search import convert_to_dict, calculate_metrics, visualize_graph

PATH = "./dataset/ProcessedCollegeMsg.txt"
COL = ["source", "target", "timeStamp"]
T = [1, 5, 10]

def create_graph(file_path: str, columns: list):
    '''
    Function to read the dataset txt into a DataFrame and convert it to a networkx directed graph,
    in order to simple handle data and implement algorithms

    input:
        file_path: string that represent the path of the dataset text file
        columns: list of dataset columns names
        analytics_file: log file where store analytics about graphs and algorithms results
    
    output:
        dynamic_graph: the networkx representation of the input graph (considering multi edges)
        static_graph: the networkx static representation of the input graph (without multi edges)
    '''
    df = pd.read_csv(file_path, sep=" ", header=None, names=columns)
    df['timeStamp'] = df['timeStamp'].apply(convert_to_dict)

    dynamic_graph = nx.MultiDiGraph()
    static_graph = nx.DiGraph()
    edges = list(zip(df["source"], df["target"], df["timeStamp"]))
    static_graph.add_edges_from(edges)
    dynamic_graph.add_edges_from(edges)
    
    return dynamic_graph, static_graph

def charikar_algorithm(subgraph: nx.MultiDiGraph | nx.DiGraph, load: dict):
    '''
    Function to find the maximum density subgraph for a specific iteration, using Charikar's greedy algorithm.

    input:
        subgraph: the considered subgraph for a specific iteration
        load: dict of load values for all nodes

    output:
        densestsubgraph: the calculated densest subgraph for that iteration
        load: updated dict of load values for all nodes
    '''
    densestsubgraph = subgraph.copy()
    max_density = nx.density(subgraph)

    while subgraph.number_of_nodes() > 0:
        # Find the node with the lowest degree
        node_to_remove = min(subgraph.nodes(), key=lambda u: load[u] + subgraph.degree[u])
        load[node_to_remove] += subgraph.degree[node_to_remove]

        # Remove the node and its edges from the graph
        subgraph.remove_node(node_to_remove)

        # Calculate the density of the remaining graph
        current_density = nx.density(subgraph)

        # If the density increases, update the subgraph
        if current_density > max_density:
            max_density = current_density
            densestsubgraph = subgraph.copy()

    return densestsubgraph, load

def greedy_plus_plus(G: nx.MultiDiGraph | nx.DiGraph, i: int):
    '''
    Function to compute the greedy++ algorithm, in order to find the densest subgraph.
    It executes Charikar's algorithm for a specified number of iteration

    input:
        G: the considered graph
        i: number of iterations

    output:
        densest_graph: the calculated densest subgraph
    '''
    densest_graph = G.copy()
    load = {node: 0 for node in G.nodes()}
    max_density = nx.density(densest_graph)
    
    for _ in range(i):
        subgraph, load = charikar_algorithm(G, load)
        current_density = nx.density(subgraph)
        
        if current_density > max_density:
            max_density = current_density
            densest_graph = subgraph.copy()
    
    return densest_graph

def main():
    DG, SG  = create_graph(PATH, COL)
    start_time = time.time()
    with open('./output_log/dsg_greedy_plus_plus.txt', 'w') as file:
        for t in T:
            start_time = time.time()
            
            densest_d = greedy_plus_plus(DG.to_undirected(), t)
            densest_flowless_d = DG.subgraph(densest_d)
            densest_s = greedy_plus_plus(SG.to_undirected(), t)
            densest_flowless_s = SG.subgraph(densest_s)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            
            file.write(f'\n---------------------- NUMBER OF ITERATIONS: {t} ----------------------\n')
            file.write(f'\n------ ORIGINAL GRAPH ------\n')
            calculate_metrics(densest_flowless_d, 'densest', None, file)
            file.write(f'\n------ STATIC GRAPH ------\n')
            calculate_metrics(densest_flowless_s, 'densest', None, file)
            file.write(f'\nEXECUTION TIME: {execution_time:.2f} s\n')
            
            visualize_graph(densest_flowless_d, f'dsg/greedy_plus_plus_d_{t}')
            visualize_graph(densest_flowless_s, f'dsg/greedy_plus_plus_s_{t}')

if __name__ == '__main__':
    main()