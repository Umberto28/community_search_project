import math
import time
import pandas as pd
import networkx as nx
from community_search import convert_to_dict, calculate_metrics, visualize_graph

PATH = "./dataset/ProcessedCollegeMsg.txt"
COL = ["source", "target", "timeStamp"]

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

    static_graph = nx.DiGraph()
    edges = list(zip(df["source"], df["target"], df["timeStamp"]))
    static_graph.add_edges_from(edges)
    
    return static_graph

def FWDDS(G, n, c):
    edges = list(G.edges())
    vertices = list(G.nodes())
    
    alpha = pd.DataFrame(0.5, index=edges, columns=range(n+1))
    beta = pd.DataFrame(0.5, index=edges, columns=range(n+1))
    r = pd.DataFrame(index=vertices, columns=["alpha", "beta"])
    
    sqrt_c = math.sqrt(c)
    inv_sqrt_c = 2 / sqrt_c
    for i in range(n):
        gamma=2/(i+2)
        for e in edges:
            r.at[e[0], "alpha"]=2*sqrt_c*(alpha[i].sum())
            r.at[e[1], "beta"]=inv_sqrt_c*(beta[i].sum())
            
        for e in edges:
            
            a = int(r.at[e[0], "alpha"] < r.at[e[1], "beta"] or 
                    (r.at[e[0], "alpha"] == r.at[e[1], "beta"] and c < 1))
            b = int(r.at[e[0], "alpha"] > r.at[e[1], "beta"] or 
                    (r.at[e[0], "alpha"] == r.at[e[1], "beta"] and c >= 1))

            alpha.at[e, i+1]=(i-gamma)*alpha.at[e, i]+a*gamma
            beta.at[e, i+1]=(i-gamma)*beta.at[e, i]+b*gamma
            
            

    return r, alpha[n], beta[n]

def appCDDS(G, r, e, c):
    verticles = G.nodes()
    denS=0
    den=0
    sS=[]
    tS=[]
    sC=[]
    tC=[]
    L=[]
    R=[]
    
    for u in verticles:
        if pd.isnull(r.loc[u, "beta"]):
            L.append(u)
        else:
            R.append(u)

    L.sort()
    R.sort()
    r2=r.index.tolist()

    for i in r2:
        if i in L:
            sC.append(i)
        else:
            tC.append(i)

        if len(sC)==0 or len(tC)==0:
            continue
        
        E_st = 0
        for edge in G.edges():
            if(edge[0] in sC and edge[1] in tC):
                E_st += 1 

        den=E_st/math.sqrt(len(sC)*len(tC))

        if den>denS:
            denS=den
            sS=sC
            tS=tC

    cO=len(sS)/len(tS)
    cP=c**2/cO

    if cO > cP:
        temp=cO
        cO=cP
        cP=temp

    sigma=r2[0]/denS
    if sigma<=math.sqrt(1+e):
        return (sS, tS, min(cO, c/(1+e)), max(cP, c*(1+e)), True)
    elif sigma<=(1+e) and cO<c/(1+e) and c*(1+e)<cP :
        return (sS, tS, cO, cP, False)
    else:
        return(sS, tS, cO, cP, False)
    
def isStable(G, alpha, beta,r):
    edges = G.edges()
    return (r[-1]>r[0] and (alpha.at[e]==0 or beta.at[e]==0 for e in edges))

def isCDDS(G, s, t, c):
    source = -1
    sink = -2
    eF=nx.DiGraph()
    E_st = 0

    for u in s:
        eF.add_edge(source, u, capacity=G.degree(u))
        
    for v in t:
        eF.add_edge(v, sink, capacity=G.degree(v))
        
    for u in s:
        for v in t:
            if G.has_edge(u,v):
                eF.add_edge(u, v, capacity=2)
    f, fD = nx.maximum_flow(eF, source, sink)

    for e in G.edges():
        if(e[0] in s and e[1] in t):
            E_st += 1

    return f == E_st

def extCDDS(G, r, alpha, beta, c):
    verticles = G.nodes()
    denS=0
    den=0
    sS=[]
    tS=[]
    sC=[]
    tC=[]
    L=[]
    R=[]
    
    for u in verticles:
        if pd.isnull(r.loc[u, "beta"]):
            L.append(u)
        else:
            R.append(u)

    L.sort()
    R.sort()
    r2=r.index.tolist()

    for i in r2:
        if i in L:
            sC.append(i)
        else:
            tC.append(i)

        if len(sC)==0 or len(tC)==0:
            continue
        
        E_st = 0
        for edge in G.edges():
            if(edge[0] in sC and edge[1] in tC):
                E_st += 1 

        den=E_st/math.sqrt(len(sC)*len(tC))

        if den>denS:
            denS=den
            sS=sC
            tS=tC

    cO=len(sS)/len(tS)
    cP=c**2/cO

    if cO > cP:
        temp=cO
        cO=cP
        cP=temp

    if isStable(G, alpha, beta, r2):
        if isCDDS(G, sS, tS, c):
            return(sS, tS, cO, cP, True)
        
        GS=nx.DiGraph()
        GS.add_weighted_edges_from(sS+tS)
        G=GS

    return(sS, tS, cO, cP, False)
    
        
def CP_DDS(G, cl, cr, e, n):
    c = (cl + cr) / 2

    while True:
        r, alpha, beta = FWDDS(G, n, c)
        if e > 0:
            sC, tC, cO, cP, f = appCDDS(G, r, e, c)
        else:
            sC, tC, cO, cP, f = extCDDS(G, r, alpha, beta, c)

        if f:
            break

    E_st = 0
    for edge in G.edges():
        if edge[0] in sC and edge[1] in tC:
            E_st += 1

    den = E_st / math.sqrt(len(sC) * len(tC))
    denS = 0
    if den > denS:
        print("\n\nD created with sC + tC\n\n")
        denS = den
        D = [sC, tC]

    if cl < cO:
        s, t = CP_DDS(G, cl, cO, e, n)
        den = E_st / math.sqrt(len(s) * len(t))
        if den > denS:
            print("\n\nD created with s + t cl<c0\n\n")
            denS = den
            D = [s,t]

    if cP < cr:
        s, t = CP_DDS(G, cO, cr, e, n)
        den = E_st / math.sqrt(len(s) * len(t))
        if den > denS:
            print("\n\nD created with s + t cp<cr\n\n")
            denS = den
            D = [s, t]

    return D

def main():
    G  = create_graph(PATH, COL)
    start_time = time.time()
    
    D = CP_DDS(G, 1/G.number_of_nodes(), G.number_of_nodes(), 0, 4)
    densest_subgraph = G.subgraph(D[0]+D[1])

    end_time = time.time()
    execution_time = end_time - start_time
    
    with open('./output_log/dsg_CP_DDS.txt', 'w') as file:
        calculate_metrics(densest_subgraph, 'densest', None, file)
        file.write(f'\nEXECUTION TIME: {execution_time:.2f} s\n')
    
    visualize_graph(densest_subgraph, 'dsg/dsg_CP_DDS')

if __name__ == '__main__':
    main()