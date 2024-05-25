import pandas as pd
import math
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

def FWDDS(G, n, c):
    edges=[G.edges]
    vertices=[G.node]
    alpha={}
    beta={}
    rA={}
    rB={}
    for e in edges:
        if alpha[0]==None:
            alpha[0]=[0.5]
        else:
            alpha[0].append(0.5)
        if beta[0]==None:
            beta[0]=[0.5]
        else:
            beta[0].append(0.5)
    for e in edges:
        
        rA[0]=2*math.sqrt(c)*sum(alpha[0])
        
        

def CPApprox(G, cL, cR, e, n):
    c=(cL+cR)/2
    f=False
    while(f):
        r, alpha, beta = FWDDS(G, n, c)
        if e>0:
            sC, tC, c0, cP, f = appCDDS(G,r,e,c)
        else:
            sC, tC, c0, cP, f = extCDDS(G,r,alpha,beta,c)
    E_st= #number of edge starti in s and end in t
    den=E_st/math.sqrt(sC.size()*tC.size())
    denS=0
    if den>denS:
        denS=den
        D={[sC],[tC]}
    if cL<c0:
        s, t = CPApprox(G, cL, c0, e)
        den=E_st/math.sqrt(sC.size()*tC.size())
        if den>denS:
            denS=den
            D={[s],[t]}
    if cP<cR:
        s, t = CPApprox(G, c0, cR, e)
        den=E_st/math.sqrt(sC.size()*tC.size())
        if den>denS:
            denS=den
            D={[s],[t]}
    return D
        