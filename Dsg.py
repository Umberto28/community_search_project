import pandas as pd
import math
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# Read the file into a DataFrame
file_path = "ProcessedData.txt"
columns = ["source", "target", "timeStamp"]
df = pd.read_csv(file_path, sep=" ", header=None, names=columns)

# Construct the directed graph
G = nx.DiGraph()
edges = list(zip(df["source"], df["target"], df["timeStamp"]))
G.add_weighted_edges_from(edges)

def FWDDS(G, n, c):
    edges= G.edges()
    vertices= G.nodes()
    alpha=pd.DataFrame(index= edges, columns=range(n))
    beta=pd.DataFrame(index= edges, columns=range(n))
    r=pd.DataFrame(index= vertices, columns=["alpha","beta"])
    for e in edges:
        alpha.loc[e, 0]=0.5
        beta.loc[e, 0]=0.5
    for e in edges:
        r.at[e[0], "alpha"]=2*math.sqrt(c)*sum(alpha[0])
        r.at[e[1], "beta"]=2/math.sqrt(c)*sum(beta[0])
    for i in range(n):
        gamma=2/(i+2)
        for e in edges:
            a = 0
            b = 0

            if(r.at[e[0], "alpha"] < r.at[e[1], "beta"] or r.at[e[0], "alpha"] == r.at[e[1], "beta"] and c<1):
                a=1
            if(r.at[e[0], "alpha"]>r.at[e[1], "beta"] or r.at[e[0], "alpha"]==r.at[e[1], "beta"] and c>=1):
                b=1

            alpha.at[e, i+1]=(i-gamma)*alpha.at[e, i]+a*gamma
            beta.at[e, i+1]=(i-gamma)*beta.at[e, i]+b*gamma

        for e in edges:
            r.at[e[0], "alpha"]=2*math.sqrt(c)*(alpha[i].sum())
            r.at[e[1], "beta"]=2/math.sqrt(c)*(beta[i].sum())

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
    source = 's'
    sink = 't'
    eF=nx.DiGraph()
    E_st = 0

    for u in s:
        eF.add_edge(source, u, capacity=G.degree(u))
        
    for v in t:
        eF.add_edge(v, sink, capacity=G.degree(v))
        
    for u in s:
        for v in t:
            if G.has_edge(u,v):
                w=G[u][v].get("weight",1)
                eF.add_edge(u, v, capacity=w)
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
    
        
def CPApprox(G, cL, cR, e, n):

    c=(cL+cR)/2
    f=True

    while(f):
        r, alpha, beta = FWDDS(G, n, c)
        if e>0:
            sC, tC, cO, cP, f = appCDDS(G,r,e,c)
        else:
            sC, tC, cO, cP, f = extCDDS(G,r,alpha,beta,c)

    E_st=0        
    for edge in G.edges():
        if(edge[0] in sC and edge[1] in tC):
            E_st+=1

    den=E_st/math.sqrt(len(sC)*len(tC))
    denS=0
    if den>denS:
        denS=den
        D=[sC+tC]
    if cL<cO:
        s, t = CPApprox(G, cL, cO, e)
        den=E_st/math.sqrt(len(sC)*len(tC))
        if den>denS:
            denS=den
            D=[s+t]
    if cP<cR:
        s, t = CPApprox(G, cO, cR, e)
        den=E_st/math.sqrt(len(sC)*len(tC))
        if den>denS:
            denS=den
            D=[s+t]
    return D


D = CPApprox(G, 2, 2, 1, 4)
densest_subgraph = nx.DiGraph()
densest_subgraph.add_edges_from(D)
print("==============================================")
print(densest_subgraph.number_of_nodes())
print(densest_subgraph.number_of_edges())
with open('densest_subgraph.txt', 'w') as file:
    file.write(densest_subgraph.number_of_nodes())
    file.write(densest_subgraph.number_of_edges())
    

