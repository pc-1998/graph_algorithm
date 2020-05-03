import itertools
import networkx as nx
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
from solver import solve

def bruteforce(G):
    def findChain(G):
        nonlocal confirmed
        nonlocal T
        nonlocal G_copy
        changed = False
        nodes = [n for n in G.nodes()]
        for v in nodes:
            if G.degree(v) == 1:
                u = list(G.neighbors(v))[0]
                confirmed.append(u)
                T.add_node(u)
                if u in confirmed and v in confirmed:
                    T.add_edge(u, v, weight=G[u][v]['weight'])
                    G_copy.remove_edge(u,v)
                G.remove_node(v)
                changed = True
        if changed:
            findChain(G)

    def remove(copy, G_copy, k):
        nonlocal min_dist
        nonlocal minT
        has_valid = False
        count = 0
        for c in itertools.combinations(G_copy.edges(), k):
            temp = copy.copy()
            if count >= 10000:
                return solve(G)
            count += 1
            for e in c:
                u, v = e[0], e[1]
                if copy.degree(u) == 1:
                    copy.remove_node(u)
                elif copy.degree(v) == 1:
                    copy.remove_node(v)
                else:
                    copy.remove_edge(u, v)
            if is_valid_network(G, copy):
                has_valid = True
                if average_pairwise_distance_fast(copy) < min_dist:
                    min_dist = average_pairwise_distance_fast(copy)
                    minT = copy.copy()
            elif nx.is_dominating_set(G, copy.nodes()) and nx.is_connected(copy):
                has_valid = True
            copy = temp
        if not has_valid:
            return
        remove(copy, G_copy, k+1)
    confirmed = []
    T = nx.Graph()
    G_copy = G.copy()
    findChain(G.copy())
    if T.number_of_nodes() != 0 and is_valid_network(G, T):
        min_dist = average_pairwise_distance_fast(T)
        minT = T.copy()
    else:
        min_dist = average_pairwise_distance_fast(G)
        minT = G.copy()
    # for v in G.nodes():
    #     if G.degree(v) == 1:
    #         confirmed.append(list(G.neighbors(v))[0])
    #remove(G.copy())
    # T = nx.Graph()
    remove(G.copy(), G_copy, 1)
    return minT