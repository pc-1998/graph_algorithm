import networkx as nx
from utils import *
import random
import numpy as np

def dominate(G):

    def prune(g):
        nonlocal min_dist, minT
        copy = g.copy()
        try:
            cycles = nx.find_cycle(g)
            for cycle in cycles:
                e = random.choice(cycle)
                copy.remove_edge(e[0], e[1])
                if is_valid_network(G, copy):
                    d = average_pairwise_distance_fast(copy)
                    if d < min_dist:
                        min_dist = d
                        minT = copy.copy()
                    prune(copy)
        except:
            e = random.choice(g.edges())
            copy.remove_edge(e[0], e[1])
            if is_valid_network(G, copy):
                d = average_pairwise_distance_fast(copy)
                if d < min_dist:
                    min_dist = d
                    minT = copy.copy()
                prune(copy)


    S = nx.dominating_set(G)
    edges_between = []
    for e in G.edges():
        if e[0] in S and e[1] in S:
            edges_between.append
    T = nx.Graph()
    for node in S:
        T.add_node(node)
    for e in edges_between:
        T.add_edge(e[0], e[1])
    if is_valid_network(G, T):
        minT = T.copy()
    else:
        minT = nx.minimum_spanning_tree(G)
    min_dist = average_pairwise_distance(minT)
    for _ in range(300):
        prune(T)
    return minT