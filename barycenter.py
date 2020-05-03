import networkx as nx
from utils import *
import random
import numpy as np

def bary(G):
    def find_next():
        nonlocal T, G_copy
        average_distances = {}
        corresponding_edge = {}
        for node in T.nodes():
            for adj in G.neighbors(node):
                if adj in T.nodes(): continue
                if adj not in average_distances.keys():
                    average_distances[adj] = (sum(
                        [all_shortest_path_lengths[adj][t] for t in G_copy.nodes()]) / G_copy.number_of_nodes() +
                                              G[node][adj]['weight'])
                    corresponding_edge[adj] = (node, adj)
                else:
                    temp = (sum(
                        [all_shortest_path_lengths[adj][t] for t in G_copy.nodes()]) / G_copy.number_of_nodes() +
                                              G[node][adj]['weight'])
                    if average_distances[adj] > temp:
                        average_distances[adj] = temp
                        corresponding_edge[adj] = (node, adj)
        next = min(average_distances.keys(), key=lambda x: average_distances[x])
        T.add_node(next)
        G_copy.remove_node(next)
        T.add_edge(corresponding_edge[next][0], corresponding_edge[next][1],
                   weight=G[corresponding_edge[next][0]][corresponding_edge[next][1]]['weight'])

    all_shortest_path_lengths = dict(nx.shortest_path_length(G, weight='weight'))
    all_shortest_paths = nx.shortest_path(G, weight='weight')
    average_shortest_path_lengths = {}
    for v in all_shortest_path_lengths.keys():
        average_shortest_path_lengths[v] = sum([all_shortest_path_lengths[v][u] for u in G.nodes()]) / (
                G.number_of_nodes() - 1)
    root = min(average_shortest_path_lengths, key=lambda x: average_shortest_path_lengths[x])
    T = nx.Graph()
    T.add_node(root)
    G_copy = G.copy()
    G_copy.remove_node(root)
    while not is_valid_network(G, T):
        find_next()

    return T

    # while not is_valid_network(G, T):
    #     center = nx.barycenter(G, weight='weight')
    #     T.add_node(center)
    #     edge_list = list(G.edges(center))
    #     e = random.choices(edge_list, weights=[np.exp(-G[x[0]][x[1]]['weight']) for x in edge_list])
    #     if e[0] == center:
    #         node = e[1]
    #     else:
    #         node = e[0]
    #     T.add_node(node)
    #     T.add_edge(node, center, weight=G[node][center]['weight'])
    #     G.reove_node(center)