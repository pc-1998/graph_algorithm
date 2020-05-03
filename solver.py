# below is the method using degree heuristic to let MST pick higher-degree nodes

import networkx as nx
from parse import read_input_file, write_output_file, read_output_file
import utils
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
import os
import sys
import matplotlib.pyplot as plt
import random
import numpy as np
from bruteforce import *
from dominating_set import *
from barycenter import *

def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """
    # Special case for one central node
    for node in G.nodes():
        if G.degree(node) == G.number_of_nodes() - 1:
            T = nx.Graph()
            T.add_node(node)
            return T
    # added: weight <- weight - degree, to let MST choose higher-degree nodes
    G_copy = G.copy()
    #     for edge in G_copy.edges.data():
    #         u, v = edge[0], edge[1]
    #         edge[2]['weight'] /= np.log(1+max(G_copy.degree(u), G_copy.degree(v)))
    T = nx.minimum_spanning_tree(G_copy)
    for edge in T.edges.data():
        u, v = edge[0], edge[1]
        edge[2]['weight'] = G[u][v]['weight']

    if T.number_of_nodes() == 1:
        return T
    elif T.number_of_nodes() == 2:
        node = list(T.nodes)[0]
        T.remove_node(node)
        return T
    else:
        u_list = []
        v_list = []
        v_u_dict = {node: [] for node in T.nodes}
        for u in T.nodes:
            if T.degree(u) == 1:
                v = list(T.neighbors(u))[0]
                u_list.append(u)
                v_list.append(v)
                v_u_dict[v].append(u)
        for u in u_list:
            T.remove_node(u)
        if len(list(T.nodes)) == 1:
            return T
        v_list = list(set(v_list))
        for v in v_list:
            copy = T.copy()
            copy.remove_node(v)
            if len(list(copy.nodes)) == 0:
                return T
            if is_valid_network(G, copy):
                T.remove_node(v)

        # newly added: brute force final deletion check
        #         node_list = list(T.nodes())
        #         for node in node_list:
        #             copy = T.copy()
        #             copy.remove_node(node)
        #             if len(list(copy.nodes)) == 0:
        #                 return T
        #             if is_valid_network(G, copy):
        #                 T.remove_node(node)
        min_T = T.copy()

        # Add in nodes with eccentricity lower than T's radius to possibly reduce the longest distance between 2 endpoints
        for node in G.nodes():
            if nx.eccentricity(G, node) <= nx.radius(T):
                T.add_node(node)
                for e in G.edges(node):
                    T.add_edge(e[0], e[1])

        for i in range(300):
            copy = T.copy()
            while is_valid_network(G, copy):
                T = copy.copy()
                node_list = list(T.nodes())
                n = random.choices(node_list)[0]
                copy.remove_node(n)
                if len(list(copy.nodes)) == 0:
                    return T
                if is_valid_network(G, copy):
                    if utils.average_pairwise_distance_fast(copy) < utils.average_pairwise_distance_fast(min_T):
                        min_T = copy.copy()

        # added
        #         min_T.remove_edge(20, 21)
        #         min_T.add_edge(13, 17)

        return min_T
        # TODO: your code here!


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    #     input_folder_path = 'inputs'
    #     for input_file in os.listdir(input_folder_path):
    #         print(input_file)
    #         full_path = os.path.join(input_folder_path, input_file)
    #         G = read_input_file(full_path)
    #         T = solve(G)
    #         assert is_valid_network(G, T), "T is not a valid network of G."
    #         # Compare previous result with new result, update if improvement seen
    #         old = read_output_file('outputs/' + input_file[:-2] + 'out', G)
    #         dist_old = average_pairwise_distance_fast(old)
    #         dist_new = average_pairwise_distance_fast(T)
    #         print("Old Average  pairwise distance: {}".format(dist_old))
    #         print("New Average  pairwise distance: {}".format(dist_new))
    #         if dist_old > dist_new:
    #             write_output_file(T, 'outputs/' + input_file[:-2] + 'out')

    path = 'inputs/small-1.in'
    G = read_input_file(path)
    # Added: if more than 15 nodes with degree <= 2
    total = sum([1 for n in G.nodes() if G.degree(n) <= 2])
    leaves = sum([1 for n in G.nodes() if G.degree(n) < 2])
    # if total >= 15 and G.number_of_nodes() <= 25 and leaves <= 12:
    #     T = bruteforce(G)
    # else:
    #     T = solve(G)
    T = bary(G)
    fig = plt.figure(figsize=(20, 30))
    fig.add_subplot(211)
    pos = nx.spring_layout(G)

    labels = nx.get_edge_attributes(G, 'weight')

    nx.draw_networkx(G, pos=pos, node_color='yellow')
    nx.draw_networkx(G.edge_subgraph(T.edges()), pos=pos, node_color='red', edge_color='red')
    nx.draw_networkx(T, pos=pos, node_color='blue')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    assert is_valid_network(G, T), "T is not a valid network of G."

    print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    write_output_file(T, 'outputs/' + path[7:-2] + 'out')




