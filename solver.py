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
import itertools


def bruteforce(G):
    list_of_trees = []
    min_dist = 1000000
    minT = nx.Graph()

    # Remove degree 1 nodes and add their neighbors to minT
    def traverse(T, G, G_copy):
        nonlocal min_dist
        nonlocal minT
        if G_copy.number_of_edges() == 0:
            return
        for e in G_copy.edges():
            u, v = e[0], e[1]
            if e in T.edges():
                continue
            copy = T.copy()
            temp = G_copy.copy()
            T.add_node(u)
            T.add_node(v)
            T.add_edge(v, u, weight=G[v][u]['weight'])
            temp.remove_edge(u, v)
            if is_valid_network(G, T):
                if average_pairwise_distance_fast(T) < min_dist:
                    min_dist = average_pairwise_distance_fast(T)
                    minT = T.copy()
                traverse(T, G, temp)
            elif not nx.is_dominating_set(G, T.nodes()) or not nx.is_connected(T):
                traverse(T, G, temp)
            T = copy.copy()

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
                    G_copy.remove_edge(u, v)
                G.remove_node(v)
                changed = True
        if changed:
            findChain(G)

    def remove(copy, G_copy, k):
        nonlocal min_dist
        nonlocal minT
        has_valid = False
        for c in itertools.combinations(G_copy.edges(), k):
            temp = copy.copy()
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
        remove(copy, G_copy, k + 1)
        # if copy.number_of_nodes() == 1:
        #     if is_valid_network(G, copy):
        #         minT = copy
        #     return
        # for e in G_copy.edges():
        #     if e not in copy.edges():
        #         continue
        #     u, v = e[0], e[1]
        #     if copy.degree(u) == 1:
        #         if u in confirmed:
        #             continue
        #         copy.remove_node(u)
        #     elif copy.degree(v) == 1:
        #         if v in confirmed:
        #             continue
        #         copy.remove_node(v)
        #     else:
        #         copy.remove_edge(u, v)
        #     if is_valid_network(G, copy):
        #         if average_pairwise_distance_fast(copy) < min_dist:
        #             min_dist = average_pairwise_distance_fast(copy)
        #             minT = copy.copy()
        #         remove(copy, G_copy)
        #     elif nx.is_dominating_set(G, copy.nodes) and nx.is_connected(copy):
        #         remove(copy, G_copy)
        #     if u not in copy.nodes():
        #         copy.add_node(u)
        #     elif v not in copy.nodes():
        #         copy.add_node(v)
        #     copy.add_edge(u, v, weight=G[u][v]['weight'])

    # minT = min(list_of_trees, key=lambda t: average_pairwise_distance_fast(t))
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
    # remove(G.copy())
    # T = nx.Graph()
    remove(G.copy(), G_copy, 1)
    return minT


def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """

    def prune(g):
        nonlocal min_T
        copy = g.copy()
        if g.number_of_edges() == 0:
            min_T = g
            return
        e = random.choice(list(g.edges()))
        if copy.degree(e[0]) == 1:
            copy.remove_node(e[0])
        elif copy.degree(e[1]) == 1:
            copy.remove_node(e[1])
        else:
            copy.remove_edge(e[0], e[1])
        if is_valid_network(G, copy):
            #             d = average_pairwise_distance_fast(copy)
            #             min_dist = average_pairwise_distance_fast(min_T)
            #             if d < min_dist:
            #                 min_T = copy.copy()
            prune(copy)
        elif nx.is_connected(copy) and nx.is_dominating_set(G, copy):
            prune(copy)

    #     Special case for one central node
    for node in G.nodes():
        if G.degree(node) == G.number_of_nodes() - 1 and not G.has_edge(node, node):
            T = nx.Graph()
            T.add_node(node)
            return T
    # added: weight <- weight - degree, to let MST choose higher-degree nodes
    G_copy = G.copy()
    for edge in G_copy.edges():
        u, v = edge[0], edge[1]
        G_copy[u][v]['weight'] /= (G_copy.degree(u) + G_copy.degree(v)) / 2
        G_copy[u][v]['weight'] *= (nx.eccentricity(G, u) + nx.eccentricity(G, v)) / 2
        G_copy[u][v]['weight'] /= np.exp(nx.edge_betweenness_centrality(G, k=5, weight='weight')[edge])
        G_copy[u][v]['weight'] *= random.uniform(0.5, 2)

    #     T = nx.Graph()
    #     all_shortest_path_lengths = dict(nx.shortest_path_length(G, weight='weight'))
    #     all_shortest_paths = nx.shortest_path(G, weight='weight')
    #     average_shortest_path_lengths = {}
    #     for v in all_shortest_path_lengths.keys():
    #         average_shortest_path_lengths[v] = sum([all_shortest_path_lengths[v][u] for u in G.nodes()]) / (
    #                 G.number_of_nodes() - 1)
    #     root = min(average_shortest_path_lengths, key=lambda x: average_shortest_path_lengths[x])
    #     T.add_node(root)
    #     G_copy = G.copy()
    #     G_copy.remove_node(root)
    #     while not is_valid_network(G, T):
    #         average_distances = {}
    #         corresponding_edge = {}
    #         for node in T.nodes():
    #             for adj in G.neighbors(node):
    #                 if adj in T.nodes(): continue
    #                 if adj not in average_distances.keys():
    #                     average_distances[adj] = (sum(
    #                         [all_shortest_path_lengths[adj][t] for t in T.nodes()]) / T.number_of_nodes() +
    #                                               G[node][adj]['weight'])
    #                     corresponding_edge[adj] = (node, adj)
    #                 else:
    #                     temp = (sum(
    #                         [all_shortest_path_lengths[adj][t] for t in T.nodes()]) / T.number_of_nodes() +
    #                                               G[node][adj]['weight'])
    #                     if average_distances[adj] > temp:
    #                         average_distances[adj] = temp
    #                         corresponding_edge[adj] = (node, adj)
    #         next = min(average_distances.keys(), key=lambda x: average_distances[x])
    #         T.add_node(next)
    #         G_copy.remove_node(next)
    #         T.add_edge(corresponding_edge[next][0], corresponding_edge[next][1],
    #                    weight=G[corresponding_edge[next][0]][corresponding_edge[next][1]]['weight'])

    #     min_T = T.copy()

    T = nx.minimum_spanning_tree(G_copy)
    for edge in T.edges.data():
        u, v = edge[0], edge[1]
        edge[2]['weight'] = G[u][v]['weight']

    min_T = T.copy()

    if T.number_of_nodes() == 1:
        return T
    elif T.number_of_nodes() == 2:
        node = list(T.nodes)[0]
        T.remove_node(node)
        return T
    else:
        for i in range(100):
            prune(T.copy())

        def findChain(G):
            nonlocal confirmed
            nonlocal G_copy
            changed = False
            nodes = [n for n in G.nodes()]
            for v in nodes:
                if G.degree(v) == 1:
                    u = list(G.neighbors(v))[0]
                    confirmed.append(u)
                    if u in confirmed and v in confirmed:
                        T.add_edge(u, v, weight=G[u][v]['weight'])
                        G_copy.remove_edge(u, v)
                    G.remove_node(v)
                    changed = True
            if changed:
                findChain(G)

        confirmed = []
        G_copy = G.copy()
        findChain(G.copy())

        for i in range(100):
            copy = min_T.copy()
            while is_valid_network(G, copy):
                if copy.number_of_edges() == 1:
                    if utils.average_pairwise_distance_fast(copy) < utils.average_pairwise_distance_fast(min_T):
                        min_T = copy.copy()
                    break
                T = copy.copy()
                edge_weights = []
                for e in T.edges():
                    if e[0] in confirmed and e[1] in confirmed:
                        edge_weights.append(0)
                    else:
                        edge_weights.append(np.exp(1 + G[e[0]][e[1]]['weight']))
                edge = random.choices(list(T.edges()))[0]
                if T.degree(edge[0]) == 1:
                    copy.remove_node(edge[0])
                elif T.degree(edge[1]) == 1:
                    copy.remove_node(edge[1])
                else:
                    copy.remove_edge(edge[0], edge[1])
                if len(list(copy.nodes)) == 0:
                    return T
                if is_valid_network(G, copy):
                    if utils.average_pairwise_distance_fast(copy) < utils.average_pairwise_distance_fast(min_T):
                        min_T = copy.copy()

        #         min_T.remove_edge(20, 21)
        #         min_T.add_edge(13, 17)

        #         # Add some edges with lower weights
        #         all_shortest_path_lengths = dict(nx.shortest_path_length(G, weight='weight'))
        #         all_shortest_paths = nx.shortest_path(G, weight='weight')
        #         dist = average_pairwise_distance_fast(T)
        #         min_T = T.copy()
        #         for i in range(100):
        #             T = min_T.copy()
        #             edge_list = list(G.edges())
        #             random.shuffle(edge_list)
        #             for e in edge_list:
        #                 if e in T.edges(): continue
        #                 copy = T.copy()
        #                 copy.add_edge(e[0], e[1], weight=G[e[0]][e[1]]['weight'])
        #                 if is_valid_network(G, copy):
        #                     if average_pairwise_distance_fast(T) > average_pairwise_distance_fast(copy):
        #                         T = copy.copy()
        #                 else: break
        #             if average_pairwise_distance_fast(T) < average_pairwise_distance_fast(min_T):
        #                 min_T = T.copy()

        return min_T
        # TODO: your code here!


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    #         input_folder_path = 'inputs'
    #         for input_file in os.listdir(input_folder_path):
    #             print(input_file)
    #             full_path = os.path.join(input_folder_path, input_file)
    #             G = read_input_file(full_path)
    # #             total = sum([1 for n in G.nodes() if G.degree(n) <= 2])
    # #             leaves = sum([1 for n in G.nodes() if G.degree(n) < 2])
    # #             if total >= 15 and G.number_of_nodes() <= 25 and leaves <= 12:
    # #                 T = bruteforce(G)
    # #             else:
    #             Tree_list = []
    #             for i in range(10):
    #                 Tree_list.append(solve(G))
    #             T = min(Tree_list, key= lambda t: average_pairwise_distance_fast(t))
    #             assert is_valid_network(G, T), "T is not a valid network of G."
    #             # Compare previous result with new result, update if improvement seen
    #             old = read_output_file('outputs/' + input_file[:-2] + 'out', G)
    #             dist_old = average_pairwise_distance_fast(old)
    #             dist_new = average_pairwise_distance_fast(T)
    #             print("Old Average  pairwise distance: {}".format(dist_old))
    #             print("New Average  pairwise distance: {}".format(dist_new))
    #             if dist_old > dist_new:
    #                 write_output_file(T, 'outputs/' + input_file[:-2] + 'out')

    file_list = "large-121, large-122, large-124, large-126, large-128, large-129, large-130, large-133, large-134, large-135, large-136, large-137, large-139"
    files = file_list.split(', ')
    for file in files:
        print(file)
        path = 'inputs/' + file + '.in'
        # path = 'inputs/small-4.in'
        G = read_input_file(path)
        Tree_list = []
        for i in range(100):
            Tree_list.append(solve(G))
        T = min(Tree_list, key=lambda t: average_pairwise_distance_fast(t))

        # Added: if more than 15 nodes with degree <= 2
        #         total = sum([1 for n in G.nodes() if G.degree(n) <= 2])
        #         leaves = sum([1 for n in G.nodes() if G.degree(n) < 2])
        #         if total >= 15 and G.number_of_nodes() <= 25 and leaves <= 12:
        #             T = bruteforce(G)
        #         else:
        #             T = solve(G)

        #         fig = plt.figure(figsize=(20, 30))
        #         fig.add_subplot(211)
        #         pos = nx.circular_layout(G)

        #         labels = nx.get_edge_attributes(G, 'weight')

        #         nx.draw_networkx(G, pos=pos, node_color='yellow')
        #         nx.draw_networkx(G.edge_subgraph(T.edges()), pos=pos, node_color='red', edge_color='red')
        #         # nx.draw_networkx(T, pos=pos, node_color='blue')
        #         nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        assert is_valid_network(G, T), "T is not a valid network of G."
        # Compare previous result with new result, update if improvement seen
        old = read_output_file('outputs/' + path[7:-2] + 'out', G)
        dist_old = average_pairwise_distance_fast(old)
        dist_new = average_pairwise_distance_fast(T)
        print("Old Average  pairwise distance: {}".format(dist_old))
        print("New Average  pairwise distance: {}".format(dist_new))
        if dist_old > dist_new:
            write_output_file(T, 'outputs/' + path[7:-2] + 'out')






