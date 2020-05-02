# import networkx as nx
# from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
import sys
import copy
from parse import *
import networkx as nx
import os


def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """
    # TODO: your code here!
    degree = G.degree()
    degree_list = [degree[i] for i in range(0, len(degree))]
    d = G.number_of_nodes() - 1
    if d in degree_list: # if some nodes connect to all the other nodes
        node = degree_list.index(d)
        res = nx.Graph()
        res.add_node(node)
        return res
    val = heuristics(G)
    improve_val = improve_heuristics(G)
    temp = add_edge(G, val)
    return improve(G, temp, improve_val)
    # pass


def heuristics(G):
    """
    :param G: Given a graph G
    :return: the estimated cost of choosing each vertex.
    """
    res = []
    degree = G.degree()
    deg_para = 10000/G.number_of_edges()
    wei_para = -100/average_pairwise_distance_fast(G)
    for i in range(0, G.number_of_nodes()):
        total_weight = 0
        for neighbour in G.neighbors(i):
            total_weight += G.get_edge_data(i, neighbour).get('weight')
        res.append(deg_para * degree[i] + wei_para * total_weight)
    return res


def improve_heuristics(G):
    """
    :param G: Given a graph G
    :return: the estimated cost of choosing each vertex.
    Used in the improve method
    """
    res = []
    degree = G.degree()
    deg_para = 0.0
    wei_para = 1.0
    for i in range(0, G.number_of_nodes()):
        total_weight = 0
        for neighbour in G.neighbors(i):
            total_weight += G.get_edge_data(i, neighbour).get('weight')
        res.append(deg_para * degree[i] + wei_para * total_weight)
    return res


def add_edge(G, val):
    """
    :param G: The original graph, used to check whether the new graph is a valid network.
    :param val: estimated cost of choosing each vertex.
    :return: possible best result
    """
    best_graph = nx.Graph()
    edge_list = []
    new_edge = ""
    node = val.index(max(val))
    new_edge += str(node)
    max_val = -float("inf")
    max_neighbour = 0
    for neighbour in G.neighbors(node):
        if val[neighbour] > max_val and neighbour != node:
            max_val = val[neighbour]
            max_neighbour = neighbour
    new_edge += " " + str(max_neighbour)
    new_edge += " " + str(G.get_edge_data(node, max_neighbour).get("weight"))
    edge_list.append(new_edge)
    new_graph = nx.parse_edgelist(edge_list, nodetype=int, data=(('weight', float),))
    val[node], val[max_neighbour] = -float("inf"), -float("inf")
    if is_valid_network(G, new_graph):
        # if average_pairwise_distance_fast(new_graph) < best_so_far:
        #     best_so_far = average_pairwise_distance_fast(new_graph)
        best_graph = copy.deepcopy(new_graph)
        return best_graph

    while True:
        new_edge = ""
        max_val = -float("inf")
        max_neighbour = 0
        max_node = 0
        check = False
        for node in new_graph.nodes():
            for neighbour in G.neighbors(node):
                if val[neighbour] > max_val:
                    max_val = val[neighbour]
                    max_neighbour = neighbour
                    max_node = node
                    check = True
        if not check:
            break
        new_edge += str(max_node)
        new_edge += " " + str(max_neighbour)
        new_edge += " " + str(G.get_edge_data(max_node, max_neighbour).get("weight"))
        edge_list.append(new_edge)
        new_graph = nx.parse_edgelist(edge_list, nodetype=int, data=(('weight', float),))
        val[max_node], val[max_neighbour] = -float("inf"), -float("inf")
        if is_valid_network(G, new_graph):
            best_graph = copy.deepcopy(new_graph)
            break
    return best_graph


def improve(G, temp, val):
    """
    :param G: The original graph
    :param temp: Best graph we get so far. It's already a valid network, but in some cases add a new vertex can
    reduce the cost
    :param val: Heurisitics
    :return: new Best graph
    """
    best_so_far = average_pairwise_distance_fast(temp)
    new_graph = copy.deepcopy(temp)
    while True:
        min_val = float("inf")
        min_neighbour = 0
        min_node = 0
        check = False
        for node in temp.nodes():
            for neighbour in G.neighbors(node):
                if val[neighbour] < min_val:
                    min_val = val[neighbour]
                    min_neighbour = neighbour
                    min_node = node
        min_weight = G.get_edge_data(min_node, min_neighbour).get("weight")
        new_graph.add_edge(min_node, min_neighbour, weight=min_weight)
        val[min_node], val[min_neighbour] = -1, -1
        if is_valid_network(G, new_graph):
            if average_pairwise_distance_fast(new_graph) < best_so_far:
                best_so_far = average_pairwise_distance_fast(new_graph)
                check = True
        if not check:
            return temp
    return new_graph


# def solve(G):
#     """
#     Args:
#         G: networkx.Graph
#     Returns:
#         T: networkx.Graph
#     """
#     # TODO: your code here!
#     mst_g = nx.minimum_spanning_tree(G)
#     best_graph = copy.deepcopy(G)
#     cost_so_far = average_pairwise_distance_fast(mst_g)
#     assert mst_g.number_of_edges() == mst_g.number_of_nodes() - 1
#     memo = []
#     for i in range(1, 4):
#         res = delete_edge(G, mst_g, i, memo)
#         if res[0] < cost_so_far:
#             cost_so_far = res[0]
#             print(cost_so_far)
#             best_graph = copy.deepcopy(res[1])
#         else:
#             break
#     return best_graph
#
# def delete_edge(G, mst, k, memo):
#     """
#     Given a graph mst, delete k edges randomly. Check all possible deletions whether they are valid networks.
#     Always delete the edges that connects to the leaf node.
#
#     G is the original graph. Used to check whether the new graph is a valid network.
#
#     memo is a list storing all possible graphs that delete k - 1 edges(k > 1).
#     """
#     best_so_far = float("inf")
#     new_graph = copy.deepcopy(mst)
#     if k == 0:
#         return [average_pairwise_distance_fast(mst), mst]
#     elif k == 1:
#         degree = mst.degree()
#         for i in mst.nodes():
#             if degree[i] == 1:  # degree(leaf node) = 1
#                 temp_g = copy.deepcopy(mst)
#                 neighbour_node = next(temp_g.neighbors(i))
#                 temp_g.remove_edge(i, neighbour_node)
#                 temp_g.remove_node(i)
#                 if is_valid_network(G, temp_g):
#                     memo.append(temp_g)
#                     # print(len(memo))
#                     temp = delete_edge(G, temp_g, 0, memo)
#                     if temp[0] < best_so_far:
#                         best_so_far = temp[0]
#                         new_graph = copy.deepcopy(temp[1])
#         if mst in memo:
#             memo.remove(mst)
#     else:
#         length = len(memo)
#         for i in range(0, length):
#             temp = delete_edge(G, memo[0], 1, memo)
#             if temp[0] < best_so_far:
#                 best_so_far = temp[0]
#                 new_graph = copy.deepcopy(temp[1])
#     return [best_so_far, new_graph]


# Here's an example of how to run your solver.
# Usage: python3 solver.py test.in
# if __name__ == '__main__':
#     assert len(sys.argv) == 2
#     path = sys.argv[1]
#     G = read_input_file(path)
#     T = solve(G)
#     assert is_valid_network(G, T)
#     print(*T.nodes())
#     for i in T.edges():
#         print(i[0], i[1])
#     print("cost:", average_pairwise_distance_fast(T))
    # print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    # write_output_file(T, 'out/test.out')

if __name__ == "__main__":
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

    path = 'inputs/small-3.in'
    G = read_input_file(path)
    # Added: if more than 15 nodes with degree <= 2
    total = sum([1 for n in G.nodes() if G.degree(n) <= 2])
    if total >= 15:
        T = bruteforce(G)
    else:
        T = solve(G)
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
    write_output_file(T, 'outputs/' + path[-10:-2] + 'out')
