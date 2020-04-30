# import networkx as nx
# from parse import read_input_file, write_output_file
# from utils import is_valid_network, average_pairwise_distance
# import os
# import sys
# import matplotlib.pyplot as plt
# import random
# import utils


# def solve(G):
#     """
#     Args:
#         G: networkx.Graph

#     Returns:
#         T: networkx.Graph
#     """

#     T = nx.minimum_spanning_tree(G)
#     if T.number_of_nodes() == 1:
#         return T
#     elif T.number_of_nodes() == 2:
#         node = list(T.nodes)[0]
#         T.remove_node(node)
#         return T
#     else:
#         u_list = []
#         v_list = []
#         v_u_dict = {node:[] for node in T.nodes}
#         for u in T.nodes:
#             if T.degree(u) == 1:
#                 v = list(T.neighbors(u))[0]
#                 u_list.append(u)
#                 v_list.append(v)
#                 v_u_dict[v].append(u)
#         for u in u_list:
#             T.remove_node(u)
#         if len(list(T.nodes)) == 1:
#             return T
#         v_list = list(set(v_list))
#         for v in v_list:
#             copy = T.copy()
#             copy.remove_node(v)
#             if len(list(copy.nodes)) == 0:
#                 return T
#             if is_valid_network(G, copy):
#                 T.remove_node(v)
        
#         # newly added: brute force final deletion check
#         # node_list = list(T.nodes())
#         # for node in node_list:
#         #     copy = T.copy()
#         #     copy.remove_node(node)
#         #     if len(list(copy.nodes)) == 0:
#         #         return T
#         #     if is_valid_network(G, copy):
#         #         T.remove_node(node)
        
#         min_T = T.copy()
#         for i in range(300):
#             copy = T.copy()
#             while is_valid_network(G, copy):
#                 T = copy.copy()
#                 node_list = list(T.nodes())
#                 n = random.choice(node_list)
#                 copy.remove_node(n)

#                 if len(list(copy.nodes)) == 0:
#                     return T
#                 if is_valid_network(G, copy):
#                     if utils.average_pairwise_distance_fast(copy) < utils.average_pairwise_distance_fast(min_T):
#                         min_T = copy.copy()
        
#         return min_T
#         # TODO: your code here!
    


# # Here's an example of how to run your solver.

# # Usage: python3 solver.py test.in

# if __name__ == '__main__':

#     input_folder_path = '/Users/chenpengyuan/Desktop/CS170/project-sp20-skeleton/inputs'
#     for input_file in os.listdir(input_folder_path):
#         print(input_file)
#         full_path = os.path.join(input_folder_path, input_file)
#         G = read_input_file(full_path)
#         T = solve(G)
#         assert is_valid_network(G, T), "T is not a valid network of G."
#         print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
#         write_output_file(T, '/Users/chenpengyuan/Desktop/CS170/project-sp20-skeleton/outputs/' + input_file[:-2] + 'out')

#     # path = '/Users/chenpengyuan/Desktop/CS170/project-sp20-skeleton/inputs/large-26.in'
#     # G = read_input_file(path)
#     # T = solve(G)
#     # fig = plt.figure(figsize=(20,30))
#     # fig.add_subplot(211)
#     # pos = nx.spring_layout(G)
#     # nx.draw_networkx(G, pos=pos, node_color='yellow')
#     # nx.draw_networkx(G.subgraph(T.nodes()), pos=pos, node_color='orange', edge_color='red')
#     # assert is_valid_network(G, T), "T is not a valid network of G."
#     # print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
#     # write_output_file(T, '/Users/chenpengyuan/Desktop/CS170/project-sp20-skeleton/outputs/' + path[-10:-2] + 'out')


# below is the method using degree heuristic to let MST pick higher-degree nodes

import networkx as nx
from parse import read_input_file, write_output_file
import utils
from utils import is_valid_network, average_pairwise_distance
import os
import sys
import matplotlib.pyplot as plt
import random
import numpy as np


def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """
    # added: weight <- weight - degree, to let MST choose higher-degree nodes
    G_copy = G.copy()
    for edge in G_copy.edges.data():
        u, v = edge[0], edge[1]
        edge[2]['weight'] /= np.log(max(G_copy.degree(u), G_copy.degree(v)))
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
        v_u_dict = {node:[] for node in T.nodes}
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
        for i in range(300):
            copy = T.copy()
            while is_valid_network(G, copy):
                T = copy.copy()
                node_list = list(T.nodes())
                n = random.choice(node_list)
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

    input_folder_path = '/Users/chenpengyuan/Desktop/CS170/project-sp20-skeleton/inputs'
    for input_file in os.listdir(input_folder_path):
        print(input_file)
        full_path = os.path.join(input_folder_path, input_file)
        G = read_input_file(full_path)
        T = solve(G)
        assert is_valid_network(G, T), "T is not a valid network of G."
        print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
        write_output_file(T, '/Users/chenpengyuan/Desktop/CS170/project-sp20-skeleton/outputs/' + input_file[:-2] + 'out')

    # path = '/Users/chenpengyuan/Desktop/CS170/project-sp20-skeleton/inputs/small-241.in'
    # G = read_input_file(path)
    # T = solve(G)
    # fig = plt.figure(figsize=(20,30))
    # fig.add_subplot(211)
    # pos = nx.spring_layout(G)
    
    # labels = nx.get_edge_attributes(G,'weight')
    
    # nx.draw_networkx(G, pos=pos, node_color='yellow')
    # nx.draw_networkx(G.edge_subgraph(T.edges()), pos=pos, node_color='orange', edge_color='yellow')
    # nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    # assert is_valid_network(G, T), "T is not a valid network of G."
    # print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    # write_output_file(T, '/Users/chenpengyuan/Desktop/CS170/project-sp20-skeleton/outputs/' + path[-10:-2] + 'out')
