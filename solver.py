import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import os
import sys


def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """

    T = nx.minimum_spanning_tree(G)
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
            temp = nx.Graph()
            temp.add_node(list(T.nodes)[0])
            return temp
        v_list = list(set(v_list))

        for v in v_list:
            copy = T.copy()
            copy.remove_node(v)
            if len(list(copy.nodes)) == 0:
                temp = nx.Graph()
                temp.add_node(list(T.nodes)[0])
                return temp
            if is_valid_network(G, copy):
                T.remove_node(v)
        return T
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

    # path = '/Users/chenpengyuan/Desktop/CS170/project-sp20-skeleton/inputs/medium-258.in'
    # G = read_input_file(path)
    # T = solve(G)
    # assert is_valid_network(G, T), "T is not a valid network of G."
    # print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    # write_output_file(T, '/Users/chenpengyuan/Desktop/CS170/project-sp20-skeleton/outputs/' + path[-10:-2] + 'out')