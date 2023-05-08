import os
import time

import networkx as nx

"""
vsi stirje algoritmi bi morali za 20 grafov tect nekje 2min
"""


# Naive approach of solving the vertex cover problem
def vc_naive_approach(G):
    C = set()

    uncovered_edges = set(G.edges)
    # tmp_G = G.copy()  # Make a copy of graph so the original isn't changed

    while uncovered_edges:
        (u, v) = uncovered_edges.pop()
        if u not in C and v not in C:
            C.add(u)

        # Is it worth doing that?
        # for adjacent_edge in list(tmp_G.edges(u)):
        #     # print(adjacent_edge)
        #     (au, av) = adjacent_edge
        #     if (au, av) in uncovered_edges:
        #         uncovered_edges.remove((au, av))
        #     if (av, au) in uncovered_edges:
        #         uncovered_edges.remove((av, au))
        #     tmp_G.remove_edge(au, av)
        # print()
    return C


def vc_greedy_approach(G):
    C = set()

    return C


if __name__ == '__main__':
    directory = 'tests'
    # start = time.time()
    # for filename in os.listdir(directory):
    #     f = os.path.join(directory, filename)
    #     # checking if it is a file
    #     if os.path.isfile(f):
    #         fo = open(f, "rb")
    #         G = nx.read_edgelist(fo)
    #         fo.close()
    #         a = nx.maximal_matching(G)
    #         b = vc_naive_approach(G)
    #         print(filename, " ", str(len(a)))
    #     # break
    # end = time.time()
    # print(end - start)

    """
         7
         |
         |
    1----2----3----8
    |    |
    |    |
    4----5
    """

    filename = "tests/test.graph"
    f = open(filename, "rb")
    G = nx.read_edgelist(f, nodetype=int)
    f.close()
    # print(G.edges())
    # print(nx.maximal_matching(G))
    print(vc_naive_approach(G))
    # print(len(vc_greedy_approach(G)))
    # print(G.edges)

