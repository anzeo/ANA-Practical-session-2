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

    while uncovered_edges:
        (u, v) = uncovered_edges.pop()
        if u not in C and v not in C:
            C.add(u)

        # A je to optimalno delat?
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

    uncovered_edges = set(G.edges)

    tmp_G = G.copy()  # Make a copy of graph so the original isn't changed

    # start = time.time()
    vertices = dict()
    for vertex in G.nodes:
        # vertices[vertex] = len(G.edges(vertex))
        adj_edge_count = len(G.edges(vertex))
        if adj_edge_count not in vertices:
            vertices[adj_edge_count] = set()
        vertices[adj_edge_count].add(vertex)
    # end = time.time()
    # print("Vertex time: ", end - start)
    # print(max(vertices))
    """

        1----2----3
        |    |
        |    |
        4----5----6
    """
    # totalMax = 0.0
    # totalAdjacent = 0.0
    # start = time.time()
    while uncovered_edges:
        # start1 = time.time()

        max_adj_edge_count = max(vertices)  # Find set of vertices with most uncovered adjacent edges.
        max_vertex = vertices[max_adj_edge_count].pop()  # Select one of vertices with most uncovered adjacent edges.
        if not vertices[max_adj_edge_count]:
            del vertices[max_adj_edge_count]
        # end1 = time.time()
        # totalMax += (end1 - start1)

        # print(max_vertex)
        C.add(max_vertex)

        # start1 = time.time()
        for adjacent_edge in list(tmp_G.edges(max_vertex)):
            (u, v) = adjacent_edge
            adj_vertex = (u if u != max_vertex else v)
            adj_vertex_degree = len(tmp_G.edges(adj_vertex))
            vertices[adj_vertex_degree].remove(adj_vertex)
            if adj_vertex_degree > 1:
                if adj_vertex_degree - 1 not in vertices:
                    vertices[adj_vertex_degree - 1] = set()
                vertices[adj_vertex_degree - 1].add(adj_vertex)
            if not vertices[adj_vertex_degree]:
                del vertices[adj_vertex_degree]

            # vertices[u] -= 1
            # vertices[v] -= 1

            if (u, v) in uncovered_edges:
                uncovered_edges.remove((u, v))
            if (v, u) in uncovered_edges:
                uncovered_edges.remove((v, u))
            tmp_G.remove_edge(u, v)

        # del vertices[max_vertex]
        # end1 = time.time()
        # totalAdjacent += (end1 - start1)

    # end = time.time()
    # print("Cover search time: ", end - start)
    # print("Total max time: ", totalMax)
    # print("Total adjacent time: ", totalAdjacent)

    return C


    return C


if __name__ == '__main__':
    directory = 'tests'
    start = time.time()
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            fo = open(f, "rb")
            G = nx.read_edgelist(fo)
            fo.close()
            a = nx.maximal_matching(G)
            naive = vc_naive_approach(G)
            greedy = vc_greedy_approach(G)
            print(filename, " ", str(len(naive)), str(len(greedy)))
        # break
    end = time.time()
    print(end - start)

    """
         7
         |
         |
    1----2----3----8
    |    |
    |    |
    4----5
    """

    """

        1----2----3
        |    |
        |    |
        4----5
    """

    # filename = "tests/test.graph"
    # f = open(filename, "rb")
    # G = nx.read_edgelist(f, nodetype=int)
    # f.close()
    # print(vc_naive_approach(G))
    # print((vc_greedy_approach(G)))

    # print(G.edges)
