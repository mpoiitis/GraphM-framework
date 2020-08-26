from time import time
import networkx as nx

def load_edgelist(file, directed=False, weighted=None):
    """
    Load graph from edge list
    :param file: the edge list file
    :param directed: whether the graph should be directed
    :param weighted: whether the graph should be weighted
    :return: the graph
    """
    t0 = time()
    if weighted == None:
        G = nx.read_edgelist(file, nodetype=int, create_using=nx.DiGraph())
    elif weighted:
        G = nx.read_edgelist(file, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else: # false. This is different from None. Only applicable to methods that support weights.
        G = nx.read_edgelist(file, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    t1 = time()
    print('Graph loaded in {}s'.format(t1 - t0))

    return G


def load_adjacencylist(file, directed=False, weighted=None):
    """
    Load graph from adjacency list
    :param file: the adjacency list file
    :param directed: whether the graph should be directed
    :param weighted: whether the graph should be weighted
    :return: the graph
    """
    t0 = time()
    if weighted == None:
        G = nx.read_adjlist(file, nodetype=int, create_using=nx.DiGraph())
    elif weighted:
        G = nx.read_adjlist(file, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else: # false. This is different from None. Only applicable to methods that support weights.
        G = nx.read_adjlist(file, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    t1 = time()
    print('Graph loaded in {}s'.format(t1 - t0))

    return G