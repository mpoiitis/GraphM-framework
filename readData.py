import os
import networkx as nx
import pandas as pd
import numpy as np
import scipy.sparse as sp
from time import time
from scipy.io import loadmat


def produce_node_dict(G, x):
    """
    Maps node names to integer codes
    :param G: the initial graph
    :param x: the initial feature matrix
    :return: adjusted graph, adjusted feature matrix, node mapping dictionary
    """

    # map node names to int indices
    node_dict = dict()
    for i, n in enumerate(G.nodes()):
        node_dict.update({str(n): i})
    G = nx.relabel_nodes(G, node_dict)  # replace values in G
    x.iloc[:, 0] = x.iloc[:, 0].astype(str)
    x.iloc[:, 0].replace(node_dict, inplace=True)  # replace values in x
    return G, x, node_dict


def load_edgelist(file, directed=False, weighted=False):
    """
    Load graph from edge list
    :param file: the edge list file
    :param directed: whether the graph should be directed
    :param weighted: whether the graph should be weighted
    :return: the graph
    """
    t0 = time()
    if weighted:
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


def load_adjacencylist(file, directed=False, weighted=False):
    """
    Load graph from adjacency list
    :param file: the adjacency list file
    :param directed: whether the graph should be directed
    :param weighted: whether the graph should be weighted
    :return: the graph
    """
    t0 = time()
    if weighted:
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


def load_karate(file, directed=False, weighted=None):
    return load_adjacencylist(file, directed, weighted)


def load_gnutella(file, directed=False, weighted=None):
    return load_edgelist(file, directed, weighted)


def load_matfile(file='data/Amherst41.mat', directed=False):
    """
    Load graph from .mat file. Column 5 - "year" - is used as user label.
    :param file: the .mat file
    :param directed: whether the graph should be directed
    :return: the graph, x feature data, y labels
    """

    if not file.lower().endswith('.mat'):
        raise Exception('Wrong file type is given. Should be *.mat')

    t0 = time()
    mat_variables = loadmat(file)
    mat_matrix = mat_variables["A"]  # adjacency matrix

    feat_matrix = mat_variables["local_info"]  # feature matrix for each node

    df = pd.DataFrame(feat_matrix)
    x = df.drop(5, axis=1)
    y = df[5]

    G = nx.DiGraph(weight=1)
    if sp.issparse(mat_matrix):
        cx = mat_matrix.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            G.add_edge(i, j, weight=v)
    else:
        raise Exception("Dense matrices not yet supported.")

    # map node names to int indices
    G, x, node_dict = produce_node_dict(G, x)

    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    t1 = time()
    print('Graph loaded in {}s'.format(t1 - t0))

    return G, x, y, node_dict


def load_citeseer_cora(citesFile='data/citeseer.cites', contentFile='data/citeseer.content', directed=True):
    """
    Load one of two citation networks available, CiteSeer and Cora
    :param citesFile: the network file (edgelist)
    :param contentFile: the user feature file
    :param directed: whether the graph should be directed
    :return: the graph, x feature data, y labels
    """
    if not citesFile.lower().endswith('.cites') or not contentFile.lower().endswith('.content'):
        raise Exception('Wrong file type is given. First file should be *.cites and second *.content')

    t0 = time()
    G = nx.read_edgelist(citesFile, create_using=nx.DiGraph())

    df = pd.read_csv(contentFile, sep="\t", header=None)
    x = df.iloc[:, :-1]  # drop last column
    x.iloc[:, 0] = x.iloc[:, 0].astype(str)
    y = df.iloc[:, -1]  # labels are stored in df's last column

    # map node names to int indices
    G, x, node_dict = produce_node_dict(G, x)

    # create all-zero rows for nodes that are not contained in the feature set
    if x.shape[0] != len(G.nodes):
        new_x = np.ndarray((len(G.nodes), x.shape[1]))
        new_y = np.ndarray((len(G.nodes), 1), dtype=str)
        for i, r in x.iterrows():
           new_x[r.iloc[0], :] = r
           new_y[r.iloc[0]] = y.iloc[i]
        x = pd.DataFrame(new_x)
        y = pd.DataFrame(new_y)

    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    t1 = time()
    print('Graph loaded in {}s'.format(t1 - t0))

    return G, x, y, node_dict


def load_facebook(file, directory, directed=False):
    """
   Load graph from facebook dataset. Column 53 - "education type" - is used as user label.
   :param file: the edge list facebook file
   :param directory: the folder which contains the files that sum up the user features
   :param directed: whether the graph should be directed
   :return: the graph, x feature data, y labels
   """

    G = load_edgelist(file, directed=directed)

    users = list()
    for filename in os.listdir(directory):
        if filename.endswith('.feat') or filename.endswith('.egofeat'):
            with open(directory + '/' + filename) as f:
                df = pd.read_csv(f, index_col=None, header=None, delimiter=' ')
                users.append(df)

    user_df = pd.concat(users, axis=0, ignore_index=True, names=[str(item) for item in np.arange(577)])
    user_df = user_df.drop_duplicates(subset=[0]) # keep unique users

    x = user_df.drop(54, axis=1)
    y = user_df[54]

    # map node names to int indices
    G, x, node_dict = produce_node_dict(G, x)

    return G, x, y, node_dict


def load_graph(args):
    """
    Loads the graph according to the specified input dataset
    :param args: the cli arguments
    :return: the corresponding graph
    """

    # Karate and GNutella do not contain features and labels
    x = None
    y = None
    node_dict = None

    if args.input == "karate":
        G = load_karate('data/karate.adjlist', directed=args.directed, weighted=args.weighted)
    elif args.input == "gnutella":
        G = load_gnutella('data/p2p-Gnutella08.edgelist', directed=args.directed, weighted=args.weighted)
    elif args.input == 'amherst':
        G, x, y, node_dict = load_matfile('data/Amherst41.mat', directed=args.directed)
    elif args.input == 'hamilton':
        G, x, y, node_dict = load_matfile('data/Hamilton46.mat', directed=args.directed)
    elif args.input == 'mich':
        G, x, y, node_dict = load_matfile('data/Mich67.mat', directed=args.directed)
    elif args.input == 'rochester':
        G, x, y, node_dict = load_matfile('data/Rochester38.mat', directed=args.directed)
    elif args.input == 'cora':
        G, x, y, node_dict = load_citeseer_cora('data/cora.cites', 'data/cora.content', directed=args.directed)
    elif args.input == 'citeseer':
        G, x, y, node_dict = load_citeseer_cora('data/citeseer.cites', 'data/citeseer.content', directed=args.directed)
    elif args.input == 'facebook':
        G, x, y, node_dict = load_facebook('data/facebook_combined.txt', 'data/facebook', directed=args.directed)
    else:
        raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist'" % args.format)

    return G, x, y, node_dict