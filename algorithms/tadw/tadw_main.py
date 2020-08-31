import networkx as nx
import numpy as np
from scipy import sparse
from tqdm import tqdm
from .tadw import DenseTADW, SparseTADW


def normalize_adjacency(G):
    """
    Method to calculate a sparse degree normalized adjacency matrix.
    :param G: Sparse graph adjacency matrix.
    :return A: Normalized adjacency matrix.
    """
    node_indices = range(len(G.nodes()))

    degs = [1.0/G.degree(node) for node in G.nodes()]
    A = sparse.coo_matrix(nx.adjacency_matrix(G), dtype=np.float32)
    degs = sparse.coo_matrix((degs, (node_indices, node_indices)), shape=A.shape, dtype=np.float32)
    A = A.dot(degs)
    return A


def preprocess_adjacency(G, order):
    """
    Preprocess the graph's adjacency, to prepare it for TADW
    """
    A = normalize_adjacency(G)
    if order > 1:
        powered_A, out_A = A, A
        for _ in tqdm(range(order - 1)):
            powered_A = powered_A.dot(A)
            out_A = out_A + powered_A
    else:
        out_A = A
    return out_A


def tadw(args, G, X=None):
    """
    Method to create adjacency matrix powers, read features, and learn embedding.
    :param args: Arguments object.
    """
    A = preprocess_adjacency(G, args.order)
    if args.features == "dense":
        node_names = X.iloc[:, 0]
        node_features = X.iloc[:, 1:]
        X = np.array(node_features.values).transpose()
        model = DenseTADW(A, X, node_names, args)
    elif args.features == "sparse":
        node_names = X.iloc[:, 0]
        node_features = X.iloc[:, 1:]
        X = sparse.coo_matrix(node_features.values, dtype=np.float32).transpose()
        model = SparseTADW(A, X, node_names, args)
    model.optimize()
    model.save_embedding()
