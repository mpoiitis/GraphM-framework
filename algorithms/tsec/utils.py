import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def preprocess_features(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.

    if isinstance(features, sp.lil.lil_matrix):
        r_mat_inv = sp.diags(r_inv)
    else:
        r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)

    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Normalize adjacency with added self loops"""
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]))
    return adj_normalized


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = np.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - np.eye(adj.shape[0])

    t_k = list()
    t_k.append(np.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        return 2 * scaled_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return t_k


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def preprocess_data(features, y):
    # each instance should contain a 1-hot vector for all the labels, instead of a single label
    enc = OneHotEncoder(handle_unknown='ignore')
    y = enc.fit_transform(y.to_numpy().reshape(-1, 1))
    y = y.todense()

    features = features.iloc[:, 1:]  # remove ids
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.33, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)  # validation set

    train_idx_reorder = list(X_train.index)  # get indices of train set
    idx_train = np.sort(train_idx_reorder).tolist()  # sort indices
    test_idx_reorder = list(X_test.index)
    idx_test = np.sort(test_idx_reorder).tolist()
    val_idx_reorder = list(X_val.index)
    idx_val = np.sort(val_idx_reorder).tolist()

    features = sp.lil_matrix(features.to_numpy())
    labels = y

    # create masks for the three datasets
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return features, y_train, y_val, y_test, train_mask, val_mask, test_mask
