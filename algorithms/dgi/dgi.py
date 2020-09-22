import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from utils import preprocess_adj, preprocess_features, preprocess_data_trunc, sparse_mx_to_torch_sparse_tensor
from .models import DGI, LogReg
import torch
import torch.nn as nn
import scipy.sparse as sp
from scipy.sparse import csr_matrix


def dgi(args, G, features, y, node_dict):
    """
        :param args: CLI arguments
        :param G: the graph
        :param features: the node features
        :param y: the node labels
        :param node_dict: node to integer mapping for faster computation
        """
    if not os.path.exists('output/models'):
        os.makedirs('output/models')

    A = nx.adjacency_matrix(G)

    # turn labels from string into integers
    encoder = LabelEncoder()
    y_transformed = encoder.fit_transform(y)
    num_classes = len(np.unique(y_transformed))

    features, labels, idx_train, idx_val, idx_test = preprocess_data_trunc(features, pd.Series(y_transformed))
    features = preprocess_features(features)

    A = preprocess_adj(A)
    A = csr_matrix(A)
    A = A.tocoo()  # convert to sparse

    if args.sparse:

        A = sparse_mx_to_torch_sparse_tensor(A)
    else:
        A = (A + sp.eye(A.shape[0])).todense()
        A = torch.FloatTensor(A[np.newaxis])
    features = torch.FloatTensor(features[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    model = DGI(features.shape[1], args.dimension, 'prelu')
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        features = features.cuda()
        A = A.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    # TRAINING PHASE
    num_nodes = len(G.nodes)
    for epoch in range(args.iter):
        model.train()
        optimiser.zero_grad()

        idx = np.random.permutation(num_nodes)
        shuf_fts = features[:, idx, :]
        lbl_1 = torch.ones(args.batch_size, num_nodes)
        lbl_2 = torch.zeros(args.batch_size, num_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()

        logits = model(features, shuf_fts, A if args.sparse else A, args.sparse, None, None, None)
        loss = b_xent(logits, lbl)
        print('Loss:', loss)

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'output/models/best_dgi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.early_stopping:
            print('Early stopping!')
            break

        loss.backward()
        optimiser.step()

    # EVALUATION PHASE. DOWNSTREAM TASK IS NODE CLASSIFICATION USING LOGISTIC REGRESSION
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_dgi.pkl'))

    embeds, _ = model.embed(features, A if args.sparse else A, args.sparse, None)
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    tot = torch.zeros(1)
    tot = tot.cuda()

    accs = []

    for _ in range(50):
        log = LogReg(args.dimension, num_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.cuda()

        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        print(acc)
        tot += acc

    print('Average accuracy:', tot / 50)

    accs = torch.stack(accs)
    print(accs.mean())
    print(accs.std())