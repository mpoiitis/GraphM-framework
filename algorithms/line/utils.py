import numpy as np
from scipy.sparse import csr_matrix
import random
import keras.backend as K
import math
import csv
from sklearn.preprocessing import LabelEncoder


def transform_data(G):
    """
    Transforms graph data to fit Keras shape
    """
    adj_data = dict(G.adjacency())
    adj_list = list()
    for k, v in adj_data.items():
        for n in v.keys():  # remove attributes nodes
            adj_list.append([k, n])
    adj_list = np.asarray(adj_list, dtype=np.int32)

    labeler = LabelEncoder()
    labeler.fit(list(set(adj_list.ravel())))

    adj_list = (labeler.transform(adj_list.ravel())).reshape(-1, 2)

    return adj_list


def LINE_loss(y_true, y_pred):
    coeff = y_true*2 - 1
    return -K.mean(K.log(K.sigmoid(coeff*y_pred)))


def batchgen_train(adj_list, numNodes, batch_size, negativeRatio, negative_sampling):

    table_size = 1e8
    power = 0.75
    sampling_table = None

    data = np.ones((adj_list.shape[0]), dtype=np.float32)
    mat = csr_matrix((data, (adj_list[:,0], adj_list[:,1])), shape = (numNodes, numNodes), dtype=np.float32)
    batch_size_ones = np.ones((batch_size), dtype=np.int32)

    nb_train_sample = adj_list.shape[0]
    index_array = np.arange(nb_train_sample)

    nb_batch = int(np.ceil(nb_train_sample / float(batch_size)))
    batches = [(i * batch_size, min(nb_train_sample, (i + 1) * batch_size)) for i in range(0, nb_batch)]

    if negative_sampling == "NON-UNIFORM":
        print("Pre-procesing for non-uniform negative sampling!")
        node_degree = np.zeros(numNodes)

        for i in range(len(adj_list)):
            node_degree[adj_list[i,0]] += 1
            node_degree[adj_list[i,1]] += 1

        norm = sum([math.pow(node_degree[i], power) for i in range(numNodes)])

        sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        for j in range(numNodes):
            p += float(math.pow(node_degree[j], power)) / norm
            while i < table_size and float(i) / table_size < p:
                sampling_table[i] = j
                i += 1

    while 1:

        for batch_index, (batch_start, batch_end) in enumerate(batches):
            pos_edge_list = index_array[batch_start:batch_end]
            pos_left_nodes = adj_list[pos_edge_list, 0]
            pos_right_nodes = adj_list[pos_edge_list, 1]

            pos_relation_y = batch_size_ones[0:len(pos_edge_list)]

            neg_left_nodes = np.zeros(len(pos_edge_list)*negativeRatio, dtype=np.float32)
            neg_right_nodes = np.zeros(len(pos_edge_list)*negativeRatio, dtype=np.float32)

            neg_relation_y = np.zeros(len(pos_edge_list)*negativeRatio, dtype=np.float32)

            h = 0
            for i in pos_left_nodes:
                for k in range(negativeRatio):
                    rn = sampling_table[random.randint(0, table_size - 1)] if negative_sampling == "NON-UNIFORM" else random.randint(0, numNodes - 1)
                    while mat[i, rn] == 1 or i == rn:
                        rn = sampling_table[random.randint(0, table_size - 1)] if negative_sampling == "NON-UNIFORM" else random.randint(0, numNodes - 1)
                    neg_left_nodes[h] = i
                    neg_right_nodes[h] = rn
                    h += 1

            left_nodes = np.concatenate((pos_left_nodes, neg_left_nodes), axis=0)
            right_nodes = np.concatenate((pos_right_nodes, neg_right_nodes), axis=0)
            relation_y = np.concatenate((pos_relation_y, neg_relation_y), axis=0)

            yield ([left_nodes, right_nodes], [relation_y])


def write_embedding(args, nodes, embed_generator):
    """
    Writes embeddings to the output path
    """
    with open(args.output, 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter=',')
        initial_info = [len(nodes), args.dimension]
        tsv_output.writerow(initial_info)
        for k in nodes:
            x = embed_generator.predict_on_batch([np.asarray([k]), np.asarray([k])])
            x = x[0][0] + x[1][0]
            x = x.tolist()
            tsv_output.writerow(x)
