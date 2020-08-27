import os
import community
import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
from tqdm import tqdm


class MNMF:
    """
    Modularity regularized non-negative matrix factorization machine class.
    The calculations use Tensorflow.
    """
    def __init__(self, args, G):
        """
        Method to parse the graph setup the similarity matrices.
        Embedding matrices and cluster centers.
        :param args: Object with parameters.
        """
        print("Model initialization started.\n")
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():
            self.args = args
            self.G = G
            self.number_of_nodes = len(nx.nodes(self.G))
            if self.number_of_nodes > 10000:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

            self.S_0 = tf.compat.v1.placeholder(tf.float64, shape=(None, None))
            self.B1 = tf.compat.v1.placeholder(tf.float64, shape=(None, None))
            self.B2 = tf.compat.v1.placeholder(tf.float64, shape=(None, None))

            self.M = tf.Variable(tf.random.uniform([self.number_of_nodes, self.args.dimension],
                                                   0, 1, dtype=tf.float64))
            self.U = tf.Variable(tf.random.uniform([self.number_of_nodes, self.args.dimension],
                                                   0, 1, dtype=tf.float64))
            self.H = tf.Variable(tf.random.uniform([self.number_of_nodes, self.args.clusters],
                                                   0, 1, dtype=tf.float64))
            self.C = tf.Variable(tf.random.uniform([self.args.clusters, self.args.dimension],
                                                   0, 1, dtype=tf.float64))

            self.S = np.float64(self.args.eta)*self.S_0 + self.B1
            self.init = tf.compat.v1.global_variables_initializer()

    def build_graph(self):
        """
        Defining the M-NMF computation graph based on the power iteration method.
        The procedure has 4 separate phases:
        1. Updating the base matrix.
        2. Updating the embedding.
        3. Updating the cluster centers.
        4. Updating the membership of nodes.
        """

        # 1. Phase
        self.enum_1 = tf.matmul(self.S, self.U, a_is_sparse=True)
        self.denom_1 = tf.matmul(self.M, tf.matmul(self.U, self.U, transpose_a=True))
        self.denom_2 = tf.maximum(np.float64(self.args.lower_control), self.denom_1)
        self.M = self.M.assign(tf.nn.l2_normalize(tf.multiply(self.M, self.enum_1/self.denom_2), 1))

        # 2. Phase
        self.enum_2 = tf.matmul(self.S, self.M, transpose_a=True, a_is_sparse=True)+self.args.alpha*tf.matmul(self.H, self.C)
        self.denom_3 = tf.matmul(self.U, tf.matmul(self.M, self.M, transpose_a=True)+self.args.alpha*tf.matmul(self.C, self.C, transpose_a=True))
        self.denom_4 = tf.maximum(np.float64(self.args.lower_control), self.denom_3)
        self.U = self.U.assign(tf.nn.l2_normalize(tf.multiply(self.U, self.enum_2/self.denom_4), 1))

        # 3. Phase
        self.enum_3 = tf.matmul(self.H, self.U, transpose_a=True)
        self.denom_5 = tf.matmul(self.C, tf.matmul(self.U, self.U, transpose_a=True))
        self.denom_6 = tf.maximum(np.float64(self.args.lower_control), self.denom_5)
        self.C = self.C.assign(tf.nn.l2_normalize(tf.multiply(self.C, self.enum_3/self.denom_6), 1))

        # 4. Phase
        self.B1H = tf.matmul(self.B1, self.H, a_is_sparse=True)
        self.B2H = tf.matmul(self.B2, self.H, a_is_sparse=True)
        self.HHH = tf.matmul(self.H, (tf.matmul(self.H, self.H, transpose_a=True)))
        self.UC = tf.matmul(self.U, self.C, transpose_b=True)
        self.rooted = tf.square(np.float64(2*self.args.beta)*self.B2H)+tf.multiply(np.float64(16*self.args.lambd)*self.HHH, (np.float64(2*self.args.beta)*self.B1H+np.float64(2*self.args.alpha)*self.UC+(np.float64(4*self.args.lambd-2*self.args.alpha))*self.H))
        self.sqroot_1 = tf.sqrt(self.rooted)
        self.enum_4 = np.float64(-2*self.args.beta)*self.B2H+self.sqroot_1
        self.denom_7 = np.float64(8*self.args.lambd)*self.HHH
        self.denom_8 = tf.maximum(np.float64(self.args.lower_control), self.denom_7)
        self.sqroot_2 = tf.sqrt(self.enum_4/self.denom_8)
        self.H = self.H.assign(tf.nn.l2_normalize(tf.multiply(self.H, self.sqroot_2), 1))

    def update_state(self, H):
        """
        Procedure to calculate the cluster memberships and modularity.
        :param H: Cluster membership indicator.
        :return current_modularity: Modularity based on the cluster memberships.
        """
        indices = np.argmax(H, axis=1)
        indices = {list(self.G.nodes)[i]: int(indices[i]) for i, _ in enumerate(indices)}
        print(indices)
        current_modularity = community.modularity(indices, self.G)
        if current_modularity > self.best_modularity:
            self.best_modularity = current_modularity
            self.optimal_indices = indices
            self.stop_index = 0
        else:
            self.stop_index = self.stop_index + 1
        return current_modularity

    def initiate_dump(self, session, feed_dict):
        """
        Method to save the clusters and node representations to disk
        """
        cluster_folder = "/".join(self.args.cluster_mean_output.split('/')[:-1])
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)

        cols = ["X_"+ str(x) for x in range(self.args.dimension)]
        self.optimal_clusters = pd.DataFrame(session.run(self.C, feed_dict=feed_dict),
                                             columns=cols)
        self.optimal_node_representations = pd.DataFrame(session.run(self.U, feed_dict=feed_dict), columns=cols)
        self.optimal_clusters.to_csv(self.args.cluster_mean_output, index=None)
        self.optimal_node_representations.to_csv(self.args.output, index=None)

    def optimize(self):
        """
        Method to run the optimization and halt it when overfitting started.
        The output matrices are all saved when optimization has finished.
        """
        self.best_modularity = 0
        self.stop_index = 0
        with tf.compat.v1.Session(graph=self.computation_graph) as session:
            self.init.run()
            print("Optimization started.\n")
            self.build_graph()
            feed_dict = {self.S_0: overlap_generator(self.G), self.B1: np.array(nx.adjacency_matrix(self.G).todense()), self.B2:modularity_generator(self.G)}
            for i in tqdm(range(self.args.iter)):
                H = session.run(self.H, feed_dict=feed_dict)
                current_modularity = self.update_state(H)
                if self.stop_index > self.args.early_stopping:
                    break
            if self.args.dump_matrices:
                self.initiate_dump(session, feed_dict)


def overlap_generator(G):
    """
    Function to generate a neighbourhood overlap matrix (second-order proximity matrix).
    :param G: Graph object.
    :return laps: Overlap matrix.
    """
    print("Second order proximity calculation.\n")
    degs = nx.degree(G)
    sets = {node:set(G.neighbors(node)) for node in nx.nodes(G)}
    laps = np.array([[float(len(sets[n_1].intersection(sets[n_2])))/(float(degs[n_1]*degs[n_2])**0.5) if n_1 != n_2 else 0.0 for n_1 in nx.nodes(G)] for n_2 in tqdm(nx.nodes(G))], dtype=np.float64)
    return laps


def modularity_generator(G):
    """
    Function to generate a modularity matrix.
    :param G: Graph object.
    :return modu: Modularity matrix.
    """
    print("Modularity calculation.\n")
    degs = nx.degree(G)
    e_count = len(nx.edges(G))
    modu = np.array([[float(degs[n_1]*degs[n_2])/(2*e_count) for n_1 in nx.nodes(G)] for n_2 in tqdm(nx.nodes(G))], dtype=np.float64)
    return modu


def mNMF(args, G):
    model = MNMF(args, G)
    model.optimize()
