import networkx as nx
from time import time
import random
import numpy as np


class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
		Simulate a random walk starting from start node.
		'''

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(self.alias_nodes[cur][0], self.alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(self.alias_edges[(prev, cur)][0],
                                               self.alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
		Repeatedly simulate random walks from each node.
		'''

        walks = []
        nodes = list(self.G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
		Get the alias edge setup lists for a given edge.
		'''

        unnormalized_probs = []
        for dst_nbr in sorted(self.G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(self.G[dst][dst_nbr]['weight'] / self.p)
            elif self.G.has_edge(dst_nbr, src):
                unnormalized_probs.append(self.G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(self.G[dst][dst_nbr]['weight'] / self.q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
		Preprocessing of transition probabilities for guiding the random walks.
		'''

        alias_nodes = {}
        for node in self.G.nodes():
            unnormalized_probs = [self.G[node][nbr]['weight'] for nbr in sorted(self.G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if self.is_directed:
            for edge in self.G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in self.G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    '''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def load_edgelist(file, directed, weighted):
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
    else:
        G = nx.read_edgelist(file, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    t1 = time()
    print('Graph loaded in {}s'.format(t1 - t0))

    if not directed:
        G = G.to_undirected()

    return G


def load_adjacencylist(file, directed, weighted):
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
    else:
        G = nx.read_adjlist(file, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    t1 = time()
    print('Graph loaded in {}s'.format(t1 - t0))

    if not directed:
        G = G.to_undirected()

    return G