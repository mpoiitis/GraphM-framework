#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time
import random
import networkx as nx


def random_walk(G, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.
        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.nodes()))]

    while len(path) < path_length:
      current = path[-1]
      neighbors = list(G.neighbors(current))
      if len(neighbors) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(neighbors))
        else:
          path.append(path[0])
      else:
        break
    return [node for node in path]


def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                          rand=random.Random(0)):
    walks = []

    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(G, path_length, rand=rand, alpha=alpha, start=node))

    return walks
