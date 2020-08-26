#! /usr/bin/env python
# -*- coding: utf-8 -*-

import random
from . import graph
from gensim.models import Word2Vec


def deepWalk(args, G):

    print("Number of nodes: {}".format(len(G.nodes())))

    num_walks = len(G.nodes()) * args.number_walks

    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * args.walk_length

    print("Data size (walks*length): {}".format(data_size))

    print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                    path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))
    print("Training...")
    model = Word2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers)


    model.wv.save_word2vec_format(args.output)
