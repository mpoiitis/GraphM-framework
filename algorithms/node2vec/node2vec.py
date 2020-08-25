from gensim.models import Word2Vec
from . import graph

def learn_embeddings(walks, args):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)
    model.save_word2vec_format(args.output)

    return

def node2vec(args):
    if args.format == "adjlist":
        nx_G = graph.load_adjacencylist(args.input, directed=args.directed, weighted= args.weighted)
    elif args.format == "edgelist":
        nx_G = graph.load_edgelist(args.input, directed=args.directed, weighted= args.weighted)
    else:
        raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist'" % args.format)

    G = graph.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks, args)
