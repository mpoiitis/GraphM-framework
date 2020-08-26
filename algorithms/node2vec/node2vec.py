from gensim.models import Word2Vec
from . import graph

def learn_embeddings(walks, args):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=args.dimension, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)
    model.wv.save_word2vec_format(args.output)

    return


def node2vec(args, G):

    n2v_G = graph.Graph(G, args.directed, args.p, args.q)
    n2v_G.preprocess_transition_probs()
    walks = n2v_G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks, args)
