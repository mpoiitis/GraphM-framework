from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from algorithms.deepwalk.deepwalk import deepWalk
from algorithms.node2vec.node2vec import node2vec
from algorithms.mnmf.mnmf import mNMF
from algorithms.line.line import line
from readData import load_graph

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    subparsers = parser.add_subparsers(dest='method', help="deepWalk. More algorithms will be implemented")

    deepwalk_parser = subparsers.add_parser('deepwalk', help='DeepWalk method')
    deepwalk_parser.add_argument('--dimension', default=64, type=int, help='Number of latent dimensions to learn for each node.')
    deepwalk_parser.add_argument('--num-walks', default=10, type=int, help='Number of random walks to start at each node')
    deepwalk_parser.add_argument('--walk-length', default=40, type=int, help='Length of the random walk started at each node')
    deepwalk_parser.add_argument('--window-size', default=5, type=int, help='Window size of skipgram model.')
    deepwalk_parser.add_argument('--seed', default=0, type=int, help='Seed for random walk generator.')
    deepwalk_parser.add_argument('--workers', default=1, type=int, help='Number of parallel processes.')

    node2vec_parser = subparsers.add_parser('node2vec', help='Node2Vec method')
    node2vec_parser.add_argument('--dimension', default=128, type=int, help='Number of latent dimensions to learn for each node. Default is 128.')
    node2vec_parser.add_argument('--num-walks', default=10, type=int, help='Number of random walks to start at each node. Default is 10.')
    node2vec_parser.add_argument('--walk-length', default=80, type=int, help='Length of the random walk started at each node. Default is 80.')
    node2vec_parser.add_argument('--window-size', default=10, type=int, help='Window size of skipgram model. Default is 10.')
    node2vec_parser.add_argument('--p', default=1, type=float, help='Return hyperparameter. Default is 1.')
    node2vec_parser.add_argument('--q', default=1, type=float, help='Inout hyperparameter. Default is 1.')
    node2vec_parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD. Default is 1.')
    node2vec_parser.add_argument('--workers', default=1, type=int, help='Number of parallel workers. Default is 1.')

    mnmf_parser = subparsers.add_parser('mnmf', help='M-NMF method')
    mnmf_parser.add_argument("--cluster-mean-output", nargs="?",  default="output/mnmf/cluster_means/means.csv", help="Cluster means path.")
    mnmf_parser.add_argument("--dump-matrices", default=True, type=bool, help="Save the embeddings to disk or not.")
    mnmf_parser.add_argument('--dimension', default=16, type=int, help='Number of latent dimensions to learn for each node. Default is 16.')
    mnmf_parser.add_argument("--clusters", default=20, type=int, help="Number of clusters.")
    mnmf_parser.add_argument("--lambd", default=0.2, type=float, help="Weight of the cluster membership constraint.")
    mnmf_parser.add_argument("--alpha", default=0.05, type=float, help="Weight of clustering cost.")
    mnmf_parser.add_argument("--beta", default=0.05, type=float, help="Weight of modularity cost.")
    mnmf_parser.add_argument("--eta", default=5.0, type=float, help="Weight of second order similarities.")
    mnmf_parser.add_argument("--iter", default=200, type=int, help="Number of weight updates.")
    mnmf_parser.add_argument("--early-stopping", default=3, type=int, help="Number of iterations to do after reaching the best modularity value.")
    mnmf_parser.add_argument("--lower-control", default=10 ** -15, type=float, help="Lowest possible component value.")

    line_parser = subparsers.add_parser('line', help='LINE method')
    line_parser.add_argument("--iter", default=100, type=int, help="Number of iterations for SGD")
    line_parser.add_argument("--dimension", default=128, type=int, help="Number of latent dimensions to learn for each node. Default is 128.")
    line_parser.add_argument("--batch-size", default=1000, type=float, help="Size of batch for the SGD")
    line_parser.add_argument("--negative-sampling", default='uniform', type=str, choices=['uniform', 'non-uniform'], help="How to perform negative sampling.")
    line_parser.add_argument("--negative-ratio", default=5, type=int, help="")

    parser.add_argument('--input', required=True, choices=['karate', 'gnutella', 'amherst', 'hamilton', 'mich', 'rochester', 'facebook', 'cora', 'citeseer'],
                        help="Input graph dataset. Options: ['karate', 'gnutella', 'amherst', 'hamilton', 'mich', 'rochester', 'facebook', 'cora', 'citeseer']")
    parser.add_argument('--output', required=True, help='Output representation file path')
    parser.add_argument('--weighted', default=False, type=bool, help='Boolean specifying (un)weighted. Default is False.')
    parser.add_argument('--directed', default=False, type=bool, help='Graph is (un)directed. Default is False.')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    G, x, y = load_graph(args)

    if args.method == 'deepwalk':
        deepWalk(args, G)
    elif args.method == 'node2vec':
        node2vec(args, G)
    elif args.method == 'mnmf':
        mNMF(args, G)
    elif args.method == 'line':
        line(args, G)
    else:
        pass