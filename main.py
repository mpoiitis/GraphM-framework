from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from algorithms.deepwalk.deepwalk import deepWalk
from algorithms.node2vec.node2vec import node2vec
from algorithms.mnmf.mnmf import mNMF
from algorithms.line.line import line
from algorithms.tadw.tadw_main import tadw
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
    line_parser.add_argument("--iter", default=100, type=int, help="Number of iterations for SGD. Default is 100")
    line_parser.add_argument("--dimension", default=128, type=int, help="Number of latent dimensions to learn for each node. Default is 128.")
    line_parser.add_argument("--batch-size", default=1000, type=int, help="Size of batch for the SGD. Default is 1000")
    line_parser.add_argument("--negative-sampling", default='uniform', type=str, choices=['uniform', 'non-uniform'], help="How to perform negative sampling. Default is 'uniform'")
    line_parser.add_argument("--negative-ratio", default=5, type=int, help="Parameter for negative sampling. Default is 5")

    tadw_parser = subparsers.add_parser('tadw', help='TADW method')
    tadw_parser.add_argument("--iter", default=200, type=int, help="Number of iterations for SGD. Default is 200")
    tadw_parser.add_argument("--dimension", default=32, type=int, help="Number of latent dimensions to learn for each node. Default is 32")
    tadw_parser.add_argument("--order", default=2, type=int, help="Target matrix approximation order. Default is 2")
    tadw_parser.add_argument("--lambd", default=1000.0, type=float, help="Regularization term coefficient. Default is 1000")
    tadw_parser.add_argument("--alpha", default=10**-6, type=float, help="Learning rate. Default is 10^-6")
    tadw_parser.add_argument("--features", default="sparse", type=str, choices=['dense', 'sparse'], help="Output embedding. Default is sparse")
    tadw_parser.add_argument("--lower-control", default=10 ** -15, type=float, help="Overflow control. Default is 10**-15")

    tsec_parser = subparsers.add_parser('tsec', help='TSEC method')
    tsec_parser.add_argument("--type", default='gcn', type=str, choices=['gcn, gcn_cheby'], help="Type of model. Default is gcn")
    tsec_parser.add_argument("--iter", default=200, type=int, help="Number of iterations for SGD. Default is 200")
    tsec_parser.add_argument("--layers", default=2, type=int, help="Number of graph convolutional layers. Default is 2")
    tsec_parser.add_argument("--hidden-dim", default=16, type=int, help="Number of units in the hidden layers. Default is 16")
    tsec_parser.add_argument("--max-degree", default=3, type=int, help="Maximum Chebyshev polynomial degree. Default is 3")
    tsec_parser.add_argument("--learning-rate", default=0.01, type=float, help="Initial learning rate. Default is 0.01")
    tsec_parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate (1 - keep probability). Default is 0.5")
    tsec_parser.add_argument("--weight-decay", default=5e-4, type=float, help="Weight for L2 loss on embedding matrix. Default is 5e-4")
    tsec_parser.add_argument("--early-stopping", default=10, type=int, help="Tolerance for early stopping (# of epochs). Default is 10")

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
    elif args.method == 'tadw':
        if not x:
            print('Node features are required to run TADW.')
        else:
            tadw(args, G, x)
    else:
        pass