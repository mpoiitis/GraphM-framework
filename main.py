from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from algorithms.deepwalk.deepwalk import deepWalk
from algorithms.node2vec.node2vec import node2vec


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    subparsers = parser.add_subparsers(dest='method', help="deepWalk. More algorithms will be implemented")

    deepwalk_parser = subparsers.add_parser('deepwalk', help='DeepWalk method')
    deepwalk_parser.add_argument('--format', default='adjlist', help='File format of input file')
    deepwalk_parser.add_argument('--input', nargs='?', required=True, help='Input graph file')
    deepwalk_parser.add_argument('--output', required=True, help='Output representation file')
    deepwalk_parser.add_argument('--representation-size', default=64, type=int, help='Number of latent dimensions to learn for each node.')
    deepwalk_parser.add_argument('--number-walks', default=10, type=int, help='Number of random walks to start at each node')
    deepwalk_parser.add_argument('--walk-length', default=40, type=int, help='Length of the random walk started at each node')
    deepwalk_parser.add_argument('--window-size', default=5, type=int, help='Window size of skipgram model.')
    deepwalk_parser.add_argument('--seed', default=0, type=int, help='Seed for random walk generator.')
    deepwalk_parser.add_argument('--undirected', default=True, type=bool, help='Treat graph as undirected.')
    deepwalk_parser.add_argument('--vertex-freq-degree', default=False, action='store_true', help='Use vertex degree to'
            'estimate the frequency of nodes in the random walks. This option is faster than calculating the vocabulary.')
    deepwalk_parser.add_argument('--workers', default=1, type=int, help='Number of parallel processes.')

    node2vec_parser = subparsers.add_parser('node2vec', help='Node2Vec method')
    node2vec_parser.add_argument('--format', default='adjlist', help='File format of input file')
    node2vec_parser.add_argument('--input', nargs='?', required=True, help='Input graph file')
    node2vec_parser.add_argument('--output', nargs='?', required=True, help='Output representation file')
    node2vec_parser.add_argument('--representation-size', type=int, default=128, help='Number of dimensions. Default is 128.')
    node2vec_parser.add_argument('--num-walks', type=int, default=10, help='Number of walks per source. Default is 10.')
    node2vec_parser.add_argument('--walk-length', type=int, default=80, help='Length of walk per source. Default is 80.')
    node2vec_parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 10.')
    node2vec_parser.add_argument('--p', type=float, default=1, help='Return hyperparameter. Default is 1.')
    node2vec_parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter. Default is 1.')
    node2vec_parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')
    node2vec_parser.add_argument('--weighted', dest='weighted', action='store_true', help='Boolean specifying (un)weighted. Default is unweighted.')
    node2vec_parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    node2vec_parser.set_defaults(weighted=False)
    node2vec_parser.add_argument('--directed', dest='directed', action='store_true', help='Graph is (un)directed. Default is undirected.')
    node2vec_parser.add_argument('--undirected', dest='undirected', action='store_false')
    node2vec_parser.set_defaults(directed=False)
    node2vec_parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers. Default is 8.')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    if args.method == 'deepwalk':
        deepWalk(args)
    else:
        node2vec(args)