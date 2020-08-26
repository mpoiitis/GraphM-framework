from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from algorithms.deepwalk.deepwalk import deepWalk
from algorithms.node2vec.node2vec import node2vec
from readData import load_graph

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    subparsers = parser.add_subparsers(dest='method', help="deepWalk. More algorithms will be implemented")

    deepwalk_parser = subparsers.add_parser('deepwalk', help='DeepWalk method')
    deepwalk_parser.add_argument('--input', required=True, choices=['karate', 'nutella', 'amherst', 'hamilton', 'mich', 'rochester', 'facebook', 'cora', 'citeseer'],
            help="Input graph dataset. Options: ['karate', 'nutella', 'amherst', 'hamilton', 'mich', 'rochester', 'facebook', 'cora', 'citeseer']")
    deepwalk_parser.add_argument('--output', required=True, help='Output representation file path')
    deepwalk_parser.add_argument('--dimension', default=64, type=int, help='Number of latent dimensions to learn for each node.')
    deepwalk_parser.add_argument('--num-walks', default=10, type=int, help='Number of random walks to start at each node')
    deepwalk_parser.add_argument('--walk-length', default=40, type=int, help='Length of the random walk started at each node')
    deepwalk_parser.add_argument('--window-size', default=5, type=int, help='Window size of skipgram model.')
    deepwalk_parser.add_argument('--seed', default=0, type=int, help='Seed for random walk generator.')
    deepwalk_parser.add_argument('--directed', default=False, type=bool, help='Graph is (un)directed. Default is False.')
    deepwalk_parser.add_argument('--workers', default=1, type=int, help='Number of parallel processes.')

    node2vec_parser = subparsers.add_parser('node2vec', help='Node2Vec method')
    node2vec_parser.add_argument('--input', required=True, choices=['karate', 'nutella', 'amherst', 'hamilton', 'mich', 'rochester', 'facebook', 'cora', 'citeseer'],
                                 help="Input graph dataset. Options: ['karate', 'nutella', 'amherst', 'hamilton', 'mich', 'rochester', 'facebook', 'cora', 'citeseer']")
    node2vec_parser.add_argument('--output', nargs='?', required=True, help='Output representation file path')
    node2vec_parser.add_argument('--dimension', default=128, type=int, help='Number of latent dimensions to learn for each node. Default is 128.')
    node2vec_parser.add_argument('--num-walks', default=10, type=int, help='Number of random walks to start at each node. Default is 10.')
    node2vec_parser.add_argument('--walk-length', default=80, type=int, help='Length of the random walk started at each node. Default is 80.')
    node2vec_parser.add_argument('--window-size', default=10, type=int, help='Window size of skipgram model. Default is 10.')
    node2vec_parser.add_argument('--p', default=1, type=float, help='Return hyperparameter. Default is 1.')
    node2vec_parser.add_argument('--q', default=1, type=float, help='Inout hyperparameter. Default is 1.')
    node2vec_parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD. Default is 1.')
    node2vec_parser.add_argument('--weighted', default=False, type=bool, help='Boolean specifying (un)weighted. Default is False.')
    node2vec_parser.add_argument('--directed', default=False, type=bool, help='Graph is (un)directed. Default is False.')
    node2vec_parser.add_argument('--workers', default=1, type=int, help='Number of parallel workers. Default is 1.')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    G, x, y = load_graph(args)

    if args.method == 'deepwalk':
        deepWalk(args, G)
    else:
        node2vec(args, G)