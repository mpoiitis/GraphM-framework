from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from algorithms.deepwalk.deepwalk import deepWalk

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    subparsers = parser.add_subparsers(dest='method', help="deepWalk. More algorithms will be implemented")

    deepwalk_parser = subparsers.add_parser('deepwalk', help='DeepWalk method')

    deepwalk_parser.add_argument('--format', default='adjlist', help='File format of input file')
    deepwalk_parser.add_argument('--input', nargs='?', required=True, help='Input graph file')
    deepwalk_parser.add_argument('--number-walks', default=10, type=int, help='Number of random walks to start at each node')
    deepwalk_parser.add_argument('--output', required=True, help='Output representation file')
    deepwalk_parser.add_argument('--representation-size', default=64, type=int, help='Number of latent dimensions to learn for each node.')
    deepwalk_parser.add_argument('--seed', default=0, type=int, help='Seed for random walk generator.')
    deepwalk_parser.add_argument('--undirected', default=True, type=bool, help='Treat graph as undirected.')
    deepwalk_parser.add_argument('--vertex-freq-degree', default=False, action='store_true', help='Use vertex degree to '
            'estimate the frequency of nodes in the random walks. This option is faster than calculating the vocabulary.')
    deepwalk_parser.add_argument('--walk-length', default=40, type=int, help='Length of the random walk started at each node')
    deepwalk_parser.add_argument('--window-size', default=5, type=int, help='Window size of skipgram model.')
    deepwalk_parser.add_argument('--workers', default=1, type=int, help='Number of parallel processes.')

    args = parser.parse_args()

    deepWalk(args)