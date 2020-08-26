# GraphM-framework
A framework for network embeddings for the task of Network Representation Learning


## DeepWalk

DeepWalk uses short random walks alongside Word2Vec to learn representations for vertices in graphs.

### Usage

**Parameters**
- input: Input graph dataset. Options: ['karate', 'nutella', 'amherst', 'hamilton', 'mich', 'rochester', 'facebook', 'cora', 'citeseer']. Required
- output: Output representation file path. Suggested: "output/name_of_the_embedding". Required
- dimension: Number of latent dimensions to learn for each node. Default: 64
- num-walks: Number of random walks to start at each node. Default: 10
- walk-length: Length of the random walk started at each node. Default: 40
- window-size: Window size of skipgram model. Default: 5
- seed: Seed for random walk generator. Default: 0
- directed: Graph is (un)directed. Default: False
- workers: Number of parallel processes. Default: 1

**Example Usage**
    ``$python main.py deepwalk --input karate --output output/karate.embeddings --num-walks 80 --dimension 128 --walk-length 40 --window-size 10 --workers 1 ``

The parameters specified here are the same as in the paper.

## Node2Vec

Node2Vec extends DeepWalk by introducing parameters p and q to allow BFS/DFS-like search during the short random walks alongside alias sampling for efficient sampling.

### Usage

**Parameters**
- input: Input graph dataset. Options: ['karate', 'nutella', 'amherst', 'hamilton', 'mich', 'rochester', 'facebook', 'cora', 'citeseer']. Required
- output: Output representation file path. Suggested: "output/name_of_the_embedding". Required
- dimension: Number of latent dimensions to learn for each node. Default: 128
- num-walks: Number of random walks to start at each node. Default: 10
- walk-length: Length of the random walk started at each node. Default: 80
- window-size: Window size of skipgram model. Default: 10
- p: Return hyperparameter. Default: 1
- q: In-Out hyperparameter. Default: 1
- iter: Number of epochs in SGD. Default: 1
- weighted  : Boolean specifying (un)weighted. Default: False
- directed: Graph is (un)directed. Default: False
- workers: Number of parallel processes. Default: 1

**Example Usage**
    ``$python main.py node2vec --input karate --output output/karate.embeddings --num-walks 80 --dimension 128 --walk-length 40 --window-size 10 --workers 1 ``

The parameters specified here are the same as in the paper.

# Notes
Gensim package has a minor modification on gensim/models/word2vec.py, line 1704 where:

    wv.vectors[i] = self.seeded_vector(wv.index2word[i] + str(self.seed), wv.vector_size)
    is replaced by
    wv.vectors[i] = self.seeded_vector(str(wv.index2word[i]) + str(self.seed), wv.vector_size)