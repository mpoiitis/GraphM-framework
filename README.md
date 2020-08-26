# GraphM-framework
A framework for network embeddings for the task of Network Representation Learning


## DeepWalk

DeepWalk uses short random walks to learn representations for vertices in graphs.

### Usage

**Example Usage**
    ``$python main.py deepwalk --format adjlist --input data/karate.adjlist --number-walks 80 --representation-size 128 --walk-length 40 --window-size 10
    --workers 1 --output output/karate.embeddings``

The parameters specified here are the same as in the paper.

**--input**:  *input_filename*

    1. ``--format adjlist`` for an adjacency list, e.g::

        1 2 3 4 5 6 7 8 9 11 12 13 14 18 20 22 32
        2 1 3 4 8 14 18 20 22 31
        3 1 2 4 8 9 10 14 28 29 33
        ...
    
    2. ``--format edgelist`` for an edge list, e.g::
    
        1 2
        1 3
        1 4
        ...

**--output**: *output_filename*

    The output representations in skipgram format - first line is header, all other lines are node-id and *d* dimensional representation:

        34 64
        1 0.016579 -0.033659 0.342167 -0.046998 ...
        2 -0.007003 0.265891 -0.351422 0.043923 ...
        ...

# Notes

Gensim package has a minor modification on gensim/models/word2vec.py, line 1704 where:

    wv.vectors[i] = self.seeded_vector(wv.index2word[i] + str(self.seed), wv.vector_size)
    is replaced by
    wv.vectors[i] = self.seeded_vector(wv.index2word[i] + self.seed, wv.vector_size)