# GraphM-framework
A framework for network embeddings for the task of Network Representation Learning

## Implemented Algorithms

- DeepWalk [[1]](#1)
- Node2Vec [[2]](#2)
- M-NMF [[8]](#8)
- LINE [[9]](#9)
- TADW [[10]](#10)
- TSEC

## Generic Parameters
- input: Input graph dataset. Options: ['karate', 'nutella', 'amherst', 'hamilton', 'mich', 'rochester', 'facebook', 'cora', 'citeseer']. Required
- output: Output representation file path. Suggested: "output/name_of_the_embedding". Required
- weighted  : Boolean specifying (un)weighted. Default: False
- directed: Graph is (un)directed. Default: False

## DeepWalk

DeepWalk[[1]](#1) uses short random walks alongside Word2Vec to learn representations for vertices in graphs.

### Usage

**Parameters**

- dimension: Number of latent dimensions to learn for each node. Default: 64
- num-walks: Number of random walks to start at each node. Default: 10
- walk-length: Length of the random walk started at each node. Default: 40
- window-size: Window size of skipgram model. Default: 5
- seed: Seed for random walk generator. Default: 0
- workers: Number of parallel processes. Default: 1

**Example Usage**
    ``$python main.py --input karate --output output/karate.embeddings deepwalk --num-walks 80 --dimension 128 --walk-length 40 --window-size 10 --workers 1 ``

The parameters specified here are the same as in the paper.

## Node2Vec

Node2Vec[[2]](#2) extends DeepWalk by introducing parameters p and q to allow BFS/DFS-like search during the short random walks alongside alias sampling for efficient sampling.

### Usage

**Parameters**
- dimension: Number of latent dimensions to learn for each node. Default: 128
- num-walks: Number of random walks to start at each node. Default: 10
- walk-length: Length of the random walk started at each node. Default: 80
- window-size: Window size of skipgram model. Default: 10
- p: Return hyperparameter. Default: 1
- q: In-Out hyperparameter. Default: 1
- iter: Number of epochs in SGD. Default: 1
- workers: Number of parallel processes. Default: 1

**Example Usage**
    ``$python main.py --input karate --output output/karate.embeddings node2vec --num-walks 80 --dimension 128 --walk-length 40 --window-size 10 --workers 1 ``

The parameters specified here are the same as in the paper.

## Modularized Nonnegative Matrix Factorization (M-NMF)

M-NMF[[8]](#8) incorporates the community structure into network embedding. Additionally, it captures 1st and 2nd order proximity but it can easily extend to higher-order proximities as well.

### Usage

**Parameters**
- dimension: Number of latent dimensions to learn for each node. Default: 16
- clusters: Number of clusters. Default: 20
- lambd: Weight of the cluster membership constraint. Default: 0.2
- alpha: Weight of clustering cost. Default: 0.05
- beta: Weight of modularity cost. Default: 0.05
- eta: Weight of second order similarities. Default: 5.0
- iter: Number of weight updates. Default: 200
- early-stopping: Number of iterations to do after reaching the best modularity value. Default: 3
- lower-control: Lowest possible component value. Default: 10 ** -15

**Example Usage**
    ``$python main.py --input karate --output output/karate.embeddings mnmf --dimension 80 --cluster 10 --lambd 0.1 --alpha 0.1 --beta 0.1 --eta 3.0 --iter 100 ``

The parameters specified here are the same as in the paper.

## Modularized Nonnegative Matrix Factorization (M-NMF)

LINE[[9]](#9) is able to embed very large-scale information networks. It is suitable for a variety of networks including directed, undirected, binary or weighted edges.

### Usage

**Parameters**
- iter: Number of iterations for SGD. Default: 100
- dimension: Number of latent dimensions to learn for each node. Default: 128
- batch-size: Size of batch for the SGD. Default: 1000
- negative-sampling: How to perform negative sampling. Default: 'uniform'
- negative-ratio: Parameter for negative sampling. Default: 5

**Example Usage**
    ``$python main.py --input karate --output output/karate.embeddings line --dimension 80 --iter 100  --batch-size 300 --negative-sampling non-uniform --negative-ratio 2``

The parameters specified here are the same as in the paper.

## Text-Associated DeepWalk (TADW)

TADW[[10]](#10) learns an embedding of nodes and fuses the node representations with node attributes. Learns the joint feature-proximal representations using regularized non-negative matrix factorization. TADW requires features to run, hence datasets Karate and p2p-Gnutella are not compatible.

### Usage

**Parameters**

- iter: Number of iterations for SGD. Default: 200
- dimension: Number of latent dimensions to learn for each node. Default: 32 (the final embedding will have 2 * 32 as two vectors are concatenated)
- lambd: Regularization term coefficient. Default: 1000
- alpha: Learning rate. Default: 10^-6
- order: Target matrix approximation order. Default: 2
- features: Output embedding. Default: sparse
- lower-control: Overflow control. Default: 10**-15

**Example Usage**
    ``$python main.py --input karate --output output/karate.embeddings tadw --features dense --iter 50 --dimension 128 --lambd 500 --alpha 1000 --order 1 ``

## Temporal Self-Supervised Embedding for Clustering (TSEC)

TSEC can be applied on every dataset for which there is available feature information. By exploiting either Graph Convolutional Neural Networks or Variational Autoencoders it will be trained through self-supervision to learn embeddings fine-tuned towards clustering.

### Usage

**Parameters**

- iter: Number of iterations for SGD. Default: 200
- dimension: Number of latent dimensions to learn for each node. Default: 128
- nn: The type of neural network. Default: 'gcn'
- type: The type of the adjacency matrix. Default: 'gcn'
- hidden-dim: The dimension of the latent layer. Currently used only on Variational Autoencoder method. Default: 512
- batch-size: Size of batch for the SGD. Default: 128
- max-degree: Maximum Chebyshev polynomial degree. Default: 3
- learning-rate: Initial learning rate. Default: 0.001
- dropout: Dropout rate (1 - keep probability). Default: 0.5
- weight-decay: Weight for L2 loss on embedding matrix. E.g. 0.008. Default: 0.0
- early-stopping: Tolerance for early stopping (# of epochs). E.g. 10. Default: None

**Example Usage**
    ``$python main.py --input karate --output output/karate.embeddings tsec --nn gcn --iter 50 --dimension 128 --batch-size 100 --learning-rate 0.01 --early-stopping 10 ``


# Benchmark Datasets
- Karate [[3]](#3)
- p2p-Gnutella [[4]](#4)
- Amherst [[5]](#5)
- Hamilton [[5]](#5)
- Mich [[5]](#5)
- Rochester [[5]](#5)
- Citeseer [[6]](#6)
- Cora [[6]](#6)
- Facebook [[7]](#7)

# Notes
Gensim package has a minor modification on gensim/models/word2vec.py, line 1704 where:

    wv.vectors[i] = self.seeded_vector(wv.index2word[i] + str(self.seed), wv.vector_size)
    is replaced by
    wv.vectors[i] = self.seeded_vector(str(wv.index2word[i]) + str(self.seed), wv.vector_size)

## References
<a id="1">[1]</a> 
Perozzi, B., Al-Rfou, R., & Skiena, S. (2014, August). Deepwalk: Online learning of social representations. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 701-710).

<a id="2">[2]</a> 
Grover, A., & Leskovec, J. (2016, August). node2vec: Scalable feature learning for networks. In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 855-864).

<a id="3">[3]</a> 
http://konect.cc/networks/ucidata-zachary/

<a id="4">[4]</a> 
https://snap.stanford.edu/data/p2p-Gnutella08.html

<a id="5">[5]</a> 
https://escience.rpi.edu/data/DA/fb100/

<a id="6">[6]</a> 
https://linqs.soe.ucsc.edu/data

<a id="7">[7]</a> 
http://snap.stanford.edu/data/egonets-Facebook.html

<a id="8">[8]</a> 
Wang, X., Cui, P., Wang, J., Pei, J., Zhu, W., & Yang, S. (2017, February). Community preserving network embedding. In AAAI (Vol. 17, pp. 203-209).

<a id="9">[9]</a> 
Tang, J., Qu, M., Wang, M., Zhang, M., Yan, J., & Mei, Q. (2015, May). Line: Large-scale information network embedding. In Proceedings of the 24th international conference on world wide web (pp. 1067-1077).

<a id="10">[10]</a>
Yang, C., Liu, Z., Zhao, D., Sun, M., & Chang, E. Y. (2015, July). Network representation learning with rich text information. In IJCAI (Vol. 2015, pp. 2111-2117).