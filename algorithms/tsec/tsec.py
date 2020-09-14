import tensorflow as tf
from sklearn.model_selection import train_test_split
import networkx as nx
import pandas as pd
from .utils import *
from numpy import sqrt
from tensorflow.keras.layers import Masking, Dense
from tensorflow.keras import Model
from .layers import GraphConvolution
from sklearn.preprocessing import LabelEncoder
from .models import VariationalAutoEncoder as VAE
import matplotlib.pyplot as plt
import os


def tsec(args, G, features, y, node_dict):
    """
    :param args: CLI arguments
    :param G: the graph
    :param features: the node features
    :param y: the node labels
    :param node_dict: node to integer mapping for faster computation
    """

    if args.input != "cora" and args.input != "citeseer":
        print('Only Cora and Citeseer datasets are supported!')
        return

    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.random.set_seed(seed)

    A = nx.adjacency_matrix(G)

    # Some preprocessing
    features = preprocess_features(features)
    if args.type == 'gcn':
        support = preprocess_adj(A)
        num_supports = 1
    elif args.type == 'gcn_cheby':
        support = chebyshev_polynomials(A, args.max_degree)
        num_supports = 1 + args.max_degree
    else:
        return

    x = np.hstack((features, support))  # horizontally stack feature and adjacency matrix together to split them equally
    encoder = LabelEncoder()
    y_transformed = encoder.fit_transform(y)
    num_labels = len(np.unique(y_transformed))

    X_train, X_test, y_train, y_test = train_test_split(x, y_transformed, test_size=0.33, random_state=42)

    model = VAE(X_train.shape[1], intermediate_dim=args.hidden_dim, latent_dim=args.dimension)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.early_stopping)  # add early stopping
    history = model.fit(X_train, X_train, epochs=args.iter, batch_size=args.batch_size, shuffle=True, validation_split=0.33, callbacks=[es], verbose=1)

    if not os.path.exists('output/models'):
        os.makedirs('output/models')

    model.save('output/models/{}_nn'.format(args.input))

    model = tf.keras.models.load_model('output/models/{}_nn'.format(args.input))
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

    # remove decoder layer to produce embeddings
    input = tf.keras.Input(shape=(X_train.shape[1],), name="encoder_input")
    encoder = model.layers[0](input)
    new_model = Model(inputs=input, outputs=encoder)
    new_model.build(input_shape=(X_train.shape[0], X_train.shape[1]))
    new_model.summary()
    result = new_model.predict(X_test)
    result = result[2]  # keep the actual embeddings

    # add initial node string name as the first column of the embedding
    list_with_index = list()
    for i, embedding in enumerate(result):
        l = list()
        l.append(list(node_dict.keys())[list(node_dict.values()).index(i)])
        # l.append(i) # un comment to save index instead of actual node string name
        l.extend(embedding)
        list_with_index.append(l)

    columns = ["id"] + ["X_" + str(dim) for dim in range(args.dimension)]
    df = pd.DataFrame(list_with_index, columns=columns)
    df.to_csv(args.output, index=None)

    # Create model
    # f = tf.keras.layers.Input(shape=(features.shape[1]))
    # s = tf.keras.layers.Input(shape=(support.shape[1]))
    # masking_1 = Masking(mask_value=0)(f)
    # masking_2 = Masking(mask_value=0)(s)
    # g1 = GraphConvolution(input_dim=features.shape[1], output_dim=args.hidden_dim, dropout=args.dropout, activation=tf.keras.activations.relu)((masking_1, masking_2))
    # g2 = GraphConvolution(input_dim=args.hidden_dim, output_dim=args.dimension, dropout=args.dropout, activation=tf.keras.activations.linear)((g1, masking_2))
    # out = Dense(num_labels, activation='softmax')(g2)
    # model = Model([f, s], out)
    # model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    #
    # model.summary()
    # model.fit((X_train[:, :features.shape[1]], X_train[:, features.shape[1]:]), y_train, epochs=args.iter, shuffle=True, validation_split=0.33, callbacks=[es], verbose=1)
    #
    # loss, acc = model.evaluate(X_test, y_test, verbose=0)
    # print('Categorical Cross Entropy: %.3f, Accuracy: %.3f' % (loss, acc))

