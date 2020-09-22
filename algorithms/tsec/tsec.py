import tensorflow as tf
import networkx as nx
import numpy as np
from utils import preprocess_adj, preprocess_features, chebyshev_polynomials, preprocess_data
from tensorflow.keras import Model
from .metrics import AccuracyByClass
from sklearn.preprocessing import LabelEncoder
from .models import VariationalAutoEncoder as VAE
from .models import GCN
import matplotlib.pyplot as plt
import os
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import pandas as pd


def display_train(history, num_classes):
    # Display graphs of how the model performed
    from scipy.ndimage.filters import gaussian_filter1d

    def smooth(key):
        return gaussian_filter1d(history.history[key], sigma=2)

    print("Final loss: ", history.history['loss'][-1])
    print("Final training accuracy: ", round(history.history['train_accuracy'][-1] * 100), "%")

    plt.rcParams['figure.figsize'] = [15, 10]

    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'])
    plt.title("Training loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.subplot(2, 2, 2)
    plt.plot(smooth('train_accuracy'))
    plt.title("Training accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    plt.subplot(2, 2, 3)
    plt.plot(smooth('val_accuracy'))
    plt.title("Validation accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    plt.subplot(2, 2, 4)
    for label in range(num_classes):
        plt.plot(smooth("train_class_acc_" + str(label)))
    plt.title("Training accuracy by class")
    plt.ylabel('Class accuracy')
    plt.xlabel('Epoch')
    plt.legend(['class_' + str(label) for label in range(num_classes)], loc='lower right')

    plt.tight_layout()
    plt.show()


def display_eval(history, results, model):
    # Get test metrics
    results_dict = dict(zip(model.metrics_names, results))
    for name in model.metrics_names:
        if "test" not in name:
            del results_dict[name]

    # Add train accuracy from the earlier training history
    results_dict["train_accuracy"] = history.history['train_accuracy'][-1]

    results_keys = list(results_dict.keys())
    results_values = [results_dict[key] for key in results_keys]

    plt.rcParams['figure.figsize'] = [15, 6]

    # Display a bar chart
    y_pos = np.arange(len(results_dict))
    plt.barh(y_pos, results_values, align='center', alpha=0.5)
    plt.yticks(y_pos, results_keys)
    plt.ylabel('Percentage')
    plt.title('Evaluation')
    plt.show()


def tsec(args, G, features, y, node_dict):
    """
    :param args: CLI arguments
    :param G: the graph
    :param features: the node features
    :param y: the node labels
    :param node_dict: node to integer mapping for faster computation
    """
    if not os.path.exists('output/models'):
        os.makedirs('output/models')

    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.random.set_seed(seed)

    A = nx.adjacency_matrix(G)

    # turn labels from string into integers
    encoder = LabelEncoder()
    y_transformed = encoder.fit_transform(y)
    num_classes = len(np.unique(y_transformed))

    if args.nn == 'autoencoder':
        features = preprocess_features(features)
        A = preprocess_adj(A)
        # horizontally stack feature and adjacency matrix together to split them equally
        x = np.hstack((features, A))
        X_train, X_test, y_train, y_test = train_test_split(x, y_transformed, test_size=0.33, random_state=42)

        model = VAE(X_train.shape[1], intermediate_dim=args.hidden_dim, latent_dim=args.dimension)

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.early_stopping)  # add early stopping
        model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

        history = model.fit(X_train, X_train, epochs=args.iter, batch_size=args.batch_size, shuffle=True,
                            validation_split=0.33, callbacks=[es] if args.early_stopping is not None else None, verbose=1)

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
        result = new_model.predict(X_test)
        result = result[2]  # keep the actual embeddings

    else:
        features, y_train, y_val, y_test, train_mask, val_mask, test_mask = preprocess_data(features, pd.Series(y_transformed))
        # Some preprocessing
        features = preprocess_features(features)
        if args.type == 'gcn':
            A = preprocess_adj(A)
        elif args.type == 'gcn_cheby':
            A = chebyshev_polynomials(A, args.max_degree)
        else:
            return

        A = csr_matrix(A)
        coo = A.tocoo()
        indices = np.array(list(zip(coo.row, coo.col)))
        tf_adj = tf.SparseTensor(indices=indices, values=tf.cast(A.data, tf.float32), dense_shape=A.shape)

        model = GCN(args.dimension, num_classes, args.dropout, args.weight_decay, tf_adj)

        # track specific metrics
        metrics = [
            AccuracyByClass("train_accuracy", tf.float32, sample_weight=train_mask, y_true=y_train),
            AccuracyByClass("test_accuracy", tf.float32, sample_weight=test_mask, y_true=y_test),
            AccuracyByClass("val_accuracy", tf.float32, sample_weight=val_mask, y_true=y_val)
        ]
        for label in range(num_classes):
            metrics.append(AccuracyByClass("train_class_acc_" + str(label), tf.float32, label, sample_weight=train_mask, y_true=y_train))
            metrics.append(AccuracyByClass("test_class_acc_" + str(label), tf.float32, label, sample_weight=test_mask, y_true=y_test))

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.early_stopping)  # add early stopping
        model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics)

        initial_state = features.todense()
        # no batch processing, following the original paper. since the adjacency matrix is the whole graph, we want to feed the whole input array in each training step
        history = model.fit(initial_state, y_train,
                            sample_weight=tf.cast(train_mask, tf.float32),  # This will be used in loss calculations
                            validation_data=(initial_state, y_val, val_mask),
                            epochs=args.iter,
                            batch_size=initial_state.shape[0],
                            shuffle=False,
                            callbacks=[es] if args.early_stopping is not None else None,
                            verbose=1)

        # display_plot(history, num_classes)

        # remove decoder layer to produce embeddings
        input = tf.keras.Input(shape=(initial_state.shape[1],), name="input")
        gc1 = model.layers[0](input)
        gc2 = model.layers[1](gc1)
        new_model = Model(inputs=input, outputs=gc2)
        # results = model.evaluate(initial_state, y_test, steps=1, batch_size=initial_state.shape[0], verbose=0)
        # display_eval(history, results, model)

        result = new_model.predict(initial_state, batch_size=initial_state.shape[0])

    model.save('output/models/{}_{}_trained'.format(args.input, args.nn))


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

