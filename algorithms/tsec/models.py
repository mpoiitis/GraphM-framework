import tensorflow as tf
from tensorflow.keras import Model
from .layers import GraphConvolution
from .metrics import masked_accuracy, masked_softmax_cross_entropy


class GCN(Model):
    def __init__(self, args, input_dim):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = args.dimension
        self.hidden_dim = args.hidden_dim
        self.learning_rate = args.learning_rate
        self.dropout = args.dropout
        self.weight_decay = args.weight_decay
        self.early_stopping = args.early_stopping

        self.layers = list()
        self.layers.append(GraphConvolution(input_dim=self.input_dim, output_dim=self.hidden_dim, activation=tf.nn.relu,
                                                    kernel_initializer='glorot_uniform', dropout=args.dropout))
        self.layers.append(GraphConvolution(input_dim=self.hidden_dim, output_dim=self.output_dim, activation=lambda x: x,
                                            kernel_initializer='glorot_uniform', dropout=args.dropout))


    def call(self, inputs, training=None):
        x, label, mask, support = inputs

        outputs = [x]
        for layer in self.layers:
            hidden = layer((outputs[-1], support), training)
            outputs.append(hidden)
        output = outputs[-1]

        # # Weight decay loss
        loss = tf.zeros([])
        for var in self.layers[0].trainable_variables:
            loss += self.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        loss += masked_softmax_cross_entropy(output, label, mask)

        acc = masked_accuracy(output, label, mask)

        return loss, acc