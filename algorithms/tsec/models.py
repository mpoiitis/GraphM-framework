import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from .layers import Encoder, Decoder, GraphConvolution


class VariationalAutoEncoder(Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        original_dim,
        intermediate_dim=64,
        latent_dim=32,
        name="autoencoder",
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed


class GCN(Model):

    def __init__(self, dimension=32, num_classes=2, dropout=0., weight_decay=0., adj=None, name="gcn", **kwargs):
        super(GCN, self).__init__(name=name, **kwargs)
        self.dimension = dimension
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.adj = adj

        self.gc1 = GraphConvolution(self.adj, units=self.dimension, activation=tf.keras.backend.relu, dropout=self.dropout, l2=self.weight_decay, name='gc1')
        self.gc2 = GraphConvolution(self.adj, units=self.dimension, activation=tf.keras.backend.relu, dropout=self.dropout, l2=self.weight_decay, name='gc2')
        self.dense = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        inputs = self.gc1(inputs)
        inputs = self.gc2(inputs)
        output = self.dense(inputs)

        return output
