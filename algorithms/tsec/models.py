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

    def __init__(self, original_dim, output_dim=32, num_labels=2, dropout=0., weight_decay=0., name="gcn", **kwargs):
        super(GCN, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.weight_decay = weight_decay

        self.layers_ = []
        self.layers_.append(GraphConvolution(original_dim, output_dim, dropout=dropout, activation=tf.keras.activations.relu, name='gc1'))
        self.layers_.append(GraphConvolution(output_dim, num_labels, dropout=dropout, activation=tf.keras.activations.softmax, name='gc2'))


    def call(self, inputs):
        x, a = inputs

        outputs = [x]

        for i, layer in enumerate(self.layers):
            hidden = layer((outputs[-1], a))
            outputs.append(hidden)

        output = outputs[-1]

        # Weight decay loss
        loss = tf.zeros([])
        for var in self.layers_[0].trainable_variables:
            loss += self.weight_decay * tf.nn.l2_loss(var)

        self.add_loss(loss)
        return output
