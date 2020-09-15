from tensorflow.keras.layers import Layer, Dense
import tensorflow as tf
from .inits import glorot


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = Dense(intermediate_dim, activation="relu")
        self.dense_mean = Dense(latent_dim)
        self.dense_log_var = Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = Dense(intermediate_dim, activation="relu")
        self.dense_output = Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class GraphConvolution(Layer):
    def __init__(self, input_dim, output_dim, bias=False, featureless=False, dropout=0., activation=tf.keras.activations.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout = dropout
        self.activation = activation
        self.bias = bias
        self.featureless = featureless

        self.weights_ = glorot([input_dim, output_dim], name='weights')
        if self.bias:
            self.bias = self.add_weight(name='bias', shape=[output_dim])


    def call(self, inputs):
        x, a = inputs
        # x = tf.nn.dropout(x, self.dropout)

        # convolve
        if not self.featureless:  # if it has features x
            pre_h = tf.matmul(x, self.weights_)
        else:
            pre_h = self.weights_
        output = tf.matmul(a, pre_h)
        print(a.shape, x.shape, self.weights_.shape, pre_h.shape, output.shape)

        # bias
        if self.bias:
            output += self.bias

        return self.activation(output)
