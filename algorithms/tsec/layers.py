from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.regularizers import l2
import tensorflow as tf

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
    def __init__(self, adjacency, units, activation=tf.identity, dropout=0.0, l2=0.0, dtype=tf.float32, name=None):
        """
            Params:
              Adjacency: a tf.SparseTensor adjacency matrix
              Units: The number of output units per node state
              Activation: The activation function to apply to the node states
              Dropout: The amount of dropout (0.0 being none, 1.0 being all units) to apply
              l2: The amount of L2 regularisation to apply
              dtype: The type of values in the tensors this layer will transform
              name: The name of this layer
        """
        super(GraphConvolution, self).__init__(dtype=dtype, name=name)

        self.adjacency = adjacency
        self.units = units
        self.activation = activation
        self.dropout = dropout
        self.l2 = l2

        assert isinstance(adjacency, tf.SparseTensor), "Adjacency matrix should be a SparseTensor"
        assert adjacency.dtype == self.dtype, "Adjacency matrix not expected dtype, got " + str(
            adjacency.dtype) + " expected " + str(self.dtype)

    def build(self, input_shape):
        """
            This method is called during the initial compilation of our model. Its
            primary job is to initialize the weights for this layer.

            Params:
              input_shape: this is the shape of the input to the layer, in our case an
                           array of (NUMBER_NODES, INPUT_SIZE)

            Build one weight, w, which will be applied to each input. Initialize
            it from the uniform distribution, scaled by the size of the matrix. Apply
            l2 loss to regularize the matrix
        """
        self.w = self.add_weight(shape=(input_shape[1], self.units), dtype=self.dtype,
                                 initializer='glorot_uniform', regularizer=l2(self.l2))

    def call(self, inputs):
        """
            This method is called to apply the layer to an incoming tensor. This is the
            real meat of the model.

            Params:
              node_state: The tf.Tensor of node states.  Shape (NUMBER_NODES, NODE_STATE_SIZE)

            Returns: The transformed node state tf.Tensor
        """
        assert isinstance(inputs, tf.Tensor), "Layer input should be a Tensor, got " + str(type(inputs))
        assert inputs.dtype == self.dtype, "Input to layer " + str(self.name) + " wrong dtype, got " + str(inputs.dtype) + " expected " + str(self.dtype)
        tf.debugging.check_numerics(inputs, "Input to layer " + str(self.name) + " has numerical instability")

        # Dropout
        inputs = tf.nn.dropout(inputs, rate=self.dropout)
        # Apply the node convolution: This means to matrix: multiply each input by the learned parameters `self.w`
        print(inputs.shape, self.w.shape)
        inputs = tf.matmul(inputs, self.w)

        # Apply the graph propagation: This means to multiply the
        # normalized adjacency matrix by the inputs
        # You can do this as a single sparse_dense_matmul()

        print(self.adjacency.shape, inputs.shape)
        inputs = tf.sparse.sparse_dense_matmul(self.adjacency, inputs)
        print(inputs.shape)
        print('===================')
        output = self.activation(inputs)

        tf.debugging.check_numerics(output, "Output of layer " + str(self.name) + " has numerical instability")

        return output

