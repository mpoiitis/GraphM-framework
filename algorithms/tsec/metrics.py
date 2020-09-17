from tensorflow.keras.metrics import Metric
import tensorflow as tf

# Metric to measure accuracy of our model, optionally by class.

# This is needed as the way we're treating our data doesn't fit the Keras
# metrics. In a normal ML flow you have seperate lists of examples (e.g. input
# and expected label) for training, testing and validation. You feed one of these
# lists in to your training/evaluation loop and measure the model's loss and
# accuracy on those examples.

# Because our network is a graph, the input to the network is a tensor of node states
# and we need to pass the entire set of node states in for the adjacency matrix
# to match the shape of the node state matrix. We cannot just pass in the training
# nodes and their labels.

# Therefore instead we mask the labels output by the network to just the training
# or testing set of labels, and measure their accuracy. Keras's sample_weight
# mechanism doesn't get applied to accuracy metrics or during testing, therefore
# I've implemented our own metric.

# This metric has one additional optional feature: It will calculate accuracy for
# a single class.

# It's important to watch the accuracy by each class label to check the network
# is discriminating between them. If a network is struggling to train, one common
# failure case is it predicts the same class for all labels as an easy way to
# decrease loss. This often presents itself as a train accuracy of 100%/NUM_CLASSES.

# By watching individual class acurracies, we can see if the network is learning
# to predict each class, or sacrificing some/all classes for one class.


class AccuracyByClass(Metric):
    def __init__(self, name, dtype, class_label=None, sample_weight=None, y_true=None):
        '''
        Parameters:
          name: The name of this metric
          dtype: The type of the data being measured
          class_label: (Optional) If you supply this, the accuracy will be measured for just that class. Otherwise overall accuracy is measured
          sample_weight: The mask you want to apply to the labels
          y_true: (Optional) The correct values for the labels output by the network

        '''
        super().__init__(name, dtype)

        self.class_label = class_label
        self.sample_weight = tf.cast(sample_weight, dtype)
        self.y_true = y_true

        self.correct = tf.Variable(initial_value=0.0, dtype=self._dtype, trainable=False,
                                   name='correct_' + str(class_label))
        self.total = tf.Variable(initial_value=0.0, dtype=self._dtype, trainable=False,
                                 name='total_' + str(class_label))

    def reset_states(self):
        self.correct.assign(0)
        self.total.assign(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Note that Keras doesn't pass sample_weight into its metrics
        # during testing

        if self.y_true is not None:
            y_true = self.y_true

        y_true_classes = tf.argmax(y_true, axis=-1)
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        correct = tf.equal(y_pred_classes, y_true_classes)

        # Create mask
        if self.class_label is not None:
            mask = tf.cast(tf.equal(y_true_classes, self.class_label), self._dtype)
        else:
            mask = tf.ones(tf.shape(y_true_classes), self._dtype, 'mask_class_true_ones')

        sample_weight = self.sample_weight

        if sample_weight is not None:
            mask *= sample_weight

        # Apply mask
        masked_total_count = tf.reduce_sum(mask)
        self.total.assign_add(masked_total_count)

        masked_correct = mask * tf.cast(correct, self._dtype)
        masked_correct_count = tf.reduce_sum(masked_correct)
        self.correct.assign_add(masked_correct_count)

        return self.result()

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)
