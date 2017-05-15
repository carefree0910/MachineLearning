import tensorflow as tf
import tensorflow.contrib.layers as layers


class LSTMCell(tf.contrib.rnn.BasicRNNCell):
    def __call__(self, x, state, scope="LSTM"):
        with tf.variable_scope(scope):
            s_old, h_old = tf.split(state, 2, 1)
            gates = layers.fully_connected(
                tf.concat([x, s_old], 1),
                num_outputs=4 * self._num_units,
                activation_fn=None)
            r1, g1, g2, g3 = tf.split(gates, 4, 1)
            r1, g1, g3 = tf.nn.sigmoid(r1), tf.nn.sigmoid(g1), tf.nn.sigmoid(g3)
            g2 = tf.nn.tanh(g2)
            h_new = h_old * r1 + g1 * g2
            s_new = tf.nn.tanh(h_new) * g3
            return s_new, tf.concat([s_new, h_new], 1)

    @property
    def state_size(self):
        return 2 * self._num_units
