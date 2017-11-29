import os
import sys
root_path = os.path.abspath("../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple

from _Dist.NeuralNetworks.NNUtil import DNDF


class LSTMCell(tf.contrib.rnn.LSTMCell):
    def __str__(self):
        return "LSTMCell"


class BasicLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    def __str__(self):
        return "BasicLSTMCell"


class CustomLSTMCell(tf.contrib.rnn.BasicRNNCell):
    def __init__(self, *args, **kwargs):
        super(CustomLSTMCell, self).__init__(*args, **kwargs)
        self._n_batch_placeholder = tf.placeholder(tf.int32, [], "n_batch_placeholder")

    def __str__(self):
        return "CustomLSTMCell"

    def __call__(self, x, state, scope="LSTM"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            s_old, h_old = state
            net = tf.concat([x, s_old], 1)

            w = tf.get_variable(
                "W", [net.shape[1].value, 4 * self._num_units], tf.float32,
                tf.contrib.layers.xavier_initializer()
            )
            b = tf.get_variable(
                "b", [4 * self._num_units], tf.float32,
                tf.zeros_initializer()
            )
            gates = tf.nn.xw_plus_b(net, w, b)

            r1, g1, g2, g3 = tf.split(gates, 4, 1)
            r1, g1, g3 = tf.nn.sigmoid(r1), tf.nn.sigmoid(g1), tf.nn.sigmoid(g3)
            g2 = tf.nn.tanh(g2)
            h_new = h_old * r1 + g1 * g2
            s_new = tf.nn.tanh(h_new) * g3
            return s_new, LSTMStateTuple(s_new, h_new)

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)


class DNDFCell(tf.contrib.rnn.BasicRNNCell):
    def __init__(self, *args, **kwargs):
        self._dndf = DNDF(reuse=True)
        self.n_batch_placeholder = kwargs.pop("n_batch_placeholder", None)
        if self.n_batch_placeholder is None:
            self.n_batch_placeholder = tf.placeholder(tf.int32, name="n_batch_placeholder")
        super(DNDFCell, self).__init__(*args, **kwargs)

    def __str__(self):
        return "DNDFCell"

    def __call__(self, x, state, scope="DNDFCell"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            s_old, h_old = state
            net = tf.concat([
                x,
                s_old,
                self._dndf(s_old, self.n_batch_placeholder, "feature"),
            ], 1)

            w = tf.get_variable(
                "W", [net.shape[1].value, 4 * self._num_units], tf.float32,
                tf.contrib.layers.xavier_initializer()
            )
            b = tf.get_variable(
                "b", [4 * self._num_units], tf.float32,
                tf.zeros_initializer()
            )
            gates = tf.nn.xw_plus_b(net, w, b)

            r1, g1, g2, g3 = tf.split(gates, 4, 1)
            r1, g1, g3 = tf.nn.sigmoid(r1), tf.nn.sigmoid(g1), tf.nn.sigmoid(g3)
            g2 = tf.nn.tanh(g2)
            h_new = h_old * r1 + g1 * g2
            s_new = tf.nn.tanh(h_new) * g3

            return s_new, LSTMStateTuple(s_new, h_new)

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)


class CellFactory:
    @staticmethod
    def get_cell(name, n_hidden, **kwargs):
        if name == "LSTM" or name == "LSTMCell":
            cell = LSTMCell
        elif name == "BasicLSTM" or name == "BasicLSTMCell":
            cell = BasicLSTMCell
        elif name == "CustomLSTM" or name == "CustomLSTMCell":
            cell = CustomLSTMCell
        elif name == "DNDF" or name == "DNDFCell":
            cell = DNDFCell
        else:
            raise NotImplementedError("Cell '{}' not implemented".format(name))
        return cell(n_hidden, **kwargs)
