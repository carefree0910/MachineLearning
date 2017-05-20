import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.layers as layers
from tensorflow.contrib.rnn import BasicRNNCell, LSTMStateTuple


class LSTMCell(BasicRNNCell):
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


class FastLSTMCell(BasicRNNCell):
    def __call__(self, x, state, scope="LSTM"):
        with tf.variable_scope(scope):
            s_old, h_old = state
            gates = layers.fully_connected(
                tf.concat([x, s_old], 1),
                num_outputs=4 * self._num_units,
                activation_fn=None)
            r1, g1, g2, g3 = tf.split(gates, 4, 1)
            r1, g1, g3 = tf.nn.sigmoid(r1), tf.nn.sigmoid(g1), tf.nn.sigmoid(g3)
            g2 = tf.nn.tanh(g2)
            h_new = h_old * r1 + g1 * g2
            s_new = tf.nn.tanh(h_new) * g3
            return s_new, LSTMStateTuple(s_new, h_new)

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)


class RNNWrapper:
    def __init__(self):
        self._log = {}
        self._optimizer = self._generator = None
        self._tfx = self._tfy = self._input = self._output = None
        self._cell = self._im = self._om = self._hidden_units = None
        self._sess = tf.Session()

    def _verbose(self):
        x_test, y_test = self._generator.gen(0, True)
        y_pred = self.predict(x_test)  # type: np.ndarray
        print("Test acc: {:8.6} %".format(np.mean(np.argmax(y_test, axis=1) == y_pred) * 100))

    def _get_output(self, rnn_outputs):
        outputs = tf.reshape(rnn_outputs[..., -3:, :], [-1, self._hidden_units * 3])
        self._output = layers.fully_connected(
            outputs, num_outputs=self._om, activation_fn=tf.nn.sigmoid)

    def fit(self, im, om, generator, hidden_units=128, cell=LSTMCell):
        self._generator = generator
        self._im, self._om, self._hidden_units = im, om, hidden_units
        self._optimizer = tf.train.AdamOptimizer(0.01)
        self._input = self._tfx = tf.placeholder(tf.float32, shape=[None, None, im])
        self._tfy = tf.placeholder(tf.float32, shape=[None, om])

        self._cell = cell(self._hidden_units)
        rnn_outputs, _ = tf.nn.dynamic_rnn(
            self._cell, self._input,
            initial_state=self._cell.zero_state(tf.shape(self._input)[0], tf.float32)
        )
        self._get_output(rnn_outputs)
        loss = -tf.reduce_mean(
            self._tfy * tf.log(self._output + 1e-8) + (1 - self._tfy) * tf.log(1 - self._output + 1e-8)
        )
        train_step = self._optimizer.minimize(loss)
        self._log["iter_err"] = []
        self._log["epoch_err"] = []
        self._sess.run(tf.global_variables_initializer())
        for _ in range(10):
            epoch_err = 0
            for __ in range(28):
                x_batch, y_batch = self._generator.gen(64)
                feed_dict = {self._tfx: x_batch, self._tfy: y_batch}
                iter_err = self._sess.run([loss, train_step], feed_dict)[0]
                self._log["iter_err"].append(iter_err)
                epoch_err += iter_err
            self._log["epoch_err"].append(epoch_err / 28)
            self._verbose()

    def predict(self, x):
        x = np.atleast_3d(x)
        output = self._sess.run(self._output, {self._tfx: x})
        return np.argmax(output, axis=1).ravel()

    def draw_err_logs(self):
        ee, ie = self._log["epoch_err"], self._log["iter_err"]
        ee_base = np.arange(len(ee))
        ie_base = np.linspace(0, len(ee) - 1, len(ie))
        plt.figure()
        plt.plot(ie_base, ie, label="Iter error")
        plt.plot(ee_base, ee, linewidth=3, label="Epoch error")
        plt.legend()
        plt.show()
