import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn


class AdditionGenerator:
    def __init__(self, im, om, n_time_step):
        self._im, self._om = im, om
        self._base = self._om
        self._n_time_step = n_time_step

    def _op(self, seq):
        return sum(seq)

    def _gen_seq(self, n_time_step, tar):
        seq = []
        for _ in range(n_time_step):
            seq.append(tar % self._base)
            tar //= self._base
        return seq

    def _gen_targets(self, n_time_step):
        return [int(random.randint(0, self._om ** n_time_step - 1) / self._im) for _ in range(self._im)]

    def gen(self, batch_size):
        x = np.empty([batch_size, self._n_time_step, self._im])
        y = np.zeros([batch_size, self._n_time_step, self._om])
        for i in range(batch_size):
            targets = self._gen_targets(self._n_time_step)
            sequences = [self._gen_seq(self._n_time_step, tar) for tar in targets]
            for j in range(self._im):
                x[i, ..., j] = sequences[j]
            y[i, range(self._n_time_step), self._gen_seq(self._n_time_step, self._op(targets))] = 1
        return x, y


class RNNWrapper:
    def __init__(self):
        self._log = {}
        self._return_all_states = None
        self._optimizer = self._generator = None
        self._tfx = self._tfy = self._input = self._output = None
        self._cell = self._im = self._om = None
        self._sess = tf.Session()

    def _verbose(self):
        x_test, y_test = self._generator.gen(1)
        ans = np.argmax(self._sess.run(self._output, {
            self._tfx: x_test
        }), axis=2).ravel()
        x_test = x_test.astype(np.int)
        print("I think {} = {}, answer: {}...".format(
            " + ".join(
                ["".join(map(lambda n: str(n), x_test[0, ..., i][::-1])) for i in range(x_test.shape[2])]
            ),
            "".join(map(lambda n: str(n), ans[::-1])),
            "".join(map(lambda n: str(n), np.argmax(y_test, axis=2).ravel()[::-1]))))

    def _get_output(self, rnn_outputs, rnn_states):
        print("Outputs :", rnn_outputs)
        print("States  :", rnn_states)
        if self._return_all_states:
            outputs = tf.concat([rnn_outputs, rnn_states], axis=2)
        else:
            outputs = rnn_outputs
        self._output = layers.fully_connected(
            outputs, num_outputs=self._om, activation_fn=tf.nn.sigmoid)

    def fit(self, im, om, generator, cell=rnn.BasicLSTMCell, return_all_states=False):
        self._generator = generator
        self._im, self._om = im, om
        self._return_all_states = return_all_states
        self._optimizer = tf.train.AdamOptimizer(0.01)
        self._input = self._tfx = tf.placeholder(tf.float32, shape=[None, None, im])
        self._tfy = tf.placeholder(tf.float32, shape=[None, None, om])

        self._cell = cell(128)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            self._cell, self._input, return_all_states=self._return_all_states,
            initial_state=self._cell.zero_state(tf.shape(self._input)[0], tf.float32)
        )
        self._get_output(rnn_outputs, rnn_states)
        loss = -tf.reduce_mean(
            self._tfy * tf.log(self._output + 1e-8) + (1 - self._tfy) * tf.log(1 - self._output + 1e-8)
        )
        train_step = self._optimizer.minimize(loss)
        self._log["iter_err"] = []
        self._log["epoch_err"] = []
        self._sess.run(tf.global_variables_initializer())
        for _ in range(20):
            epoch_err = 0
            for __ in range(128):
                x_batch, y_batch = self._generator.gen(64)
                feed_dict = {self._tfx: x_batch, self._tfy: y_batch}
                iter_err = self._sess.run([loss, train_step], feed_dict)[0]
                self._log["iter_err"].append(iter_err)
                epoch_err += iter_err
            self._log["epoch_err"].append(epoch_err / 128)
            self._verbose()

    def predict(self, x):
        x = np.atleast_3d(x)
        output = self._sess.run(self._output, {self._tfx: x})
        return np.argmax(output, axis=2).ravel()

    def draw_err_logs(self):
        ee, ie = self._log["epoch_err"], self._log["iter_err"]
        ee_base = np.arange(len(ee))
        ie_base = np.linspace(0, len(ee) - 1, len(ie))
        plt.figure()
        plt.plot(ie_base, ie, label="Iter error")
        plt.plot(ee_base, ee, linewidth=3, label="Epoch error")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    random_scale = 2
    digit_len = 4
    digit_base = 10
    n_digit = 2
    _generator = AdditionGenerator(n_digit, digit_base, n_time_step=digit_len)

    # Return final state only
    # BasicRNNCell
    print("=" * 60 + "\n" + "Return final state only (BasicRNNCell)\n" + "-" * 60)
    lstm = RNNWrapper()
    lstm.fit(n_digit, digit_base, _generator, rnn.BasicRNNCell)
    lstm.draw_err_logs()

    # BasicLSTMCell
    print("=" * 60 + "\n" + "Return final state only (BasicLSTMCell)\n" + "-" * 60)
    tf.reset_default_graph()
    lstm = RNNWrapper()
    lstm.fit(n_digit, digit_base, _generator)
    lstm.draw_err_logs()

    # LSTMCell
    print("=" * 60 + "\n" + "Return final state only (LSTMCell)\n" + "-" * 60)
    tf.reset_default_graph()
    lstm = RNNWrapper()
    lstm.fit(n_digit, digit_base, _generator, rnn.LSTMCell)
    lstm.draw_err_logs()

    # GRUCell
    print("=" * 60 + "\n" + "Return final state only (GRUCell)\n" + "-" * 60)
    tf.reset_default_graph()
    lstm = RNNWrapper()
    lstm.fit(n_digit, digit_base, _generator, rnn.GRUCell)
    lstm.draw_err_logs()

    # Return all states
    # LSTMCell
    print("=" * 60 + "\n" + "Return all states generated (LSTMCell)\n" + "-" * 60)
    tf.reset_default_graph()
    lstm = RNNWrapper()
    lstm.fit(n_digit, digit_base, _generator, rnn.LSTMCell, return_all_states=True)
    lstm.draw_err_logs()

    # GRUCell
    print("=" * 60 + "\n" + "Return all states generated (GRUCell)\n" + "-" * 60)
    tf.reset_default_graph()
    lstm = RNNWrapper()
    lstm.fit(n_digit, digit_base, _generator, rnn.GRUCell, return_all_states=True)
    lstm.draw_err_logs()
