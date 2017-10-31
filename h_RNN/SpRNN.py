import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

import random
import numpy as np
import tensorflow as tf

from h_RNN.RNN import RNNWrapper, OpGenerator


class SparseRNN(RNNWrapper):
    def __init__(self):
        super(SparseRNN, self).__init__()
        self._squeeze = True
        self._activation = None

    def _define_input(self, im, om):
        self._input = self._tfx = tf.placeholder(tf.float32, shape=[None, None, im])
        self._tfy = tf.placeholder(tf.int32, shape=[None])

    def _get_loss(self, eps):
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._tfy, logits=self._output)
        )


class SpRNNForOp(SparseRNN):
    def __init__(self):
        super(SpRNNForOp, self).__init__()
        self._op = ""

    def _verbose(self):
        x_test, y_test = self._generator.gen(1)
        ans = self.predict(x_test)
        x_test = x_test.astype(np.int)
        print("I think {} = {}, answer: {}...".format(
            " {} ".format(self._op).join(
                ["".join(map(lambda n: str(n), x_test[0, ..., i][::-1])) for i in range(x_test.shape[2])]
            ),
            "".join(map(lambda n: str(n), ans)),
            "".join(map(lambda n: str(n), y_test))))


class SpRNNForAddition(SpRNNForOp):
    def __init__(self):
        super(SpRNNForAddition, self).__init__()
        self._op = "+"


class SpRNNForMultiple(SpRNNForOp):
    def __init__(self):
        super(SpRNNForMultiple, self).__init__()
        self._op = "*"


class SpOpGenerator(OpGenerator):
    def __init__(self, im, om, n_time_step, random_scale):
        super(SpOpGenerator, self).__init__(im, om, n_time_step, random_scale)
        self._base = round(self._om ** (1 / (n_time_step + random_scale)))

    def gen(self, batch_size, test=False, boost=0):
        if boost:
            n_time_step = self._n_time_step + self._random_scale + random.randint(1, boost)
        else:
            n_time_step = self._n_time_step + random.randint(0, self._random_scale)
        x = np.empty([batch_size, n_time_step, self._im])
        y = np.zeros(batch_size, dtype=np.int32)
        for i in range(batch_size):
            targets = self._gen_targets(n_time_step)
            sequences = [self._gen_seq(n_time_step, tar) for tar in targets]
            for j in range(self._im):
                x[i, ..., j] = sequences[j]
            y[i] = self._op(targets)
        return x, y


class SpAdditionGenerator(SpOpGenerator):
    def _op(self, seq):
        return sum(seq)

    def _gen_targets(self, n_time_step):
        return [int(random.randint(0, self._base ** n_time_step - 1) / self._im) for _ in range(self._im)]


class SpMultipleGenerator(SpOpGenerator):
    def _op(self, seq):
        return np.prod(seq)

    def _gen_targets(self, n_time_step):
        return [int(random.randint(0, self._base ** n_time_step - 1) ** (1 / self._im)) for _ in range(self._im)]

if __name__ == '__main__':
    _random_scale = 0
    _digit_len, _digit_base, _n_digit = 4, 10, 2
    _generator = SpAdditionGenerator(
        _n_digit, _digit_base ** (_digit_len + _random_scale),
        n_time_step=_digit_len, random_scale=_random_scale
    )
    lstm = SpRNNForAddition()
    lstm.fit(
        _n_digit, _digit_base ** (_digit_len + _random_scale), _generator,
        # cell=tf.contrib.rnn.GRUCell,
        # cell=tf.contrib.rnn.LSTMCell,
        # cell=tf.contrib.rnn.BasicRNNCell,
        # cell=tf.contrib.rnn.BasicLSTMCell,
        epoch=100, n_history=2
    )
    lstm.draw_err_logs()
