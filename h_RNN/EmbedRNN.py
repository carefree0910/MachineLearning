import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import tensorflow as tf

from h_RNN.RNN import Generator
from h_RNN.SpRNN import SparseRNN


class EmbedRNN(SparseRNN):
    def __init__(self, embedding_size=200):
        super(EmbedRNN, self).__init__()
        self._embedding_size = embedding_size

    def _define_input(self, im, om):
        self._tfx = tf.placeholder(tf.int32, shape=[None, None])
        embeddings = tf.Variable(tf.random_uniform([im, self._embedding_size], -1.0, 1.0))
        self._input = tf.nn.embedding_lookup(embeddings, self._tfx)
        self._tfy = tf.placeholder(tf.int32, shape=[None])


class EmbedRNNForOp(EmbedRNN):
    def __init__(self, embedding_size=200):
        super(EmbedRNNForOp, self).__init__(embedding_size)
        self._op = ""

    def _verbose(self):
        x_test, y_test = self._generator.gen(1)
        ans = self.predict(x_test)
        x_test = x_test.astype(np.int)
        print("I think {} = {}, answer: {}...".format(
            " {} ".format(self._op).join(map(lambda n: str(n), x_test[0])),
            "".join(map(lambda n: str(n), ans)),
            "".join(map(lambda n: str(n), y_test))))


class EmbedRNNForAddition(EmbedRNNForOp):
    def __init__(self, embedding_size=200):
        super(EmbedRNNForAddition, self).__init__(embedding_size)
        self._op = "+"


class EmbedRNNForMultiple(EmbedRNNForOp):
    def __init__(self, embedding_size=200):
        super(EmbedRNNForMultiple, self).__init__(embedding_size)
        self._op = "*"


class EmbedOpGenerator(Generator):
    def __init__(self, im, om, n_digit):
        super(EmbedOpGenerator, self).__init__(im, om)
        self._n_digit = n_digit

    def _op(self, x):
        return 0

    def _get_x(self):
        return 0

    def gen(self, batch_size, test=False, boost=0):
        x = np.empty([batch_size, self._n_digit], dtype=np.int32)
        y = np.zeros(batch_size, dtype=np.int32)
        for i in range(batch_size):
            x[i] = self._get_x()
            y[i] = self._op(x[i])
        return x, y


class EmbedAdditionGenerator(EmbedOpGenerator):
    def _op(self, seq):
        return sum(seq)

    def _get_x(self):
        return np.random.randint(0, int(min(self._im, self._om / self._n_digit)), self._n_digit)


class EmbedMultipleGenerator(EmbedOpGenerator):
    def _op(self, seq):
        return np.prod(seq)

    def _get_x(self):
        return np.random.randint(0, int(min(self._im, self._om ** (1 / self._n_digit))), self._n_digit)

if __name__ == '__main__':
    _n_digit = 2
    _im, _om = 100, 10000
    _generator = EmbedMultipleGenerator(_im, _om, n_digit=_n_digit)
    lstm = EmbedRNNForMultiple(embedding_size=50)
    lstm.fit(
        _im, _om, _generator,
        # cell=tf.contrib.rnn.GRUCell,
        # cell=tf.contrib.rnn.LSTMCell,
        # cell=tf.contrib.rnn.BasicRNNCell,
        # cell=tf.contrib.rnn.BasicLSTMCell,
        epoch=20, n_history=_n_digit
    )
    lstm.draw_err_logs()
