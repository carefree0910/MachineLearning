import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

import time
import numpy as np
import tensorflow as tf

from h_RNN.RNN import RNNWrapper, Generator
from h_RNN.SpRNN import SparseRNN
from Util.Util import DataUtil


class MnistGenerator(Generator):
    def __init__(self, im=None, om=None, one_hot=True):
        super(MnistGenerator, self).__init__(im, om)
        self._x, self._y = DataUtil.get_dataset("mnist", "../_Data/mnist.txt", quantized=True, one_hot=one_hot)
        self._x = self._x.reshape(-1, 28, 28)
        self._x_train, self._x_test = self._x[:1800], self._x[1800:]
        self._y_train, self._y_test = self._y[:1800], self._y[1800:]

    def gen(self, batch, test=False, **kwargs):
        if batch == 0:
            if test:
                return self._x_test, self._y_test
            return self._x_train, self._y_train
        batch = np.random.choice(len(self._x_train), batch)
        return self._x_train[batch], self._y_train[batch]

if __name__ == '__main__':
    n_history = 3
    print("=" * 60, "\n" + "Normal LSTM", "\n" + "-" * 60)
    generator = MnistGenerator()
    t = time.time()
    tf.reset_default_graph()
    rnn = RNNWrapper()
    rnn.fit(28, 10, generator, n_history=n_history, epoch=10, squeeze=True)
    print("Time Cost: {}".format(time.time() - t))
    rnn.draw_err_logs()

    print("=" * 60, "\n" + "Sparse LSTM" + "\n" + "-" * 60)
    generator = MnistGenerator(one_hot=False)
    t = time.time()
    tf.reset_default_graph()
    rnn = SparseRNN()
    rnn.fit(28, 10, generator, n_history=n_history, epoch=10)
    print("Time Cost: {}".format(time.time() - t))
    rnn.draw_err_logs()
