import time
import tflearn
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
        if batch < 0:
            if test:
                return self._x_test, self._y_test
            return self._x_train, self._y_train
        batch = np.random.choice(len(self._x_train), batch)
        return self._x_train[batch], self._y_train[batch]

if __name__ == '__main__':
    _n_history = 3

    print("=" * 60, "\n" + "Normal LSTM", "\n" + "-" * 60)
    _t = time.time()
    tf.reset_default_graph()
    rnn = RNNWrapper(n_history=_n_history, epoch=10, squeeze=True)
    rnn.fit(28, 10, MnistGenerator)
    print("Time Cost: {}".format(time.time() - _t))
    rnn.draw_err_logs()

    print("=" * 60, "\n" + "Sparse LSTM" + "\n" + "-" * 60)
    _t = time.time()
    tf.reset_default_graph()
    rnn = SparseRNN(n_history=_n_history, epoch=10, generator_params={"one_hot": False})
    rnn.fit(28, 10, MnistGenerator)
    print("Time Cost: {}".format(time.time() - _t))
    rnn.draw_err_logs()

    print("=" * 60, "\n" + "Tflearn", "\n" + "-" * 60)
    _t = time.time()
    tf.reset_default_graph()
    net = tflearn.input_data(shape=[None, 28, 28])
    net = tf.concat(tflearn.lstm(net, 128, return_seq=True)[-_n_history:], axis=1)
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name="output1")
    model = tflearn.DNN(net, tensorboard_verbose=0)
    generator = MnistGenerator()
    model.fit(*generator.gen(-1), n_epoch=10, validation_set=generator.gen(-1, True), show_metric=True)
    print("Time Cost: {}".format(time.time() - _t))
