import time
import tflearn
import numpy as np
import tensorflow as tf

from Zhihu.RNN.RNN import RNNWrapper, FastLSTMCell
from Util.Util import DataUtil


class MnistGenerator:
    def __init__(self, im=None, om=None):
        self._im, self._om = im, om
        self._x, self._y = DataUtil.get_dataset("mnist", "../../_Data/mnist.txt", quantized=True, one_hot=True)
        self._x = self._x.reshape(-1, 28, 28)
        self._x_train, self._x_test = self._x[:1800], self._x[1800:]
        self._y_train, self._y_test = self._y[:1800], self._y[1800:]

    def gen(self, batch, test=False):
        if batch == 0:
            if test:
                return self._x_test, self._y_test
            return self._x_train, self._y_train
        batch = np.random.choice(len(self._x_train), batch)
        return self._x_train[batch], self._y_train[batch]

if __name__ == '__main__':
    print("=" * 60, "\n" + "My LSTM", "\n" + "-" * 60)
    generator = MnistGenerator()
    t = time.time()
    tf.reset_default_graph()
    rnn = RNNWrapper()
    rnn.fit(28, 10, generator)
    print("Time Cost: {}".format(time.time() - t))
    rnn.draw_err_logs()

    print("=" * 60, "\n" + "My Fast LSTM", "\n" + "-" * 60)
    generator = MnistGenerator()
    t = time.time()
    tf.reset_default_graph()
    rnn = RNNWrapper()
    rnn.fit(28, 10, generator, cell=FastLSTMCell)
    print("Time Cost: {}".format(time.time() - t))
    rnn.draw_err_logs()

    print("=" * 60, "\n" + "Tflearn", "\n" + "-" * 60)
    generator = MnistGenerator()
    t = time.time()
    tf.reset_default_graph()
    net = tflearn.input_data(shape=[None, 28, 28])
    net = tf.concat(tflearn.lstm(net, 128, return_seq=True)[-3:], axis=1)
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', batch_size=64,
                             loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(*generator.gen(0), n_epoch=10, validation_set=generator.gen(0, True), show_metric=True)
    print("Time Cost: {}".format(time.time() - t))
