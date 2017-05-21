import time
import tflearn
import tensorflow as tf

from RNN.Wrapper import RNNWrapper
from RNN.Generator import Generator
from Util.Util import DataUtil


# Generator
class MnistGenerator(Generator):
    def __init__(self, im=None, om=None, one_hot=True):
        super(MnistGenerator, self).__init__(im, om)
        self._x, self._y = DataUtil.get_dataset("mnist", "../../_Data/mnist.txt", quantized=True, one_hot=one_hot)
        self._x = self._x.reshape(-1, 28, 28)
        self._x_train, self._x_test = self._x[:1800], self._x[1800:]
        self._y_train, self._y_test = self._y[:1800], self._y[1800:]


# Test Case
def test_mnist(n_history=3, draw=False):
    print("=" * 60, "\n" + "Normal LSTM", "\n" + "-" * 60)
    generator = MnistGenerator()
    t = time.time()
    tf.reset_default_graph()
    rnn = RNNWrapper(n_history=n_history, epoch=10, squeeze=True)
    rnn.fit(28, 10, generator, n_iter=28)
    print("Time Cost: {}".format(time.time() - t))
    if draw:
        rnn.draw_err_logs()

    print("=" * 60, "\n" + "Sparse LSTM" + "\n" + "-" * 60)
    generator = MnistGenerator(one_hot=False)
    t = time.time()
    tf.reset_default_graph()
    rnn = RNNWrapper(n_history=n_history, epoch=10, squeeze=True, use_sparse_labels=True)
    rnn.fit(28, 10, generator, n_iter=28)
    print("Time Cost: {}".format(time.time() - t))
    if draw:
        rnn.draw_err_logs()

    print("=" * 60, "\n" + "Tflearn", "\n" + "-" * 60)
    generator = MnistGenerator()
    t = time.time()
    tf.reset_default_graph()
    net = tflearn.input_data(shape=[None, 28, 28])
    net = tf.concat(tflearn.lstm(net, 128, return_seq=True)[-n_history:], axis=1)
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', batch_size=64,
                             loss='categorical_crossentropy', name="output1")
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(*generator.gen(0), n_epoch=10, validation_set=generator.gen(0, True), show_metric=True)
    print("Time Cost: {}".format(time.time() - t))

if __name__ == '__main__':
    test_mnist(draw=True)
