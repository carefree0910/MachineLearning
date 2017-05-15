import time
import tflearn
import tensorflow as tf

from RNN.Wrapper import RNNWrapper
from RNN.Generators import MnistGenerator


def test_mnist(n_history=3):
    print("=" * 60, "\n" + "Normal LSTM", "\n" + "-" * 60)
    t = time.time()
    tf.reset_default_graph()
    rnn = RNNWrapper(n_history=n_history, epoch=10, squeeze=True)
    rnn.fit(28, 10, MnistGenerator)
    print("Time Cost: {}".format(time.time() - t))
    rnn.draw_err_logs()

    print("=" * 60, "\n" + "Sparse LSTM" + "\n" + "-" * 60)
    t = time.time()
    tf.reset_default_graph()
    rnn = RNNWrapper(n_history=n_history, epoch=10, sparse=True, generator_params={"one_hot": False})
    rnn.fit(28, 10, MnistGenerator)
    print("Time Cost: {}".format(time.time() - t))
    rnn.draw_err_logs()

    print("=" * 60, "\n" + "Tflearn", "\n" + "-" * 60)
    t = time.time()
    tf.reset_default_graph()
    net = tflearn.input_data(shape=[None, 28, 28])
    net = tf.concat(tflearn.lstm(net, 128, return_seq=True)[-n_history:], axis=1)
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name="output1")
    model = tflearn.DNN(net, tensorboard_verbose=0)
    generator = MnistGenerator()
    model.fit(*generator.gen(1), n_epoch=10, validation_set=generator.gen(1, True), show_metric=True)
    print("Time Cost: {}".format(time.time() - t))

if __name__ == '__main__':
    test_mnist()
