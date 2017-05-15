import tensorflow as tf

from RNN.Generators import *
from RNN.Wrapper import RNNWrapper


# RNN For Op
class RNNForOp(RNNWrapper):
    def __init__(self, **kwargs):
        super(RNNForOp, self).__init__(**kwargs)
        self._params["boost"] = kwargs.get("boost", 2)
        if not self._params["generator_params"]:
            self._params["generator_params"] = {
                "n_time_step": 6, "random_scale": 2
            }
        self._op = ""

    def _verbose(self):
        x_test, y_test = self._generator.gen(1, boost=self._params["boost"])
        ans = np.argmax(self._sess.run(self._output, {
            self._tfx: x_test
        }), axis=2).ravel()
        x_test = x_test.astype(np.int)
        print("I think {} = {}, answer: {}...".format(
            " {} ".format(self._op).join(
                ["".join(map(lambda n: str(n), x_test[0, ..., i][::-1])) for i in range(x_test.shape[2])]
            ),
            "".join(map(lambda n: str(n), ans[::-1])),
            "".join(map(lambda n: str(n), np.argmax(y_test, axis=2).ravel()[::-1]))))


class RNNForAddition(RNNForOp):
    def __init__(self, **kwargs):
        super(RNNForAddition, self).__init__(**kwargs)
        self._op = "+"


class RNNForMultiple(RNNForOp):
    def __init__(self, **kwargs):
        super(RNNForMultiple, self).__init__(**kwargs)
        self._op = "*"


# Sparse RNN For Op
class SpRNNForOp(RNNWrapper):
    def __init__(self, **kwargs):
        kwargs["sparse"] = True
        super(SpRNNForOp, self).__init__(**kwargs)
        if not self._params["generator_params"]:
            self._params["generator_params"] = {
                "n_time_step": 6, "random_scale": 2
            }
        self._params["boost"] = 0
        self._op = ""

    def _verbose(self):
        x_test, y_test = self._generator.gen(1, boost=self._params["boost"])
        ans = np.argmax(self._sess.run(self._output, {self._tfx: x_test}), axis=1)
        x_test = x_test.astype(np.int)
        print("I think {} = {}, answer: {}...".format(
            " {} ".format(self._op).join(
                ["".join(map(lambda n: str(n), x_test[0, ..., i][::-1])) for i in range(x_test.shape[2])]
            ),
            "".join(map(lambda n: str(n), ans)),
            "".join(map(lambda n: str(n), y_test))))


class SpRNNForAddition(SpRNNForOp):
    def __init__(self, **kwargs):
        super(SpRNNForAddition, self).__init__(**kwargs)
        self._op = "+"


class SpRNNForMultiple(SpRNNForOp):
    def __init__(self, **kwargs):
        super(SpRNNForMultiple, self).__init__(**kwargs)
        self._op = "*"


# Embedding RNN For Op
class EmbedRNNForOp(RNNWrapper):
    def __init__(self, **kwargs):
        super(EmbedRNNForOp, self).__init__(**kwargs)
        if not self._params["generator_params"]:
            self._params["generator_params"] = {"n_digit": 2}
        self._params["boost"] = 0
        self._op = ""

    def _verbose(self):
        x_test, y_test = self._generator.gen(1, boost=self._params["boost"])
        ans = np.argmax(self._sess.run(self._output, {self._tfx: x_test}), axis=1)
        x_test = x_test.astype(np.int)
        print("I think {} = {}, answer: {}...".format(
            " {} ".format(self._op).join(map(lambda n: str(n), x_test[0])),
            "".join(map(lambda n: str(n), ans)),
            "".join(map(lambda n: str(n), y_test))))


class EmbedRNNForAddition(EmbedRNNForOp):
    def __init__(self, **kwargs):
        super(EmbedRNNForAddition, self).__init__(**kwargs)
        self._op = "+"


class EmbedRNNForMultiple(EmbedRNNForOp):
    def __init__(self, **kwargs):
        super(EmbedRNNForMultiple, self).__init__(**kwargs)
        self._op = "*"


# Test Cases
def test_rnn(random_scale=2, digit_len=2, digit_base=10, n_digit=3):
    tf.reset_default_graph()
    lstm = RNNForAddition(
        # cell=tf.contrib.rnn.GRUCell,
        # cell=tf.contrib.rnn.LSTMCell,
        # cell=tf.contrib.rnn.BasicRNNCell,
        # cell=tf.contrib.rnn.BasicLSTMCell,
        generator_params={"n_time_step": digit_len, "random_scale": random_scale}, epoch=100, boost=2
    )
    lstm.fit(n_digit, digit_base, AdditionGenerator)
    lstm.draw_err_logs()


def test_sp_rnn(random_scale=0, digit_len=4, digit_base=10, n_digit=2):
    tf.reset_default_graph()
    lstm = SpRNNForMultiple(
        # cell=tf.contrib.rnn.GRUCell,
        # cell=tf.contrib.rnn.LSTMCell,
        # cell=tf.contrib.rnn.BasicRNNCell,
        # cell=tf.contrib.rnn.BasicLSTMCell,
        epoch=20, n_history=2,
        generator_params={"n_time_step": digit_len, "random_scale": random_scale}
    )
    lstm.fit(n_digit, digit_base ** (digit_len + random_scale), SpMultipleGenerator)
    lstm.draw_err_logs()


def test_embed_rnn(n_digit=2, im=100, om=10000):
    tf.reset_default_graph()
    lstm = EmbedRNNForMultiple(
        # cell=tf.contrib.rnn.GRUCell,
        # cell=tf.contrib.rnn.LSTMCell,
        # cell=tf.contrib.rnn.BasicRNNCell,
        # cell=tf.contrib.rnn.BasicLSTMCell,
        epoch=20, sparse=True, embedding_size=50, use_final_state=False,
        n_history=n_digit, generator_params={"n_digit": n_digit}
    )
    lstm.fit(im, om, EmbedMultipleGenerator)
    lstm.draw_err_logs()

if __name__ == '__main__':
    test_rnn()
    test_sp_rnn()
    test_embed_rnn()
