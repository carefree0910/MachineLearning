import random
import tensorflow as tf

from RNN.Generator import *
from RNN.Wrapper import RNNWrapper


# RNN For Op
class RNNForOp(RNNWrapper):
    def __init__(self, **kwargs):
        super(RNNForOp, self).__init__(**kwargs)
        self._params["boost"] = kwargs.get("boost", 2)
        self._op = ""

    def _verbose(self):
        x_test, y_test = self._generator.gen(1, boost=self._params["boost"])
        ans = np.argmax(self._sess.run(self._output, {
            self._tfx: x_test
        }), axis=2).ravel()
        x_test = x_test.astype(np.int)
        if self._use_sparse_labels:
            y_test = y_test.ravel()
        else:
            y_test = np.argmax(y_test, axis=2).ravel()
        print("I think {} = {}, answer: {}...".format(
            " {} ".format(self._op).join(
                ["".join(map(lambda n: str(n), x_test[0, ..., i][::-1])) for i in range(x_test.shape[2])]
            ),
            "".join(map(lambda n: str(n), ans[::-1])),
            "".join(map(lambda n: str(n), y_test[::-1]))))


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
        kwargs["use_sparse_labels"] = kwargs["squeeze"] = True
        super(SpRNNForOp, self).__init__(**kwargs)
        self._params["boost"] = 0
        self._op = ""

    def _verbose(self):
        x_test, y_test = self._generator.gen(1, boost=self._params["boost"])
        if self._squeeze:
            y_test = [y_test]
        if self._squeeze:
            ans = [np.argmax(self._sess.run(self._output, {self._tfx: x_test}), axis=1)]
        else:
            ans = np.argmax(self._sess.run(self._output, {self._tfx: x_test}), axis=2)
        x_test = x_test.astype(np.int)
        print("I think {} = {}, answer: {}...".format(
            " {} ".format(self._op).join(
                ["".join(map(lambda n: str(n), x_test[0, ..., i][::-1])) for i in range(x_test.shape[2])]
            ),
            "".join(map(lambda n: str(n), ans[0][::-1])),
            "".join(map(lambda n: str(n), y_test[0][::-1]))))


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
        kwargs["use_sparse_labels"] = kwargs["squeeze"] = True
        super(EmbedRNNForOp, self).__init__(**kwargs)
        self._params["boost"] = 0
        self._op = ""

    def _verbose(self):
        x_test, y_test = self._generator.gen(1, boost=self._params["boost"])
        ans = np.argmax(self._sess.run(self._output, {self._tfx: x_test}), axis=1)
        x_test = x_test.astype(np.int)
        print("I think {} = {}, answer: {}...".format(
            " {} ".format(self._op).join(map(lambda n: str(n), x_test[0])),
            ans[0], y_test[0]))


class EmbedRNNForAddition(EmbedRNNForOp):
    def __init__(self, **kwargs):
        super(EmbedRNNForAddition, self).__init__(**kwargs)
        self._op = "+"


class EmbedRNNForMultiple(EmbedRNNForOp):
    def __init__(self, **kwargs):
        super(EmbedRNNForMultiple, self).__init__(**kwargs)
        self._op = "*"


# Generators

# Op Generator
class OpGenerator(Generator):
    def __init__(self, im, om, n_time_step, random_scale, use_sparse_labels=False):
        super(OpGenerator, self).__init__(im, om)
        self._base = self._om
        self._n_time_step = n_time_step
        self._random_scale = random_scale
        self._use_sparse_labels = use_sparse_labels

    def _op(self, seq):
        return 0

    def _gen_seq(self, n_time_step, tar):
        seq = []
        for _ in range(n_time_step):
            seq.append(tar % self._base)
            tar //= self._base
        return seq

    def _gen_targets(self, n_time_step):
        return []

    def gen(self, batch_size, test=False, boost=0):
        if boost:
            n_time_step = self._n_time_step + self._random_scale + random.randint(1, boost)
        else:
            n_time_step = self._n_time_step + random.randint(0, self._random_scale)
        x = np.empty([batch_size, n_time_step, self._im])
        if not self._use_sparse_labels:
            y = np.zeros([batch_size, n_time_step, self._om])
        else:
            y = np.zeros([batch_size, n_time_step], dtype=np.int32)
        for i in range(batch_size):
            targets = self._gen_targets(n_time_step)
            sequences = [self._gen_seq(n_time_step, tar) for tar in targets]
            for j in range(self._im):
                x[i, ..., j] = sequences[j]
            ans_seq = self._gen_seq(n_time_step, self._op(targets))
            if not self._use_sparse_labels:
                y[i, range(n_time_step), ans_seq] = 1
            else:
                y[i] = ans_seq
        return x, y


class AdditionGenerator(OpGenerator):
    def _op(self, seq):
        return sum(seq)

    def _gen_targets(self, n_time_step):
        return [int(random.randint(0, self._om ** n_time_step - 1) / self._im) for _ in range(self._im)]


class MultipleGenerator(OpGenerator):
    def _op(self, seq):
        return np.prod(seq)

    def _gen_targets(self, n_time_step):
        return [int(random.randint(0, self._om ** n_time_step - 1) ** (1 / self._im)) for _ in range(self._im)]


# Sparse Op Generator
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


# Embedding Sparse Op Generator
class EmbedOpGenerator(Generator):
    def __init__(self, im, om, n_digit, random_scale=0):
        super(EmbedOpGenerator, self).__init__(im, om)
        self._n_digit = n_digit
        self._random_scale = random_scale

    def _op(self, x):
        return 0

    def _get_x(self, n_digit):
        return 0

    def gen(self, batch_size, test=False, boost=0):
        n_digit = self._n_digit + random.randint(0, self._random_scale)
        x = np.empty([batch_size, n_digit], dtype=np.int32)
        y = np.zeros(batch_size, dtype=np.int32)
        for i in range(batch_size):
            x[i] = self._get_x(n_digit)
            y[i] = self._op(x[i])
        return x, y


class EmbedAdditionGenerator(EmbedOpGenerator):
    def _op(self, seq):
        return sum(seq)

    def _get_x(self, n_digit):
        return np.random.randint(0, int(min(self._im, self._om / n_digit)), n_digit)


class EmbedMultipleGenerator(EmbedOpGenerator):
    def _op(self, seq):
        return np.prod(seq)

    def _get_x(self, n_digit):
        return np.random.randint(0, int(min(self._im, self._om ** (1 / n_digit))), n_digit)


# Test Cases
def test_rnn(random_scale=2, digit_len=2, digit_base=10, n_digit=3, use_sparse_labels=False):
    tf.reset_default_graph()
    generator = AdditionGenerator(
        n_digit, digit_base, n_time_step=digit_len, random_scale=random_scale,
        use_sparse_labels=use_sparse_labels
    )
    lstm = RNNForAddition(
        # cell=tf.contrib.rnn.GRUCell,
        # cell=tf.contrib.rnn.LSTMCell,
        # cell=tf.contrib.rnn.BasicRNNCell,
        # cell=tf.contrib.rnn.BasicLSTMCell,
        epoch=100, boost=2, use_sparse_labels=use_sparse_labels
    )
    lstm.fit(n_digit, digit_base, generator)
    lstm.draw_err_logs()


def test_sp_rnn(random_scale=0, digit_len=4, digit_base=10, n_digit=2):
    tf.reset_default_graph()
    generator = SpMultipleGenerator(
        n_digit, digit_base ** (digit_len + random_scale),
        n_time_step=digit_len, random_scale=random_scale
    )
    lstm = SpRNNForMultiple(
        # cell=tf.contrib.rnn.GRUCell,
        # cell=tf.contrib.rnn.LSTMCell,
        # cell=tf.contrib.rnn.BasicRNNCell,
        # cell=tf.contrib.rnn.BasicLSTMCell,
        epoch=20, n_history=2
    )
    lstm.fit(n_digit, digit_base ** (digit_len + random_scale), generator)
    lstm.draw_err_logs()


def test_embed_rnn(random_scale=0, n_digit=2, im=100, om=10000):
    tf.reset_default_graph()
    generator = EmbedMultipleGenerator(im, om, n_digit=n_digit, random_scale=random_scale)
    lstm = EmbedRNNForMultiple(
        # cell=tf.contrib.rnn.GRUCell,
        # cell=tf.contrib.rnn.LSTMCell,
        # cell=tf.contrib.rnn.BasicRNNCell,
        # cell=tf.contrib.rnn.BasicLSTMCell,
        epoch=20, use_sparse_labels=True, embedding_size=50,
        use_final_state=False, n_history=n_digit
    )
    lstm.fit(im, om, generator)
    lstm.draw_err_logs()

if __name__ == '__main__':
    test_rnn()
    test_rnn(use_sparse_labels=True)
    test_sp_rnn()
    test_embed_rnn()
    test_embed_rnn(1)
