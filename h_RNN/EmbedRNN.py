import numpy as np

from h_RNN.RNN import RNNWrapper, Generator


class EmbedRNNForOp(RNNWrapper):
    def __init__(self):
        super(EmbedRNNForOp, self).__init__()
        if not self._generator_params:
            self._generator_params = {"n_digit": 2}
        self._op = ""

    def _verbose(self):
        x_test, y_test = self._generator.gen(1)
        ans = np.argmax(self._sess.run(self._output, {self._tfx: x_test}), axis=1)
        x_test = x_test.astype(np.int)
        print("I think {} = {}, answer: {}...".format(
            " {} ".format(self._op).join(map(lambda n: str(n), x_test[0])),
            "".join(map(lambda n: str(n), ans)),
            "".join(map(lambda n: str(n), y_test))))


class EmbedRNNForAddition(EmbedRNNForOp):
    def __init__(self):
        super(EmbedRNNForAddition, self).__init__()
        self._op = "+"


class EmbedRNNForMultiple(EmbedRNNForOp):
    def __init__(self):
        super(EmbedRNNForMultiple, self).__init__()
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
    lstm = EmbedRNNForMultiple()
    lstm.fit(_im, _om, EmbedMultipleGenerator,
             epoch=20, sparse=True, embedding_size=50, use_final_state=False,
             n_history=_n_digit, generator_params={"n_digit": _n_digit})
    lstm.draw_err_logs()
