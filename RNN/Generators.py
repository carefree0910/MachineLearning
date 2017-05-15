import random
import numpy as np

from Util.Util import DataUtil


# Generator Framework
class Generator:
    def __init__(self, im=None, om=None, **kwargs):
        self._im, self._om = im, om

    def gen(self, batch, test=False, **kwargs):
        pass


# Mnist Generator
class MnistGenerator(Generator):
    def __init__(self, im=None, om=None, one_hot=True):
        super(MnistGenerator, self).__init__(im, om)
        self._x, self._y = DataUtil.get_dataset("mnist", "../_Data/mnist.txt", quantized=True, one_hot=one_hot)
        self._x = self._x.reshape(-1, 28, 28)
        self._x_train, self._x_test = self._x[:1800], self._x[1800:]
        self._y_train, self._y_test = self._y[:1800], self._y[1800:]

    def gen(self, batch, test=False, **kwargs):
        if batch == 1:
            if test:
                return self._x_test, self._y_test
            return self._x_train, self._y_train
        batch = np.random.choice(len(self._x_train), batch)
        return self._x_train[batch], self._y_train[batch]


# Op Generator
class OpGenerator(Generator):
    def __init__(self, im, om, n_time_step, random_scale):
        super(OpGenerator, self).__init__(im, om)
        self._base = self._om
        self._n_time_step = n_time_step
        self._random_scale = random_scale

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
        y = np.zeros([batch_size, n_time_step, self._om])
        for i in range(batch_size):
            targets = self._gen_targets(n_time_step)
            sequences = [self._gen_seq(n_time_step, tar) for tar in targets]
            for j in range(self._im):
                x[i, ..., j] = sequences[j]
            y[i, range(n_time_step), self._gen_seq(n_time_step, self._op(targets))] = 1
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
