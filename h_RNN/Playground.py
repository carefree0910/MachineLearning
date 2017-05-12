import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from g_CNN.Optimizers import OptFactory
from Util.ProgressBar import ProgressBar


class RNN1:
    def __init__(self, u, v, w):
        self._u, self._v, self._w = np.asarray(u), np.asarray(v), np.asarray(w)
        self._states = None

    def activate(self, x):
        return x

    def transform(self, x):
        return x

    def run(self, x):
        output = []
        x = np.atleast_2d(x)
        self._states = np.zeros([len(x)+1, self._u.shape[0]])
        for t, xt in enumerate(x):
            self._states[t] = self.activate(
                self._u.dot(xt) + self._w.dot(self._states[t-1])
            )
            output.append(self.transform(
                self._v.dot(self._states[t]))
            )
        return np.array(output)


class RNN2(RNN1):
    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def transform(self, x):
        safe_exp = np.exp(x - np.max(x))
        return safe_exp / np.sum(safe_exp)

    def bptt(self, x, y):
        x, y, n = np.asarray(x), np.asarray(y), len(y)
        o = self.run(x)
        dis = o - y
        dv = dis.T.dot(self._states[:-1])
        du = np.zeros_like(self._u)
        dw = np.zeros_like(self._w)
        for t in range(n-1, -1, -1):
            ds = self._v.T.dot(dis[t])
            for bptt_step in range(t, max(-1, t-10), -1):
                du += np.outer(ds, x[bptt_step])
                dw += np.outer(ds, self._states[bptt_step-1])
                st = self._states[bptt_step-1]
                ds = self._w.T.dot(ds) * st * (1 - st)
        return du, dv, dw

    def loss(self, x, y):
        o = self.run(x)
        # noinspection PyTypeChecker
        return np.sum(
            -y * np.log(np.maximum(o, 1e-12)) - (1 - y) * np.log(np.maximum(1 - o, 1e-12))
        )


class LSTMCell(tf.contrib.rnn.BasicRNNCell):
    def __call__(self, x, state, scope="LSTM"):
        with tf.variable_scope(scope):
            s_old, h_old = tf.split(state, 2, 1)
            gates = layers.fully_connected(
                tf.concat([x, s_old], 1),
                num_outputs=4 * self._num_units,
                activation_fn=None)
            r1, g1, g2, g3 = tf.split(gates, 4, 1)
            r1, g1, g3 = tf.nn.sigmoid(r1), tf.nn.sigmoid(g1), tf.nn.sigmoid(g3)
            g2 = tf.nn.tanh(g2)
            h_new = h_old * r1 + g1 * g2
            s_new = tf.nn.tanh(h_new) * g3
            return s_new, tf.concat([s_new, h_new], 1)

    @property
    def state_size(self):
        return 2 * self._num_units


class LSTM:
    def __init__(self, **kwargs):
        self._log = {}
        self._optimizer = None
        self._generator = None
        self._tfx = self._tfy = self._output = None
        self._sess = tf.Session()

        self._params = {
            "generator_params": kwargs.get("generator_params", {}),
            "n_time_step": kwargs.get("n_time_step", 6),
            "random_scale": kwargs.get("random_scale", 2),
            "n_hidden": kwargs.get("n_hidden", 64),
            "activation": kwargs.get("activation", tf.nn.sigmoid),
            "lr": kwargs.get("lr", 0.01),
            "epoch": kwargs.get("epoch", 20),
            "n_iter": kwargs.get("n_iter", 128),
            "optimizer": kwargs.get("optimizer", "Adam"),
            "batch_size": kwargs.get("batch_size", 64),
            "eps": kwargs.get("eps", 1e-8),
            "verbose": kwargs.get("verbose", True)
        }

    def __getitem__(self, item):
        return getattr(self, str(item), None)

    def fit(self, im, om, generator, generator_params=None,
            n_time_step=None, random_scale=None, n_hidden=None, activation=None,
            lr=None, epoch=None, n_iter=None, batch_size=None, optimizer=None, eps=None, verbose=None):
        if generator_params is None:
            generator_params = self._params["generator_params"]
        if n_time_step is None:
            n_time_step = self._params["n_time_step"]
        if random_scale is None:
            random_scale = self._params["random_scale"]
        if n_hidden is None:
            n_hidden = self._params["n_hidden"]
        if activation is None:
            activation = self._params["activation"]
        if lr is None:
            lr = self._params["lr"]
        if epoch is None:
            epoch = self._params["epoch"]
        if n_iter is None:
            n_iter = self._params["n_iter"]
        if optimizer is None:
            optimizer = self._params["optimizer"]
        if batch_size is None:
            batch_size = self._params["batch_size"]
        if eps is None:
            eps = self._params["eps"]
        if verbose is None:
            verbose = self._params["verbose"]

        self._generator = generator(n_time_step, random_scale, im, **generator_params)
        self._optimizer = OptFactory().get_optimizer_by_name(optimizer, lr)
        self._tfx = tf.placeholder(tf.float32, shape=[None, None, im])
        self._tfy = tf.placeholder(tf.float32, shape=[None, None, om])

        cell = LSTMCell(n_hidden)
        initial_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell, self._tfx, initial_state=initial_state, time_major=True)
        self._output = tf.map_fn(
            lambda _x: layers.fully_connected(_x, num_outputs=om, activation_fn=activation),
            rnn_outputs
        )
        err = -tf.reduce_mean(
            self._tfy * tf.log(self._output + eps) + (1 - self._tfy) * tf.log(1 - self._output + eps)
        )
        train_step = self._optimizer.minimize(err)
        self._log["iter_err"] = []
        self._log["epoch_err"] = []

        with self._sess.as_default() as sess:
            sess.run(tf.global_variables_initializer())
            bar = ProgressBar(max_value=epoch, name="Epoch")
            for _ in range(epoch):
                epoch_err = 0
                sub_bar = ProgressBar(max_value=n_iter, name="Iter")
                for __ in range(n_iter):
                    x, y = self._generator.gen(batch_size)
                    iter_err = sess.run([err, train_step], {
                        self._tfx: x, self._tfy: y,
                    })[0]
                    self._log["iter_err"].append(iter_err)
                    epoch_err += iter_err
                    sub_bar.update()
                self._log["epoch_err"].append(epoch_err / n_iter)
                if verbose:
                    self._verbose(sess)
                bar.update()

    def _verbose(self, sess):
        pass


class LSTMForAddition(LSTM):
    def _verbose(self, sess):
        x_test, y_test = self._generator.gen(1)
        ans = np.argmax(sess.run(self._output, {
            self._tfx: x_test
        }), axis=2).ravel()
        x_test = x_test.astype(np.int)
        print("I think {} = {}, answer: {}...".format(
            " + ".join(
                ["".join(map(lambda n: str(n), x_test[..., 0, i])) for i in range(x_test.shape[2])]
            ),
            "".join(map(lambda n: str(n), ans)),
            "".join(map(lambda n: str(n), np.argmax(y_test, axis=2).ravel()))))


class AddGenerator:
    def __init__(self, n_time_step, random_scale, im, base=10):
        self._n_time_step = n_time_step
        self._random_scale = random_scale
        self._im, self._base = im, base

    def _gen_seq(self, n_time_step, tar):
        seq = []
        for _ in range(n_time_step):
            seq.append(tar % self._base)
            tar //= self._base
        return seq

    def gen(self, batch_size):
        n_time_step = self._n_time_step + random.randint(0, self._random_scale)
        x = np.empty([n_time_step, batch_size, self._im])
        y = np.zeros([n_time_step, batch_size, self._im])
        for i in range(batch_size):
            targets = [int(random.randint(0, self._base ** self._n_time_step - 1) / self._im) for _ in range(self._im)]
            sequences = [self._gen_seq(n_time_step, tar) for tar in targets]
            for j in range(self._im):
                x[:, i, j] = sequences[j]
            y[range(n_time_step), i, self._gen_seq(n_time_step, sum(targets))] = 1
        return x, y

if __name__ == '__main__':
    # n_sample = 5
    # rnn = RNN1(np.eye(n_sample), np.eye(n_sample), np.eye(n_sample) * 2)
    # print(rnn.run(np.eye(n_sample)))
    length, _base = 2, 2
    lstm = LSTMForAddition(generator_params={"base": _base}, n_time_step=2)
    lstm.fit(length, _base, generator=AddGenerator)
