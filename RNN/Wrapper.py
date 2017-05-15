import numpy as np
import matplotlib.pyplot as plt

from RNN.Cell import *
from g_CNN.Optimizers import OptFactory
from Util.ProgressBar import ProgressBar


class RNNWrapper:
    def __init__(self, **kwargs):
        self._log = {}
        self._optimizer = self._generator = None
        self._tfx = self._tfy = self._input = self._output = None
        self._im = self._om = self._activation = None
        self._sess = tf.Session()

        self._squeeze = kwargs.get("squeeze", False)
        self._sparse = kwargs.get("sparse", False)
        self._embedding_size = kwargs.get("embedding_size", None)
        self._use_final_state = kwargs.get("use_final_state", False)
        self._params = {
            "generator_params": kwargs.get("generator_params", {}),
            "cell": kwargs.get("cell", LSTMCell),
            "n_hidden": kwargs.get("n_hidden", 128),
            "n_history": kwargs.get("n_history", 0),
            "activation": kwargs.get("activation", tf.nn.sigmoid),
            "lr": kwargs.get("lr", 0.01),
            "epoch": kwargs.get("epoch", 25),
            "n_iter": kwargs.get("n_iter", 128),
            "optimizer": kwargs.get("optimizer", "Adam"),
            "batch_size": kwargs.get("batch_size", 64),
            "eps": kwargs.get("eps", 1e-8),
            "verbose": kwargs.get("verbose", 1)
        }

        if self._sparse:
            self._squeeze = True
            self._params["n_history"] = kwargs.get("n_history", 1)
            self._params["activation"] = kwargs.get("activation", None)

    def _verbose(self):
        x_test, y_test = self._generator.gen(1, True)
        axis = 1 if self._squeeze else 2
        if self._sparse:
            y_true = y_test
        else:
            y_true = np.argmax(y_test, axis=axis).ravel()  # type: np.ndarray
        y_pred = np.argmax(self._sess.run(self._output, {self._tfx: x_test}), axis=axis).ravel()  # type: np.ndarray
        print("Test acc: {:8.6} %".format(np.mean(y_true == y_pred) * 100))

    def _define_input(self, im, om):
        if self._embedding_size:
            self._tfx = tf.placeholder(tf.int32, shape=[None, None])
            embeddings = tf.Variable(tf.random_uniform([im, self._embedding_size], -1.0, 1.0))
            self._input = tf.nn.embedding_lookup(embeddings, self._tfx)
        else:
            self._input = self._tfx = tf.placeholder(tf.float32, shape=[None, None, im])
        if self._sparse:
            self._tfy = tf.placeholder(tf.int32, shape=[None])
        elif self._squeeze:
            self._tfy = tf.placeholder(tf.float32, shape=[None, om])
        else:
            self._tfy = tf.placeholder(tf.float32, shape=[None, None, om])

    def _get_loss(self, eps):
        if self._sparse:
            return tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._tfy, logits=self._output)
            )
        return -tf.reduce_mean(
            self._tfy * tf.log(self._output + eps) + (1 - self._tfy) * tf.log(1 - self._output + eps)
        )

    def _get_output(self, rnn_outputs, rnn_final_state, n_history):
        if n_history == 0 and self._squeeze:
            raise ValueError("'n_history' should not be 0 when trying to squeeze the outputs")
        if self._sparse and not self._squeeze:
            raise ValueError("Please squeeze the outputs when using sparse labels")
        if n_history == 1 and self._squeeze:
            outputs = rnn_outputs[..., -1, :]
        else:
            outputs = rnn_outputs[..., -n_history:, :]
            if self._squeeze:
                outputs = tf.reshape(outputs, [-1, n_history * int(outputs.get_shape()[2])])
        if self._use_final_state and self._squeeze:
            outputs = tf.concat([outputs, rnn_final_state], axis=1)
        self._output = layers.fully_connected(
            outputs, num_outputs=self._om, activation_fn=self._activation)

    def fit(self, im, om, generator, generator_params=None, cell=None,
            squeeze=None, sparse=None, embedding_size=None, use_final_state=None, n_hidden=None, n_history=None,
            activation=None, lr=None, epoch=None, n_iter=None, batch_size=None, optimizer=None, eps=None, verbose=None):
        if generator_params is None:
            generator_params = self._params["generator_params"]
        if cell is None:
            cell = self._params["cell"]
        if n_hidden is None:
            n_hidden = self._params["n_hidden"]
        if n_history is None:
            n_history = self._params["n_history"]
        if squeeze:
            self._squeeze = True
        if sparse:
            self._sparse = True
        if embedding_size:
            self._embedding_size = embedding_size
        if use_final_state:
            self._use_final_state = True
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

        self._im, self._om, self._activation = im, om, activation
        self._generator = generator(im, om, **generator_params)
        self._optimizer = OptFactory().get_optimizer_by_name(optimizer, lr)
        self._define_input(im, om)

        cell = cell(n_hidden)
        initial_state = cell.zero_state(tf.shape(self._input)[0], tf.float32)
        rnn_outputs, rnn_final_state = tf.nn.dynamic_rnn(
            cell, self._input, initial_state=initial_state)
        self._get_output(rnn_outputs, rnn_final_state, n_history)
        loss = self._get_loss(eps)
        train_step = self._optimizer.minimize(loss)
        self._log["iter_err"] = []
        self._log["epoch_err"] = []
        self._sess.run(tf.global_variables_initializer())
        bar = ProgressBar(max_value=epoch, name="Epoch", start=False)
        if verbose >= 2:
            bar.start()
        for _ in range(epoch):
            epoch_err = 0
            sub_bar = ProgressBar(max_value=n_iter, name="Iter", start=False)
            if verbose >= 2:
                sub_bar.start()
            for __ in range(n_iter):
                x_batch, y_batch = self._generator.gen(batch_size)
                iter_err = self._sess.run([loss, train_step], {
                    self._tfx: x_batch, self._tfy: y_batch,
                })[0]
                self._log["iter_err"].append(iter_err)
                epoch_err += iter_err
                if verbose >= 2:
                    sub_bar.update()
            self._log["epoch_err"].append(epoch_err / n_iter)
            if verbose >= 1:
                self._verbose()
                if verbose >= 2:
                    bar.update()

    def draw_err_logs(self):
        ee, ie = self._log["epoch_err"], self._log["iter_err"]
        ee_base = np.arange(len(ee))
        ie_base = np.linspace(0, len(ee) - 1, len(ie))
        plt.figure()
        plt.plot(ie_base, ie, label="Iter error")
        plt.plot(ee_base, ee, linewidth=3, label="Epoch error")
        plt.legend()
        plt.show()
