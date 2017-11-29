import os
import sys
root_path = os.path.abspath("../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import tensorflow as tf

from _Dist.NeuralNetworks.NNUtil import Toolbox
from _Dist.NeuralNetworks.Base import Generator3d
from _Dist.NeuralNetworks.c_BasicNN.NN import Basic
from _Dist.NeuralNetworks.h_RNN.Cell import CellFactory


class Basic3d(Basic):
    def _gen_batch(self, generator, n_batch, gen_random_subset=False, one_hot=False):
        if gen_random_subset:
            data, weights = generator.gen_random_subset(n_batch)
        else:
            data, weights = generator.gen_batch(n_batch)
        x = np.array([d[0] for d in data], np.float32)
        y = np.array([d[1] for d in data], np.float32)
        if not one_hot:
            return x, y, weights
        if self.n_class == 1:
            y = y.reshape([-1, 1])
        else:
            y = Toolbox.get_one_hot(y, self.n_class)
        return x, y, weights


class RNN(Basic3d):
    def __init__(self, *args, **kwargs):
        self.n_time_step = kwargs.pop("n_time_step", None)

        super(RNN, self).__init__(*args, **kwargs)
        self._name_appendix = "RNN"
        self._generator_base = Generator3d

        self._using_dndf_cell = False
        self._n_batch_placeholder = None
        self._cell = self._cell_name = None
        self.n_hidden = self.n_history = self.use_final_state = None

    def init_model_param_settings(self):
        super(RNN, self).init_model_param_settings()
        self._cell_name = self.model_param_settings.get("cell", "CustomLSTM")

    def init_model_structure_settings(self):
        super(RNN, self).init_model_structure_settings()
        self.n_hidden = self.model_structure_settings.get("n_hidden", 128)
        self.n_history = self.model_structure_settings.get("n_history", 0)
        self.use_final_state = self.model_structure_settings.get("use_final_state", True)

    def init_from_data(self, x, y, x_test, y_test, sample_weights, names):
        if self.n_time_step is None:
            assert len(x.shape) == 3, "n_time_step is not provided, hence len(x.shape) should be 3"
            self.n_time_step = x.shape[1]
        if len(x.shape) == 2:
            x = x.reshape(len(x), self.n_time_step, -1)
        else:
            assert self.n_time_step == x.shape[1], "n_time_step is set to be {}, but {} found".format(
                self.n_time_step, x.shape[1]
            )
        if len(x_test.shape) == 2:
            x_test = x_test.reshape(len(x_test), self.n_time_step, -1)
        super(RNN, self).init_from_data(x, y, x_test, y_test, sample_weights, names)

    def _define_input_and_placeholder(self):
        self._is_training = tf.placeholder(tf.bool, name="is_training")
        self._tfx = tf.placeholder(tf.float32, [None, self.n_time_step, self.n_dim], name="X")
        self._tfy = tf.placeholder(tf.float32, [None, self.n_class], name="Y")

    def _build_model(self, net=None):
        self._model_built = True
        if net is None:
            net = self._tfx

        self._cell = CellFactory.get_cell(self._cell_name, self.n_hidden)
        if "DNDF" in self._cell_name:
            self._using_dndf_cell = True
            self._n_batch_placeholder = self._cell.n_batch_placeholder

        initial_state = self._cell.zero_state(tf.shape(net)[0], tf.float32)
        rnn_outputs, rnn_final_state = tf.nn.dynamic_rnn(self._cell, net, initial_state=initial_state)

        if self.n_history == 0:
            net = None
        elif self.n_history == 1:
            net = rnn_outputs[..., -1, :]
        else:
            net = rnn_outputs[..., -self.n_history:, :]
            net = tf.reshape(net, [-1, self.n_history * int(net.shape[2].value)])
        if self.use_final_state:
            if net is None:
                net = rnn_final_state[1]
            else:
                net = tf.concat([net, rnn_final_state[1]], axis=1)
        return super(RNN, self)._build_model(net)

    def _get_feed_dict(self, x, y=None, weights=None, is_training=False):
        feed_dict = super(RNN, self)._get_feed_dict(x, y, weights, is_training)
        if self._using_dndf_cell:
            feed_dict[self._n_batch_placeholder] = len(x)
        return feed_dict

    def _define_py_collections(self):
        super(RNN, self)._define_py_collections()
        self.py_collections.append("_using_dndf_cell")

    def _define_tf_collections(self):
        super(RNN, self)._define_tf_collections()
        self.tf_collections.append("_n_batch_placeholder")
