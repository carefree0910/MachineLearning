import os
import sys
root_path = os.path.abspath("../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import tensorflow as tf

from _Dist.NeuralNetworks.Base import Generator4d
from _Dist.NeuralNetworks.h_RNN.RNN import Basic3d
from _Dist.NeuralNetworks.NNUtil import Activations


class Basic4d(Basic3d):
    def _calculate(self, x, y=None, weights=None, tensor=None, n_elem=1e7, is_training=False):
        return super(Basic4d, self)._calculate(x, y, weights, tensor, n_elem / 10, is_training)


class CNN(Basic4d):
    def __init__(self, *args, **kwargs):
        self.height, self.width = kwargs.pop("height", None), kwargs.pop("width", None)

        super(CNN, self).__init__(*args, **kwargs)
        self._name_appendix = "CNN"
        self._generator_base = Generator4d

        self.conv_activations = None
        self.n_filters = self.filter_sizes = self.poolings = None

    def init_model_param_settings(self):
        super(CNN, self).init_model_param_settings()
        self.conv_activations = self.model_param_settings.get("conv_activations", "relu")

    def init_model_structure_settings(self):
        super(CNN, self).init_model_structure_settings()
        self.n_filters = self.model_structure_settings.get("n_filters", [32, 32])
        self.filter_sizes = self.model_structure_settings.get("filter_sizes", [(3, 3), (3, 3)])
        self.poolings = self.model_structure_settings.get("poolings", [None, "max_pool"])
        if not len(self.filter_sizes) == len(self.poolings) == len(self.n_filters):
            raise ValueError("Length of filter_sizes, n_filters & pooling should be the same")
        if isinstance(self.conv_activations, str):
            self.conv_activations = [self.conv_activations] * len(self.filter_sizes)

    def init_from_data(self, x, y, x_test, y_test, sample_weights, names):
        if self.height is None or self.width is None:
            assert len(x.shape) == 4, "height and width are not provided, hence len(x.shape) should be 4"
            self.height, self.width = x.shape[1:3]
        if len(x.shape) == 2:
            x = x.reshape(len(x), self.height, self.width, -1)
        else:
            assert self.height == x.shape[1], "height is set to be {}, but {} found".format(self.height, x.shape[1])
            assert self.width == x.shape[2], "width is set to be {}, but {} found".format(self.height, x.shape[2])
        if x_test is not None and len(x_test.shape) == 2:
            x_test = x_test.reshape(len(x_test), self.height, self.width, -1)
        super(CNN, self).init_from_data(x, y, x_test, y_test, sample_weights, names)

    def _define_input_and_placeholder(self):
        self._is_training = tf.placeholder(tf.bool, name="is_training")
        self._tfx = tf.placeholder(tf.float32, [None, self.height, self.width, self.n_dim], name="X")
        self._tfy = tf.placeholder(tf.float32, [None, self.n_class], name="Y")

    def _build_model(self, net=None):
        self._model_built = True
        if net is None:
            net = self._tfx
        for i, (filter_size, n_filter, pooling) in enumerate(zip(
            self.filter_sizes, self.n_filters, self.poolings
        )):
            net = tf.layers.conv2d(net, n_filter, filter_size, padding="same")
            net = tf.layers.batch_normalization(net, training=self._is_training)
            activation = self.conv_activations[i]
            if activation is not None:
                net = getattr(Activations, activation)(net, activation)
            net = tf.layers.dropout(net, training=self._is_training)
            if pooling is not None:
                net = tf.layers.max_pooling2d(net, 2, 2, name="pool")

        fc_shape = np.prod([net.shape[i].value for i in range(1, 4)])
        net = tf.reshape(net, [-1, fc_shape])
        super(CNN, self)._build_model(net)
