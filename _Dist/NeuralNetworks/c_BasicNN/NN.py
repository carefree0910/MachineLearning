import os
import sys
root_path = os.path.abspath("../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import tensorflow as tf

from _Dist.NeuralNetworks.Util import *
from _Dist.NeuralNetworks.Base import Base


class Basic(Base):
    def __init__(self, *args, **kwargs):
        super(Basic, self).__init__(*args, *kwargs)
        self.model_saving_name = "NN"
        self.activations = kwargs.get("activations", "relu")
        self.hidden_units = self.kwargs.get("hidden_units", (256, 256))

    def _define_py_collections(self):
        self.py_collections = ["hidden_units"]

    def _define_input(self):
        self._tfx = tf.placeholder(tf.float32, [None, self.n_dim])
        self._tfy = tf.placeholder(tf.float32, [None, self.n_class])

    def _fully_connected_linear(self, net, shape, appendix):
        w = init_w(shape, "W{}".format(appendix))
        b = init_b([shape[1]], "b{}".format(appendix))
        self._ws.append(w)
        self._bs.append(b)
        return tf.nn.xw_plus_b(net, w, b, "linear{}".format(appendix))

    def _build_layer(self, i, net):
        activation = self.activations[i]
        if activation is not None:
            net = getattr(Activations, activation)(net, "{}{}".format(activation, i))
        return net

    def _build_model(self):
        self._model_built = True
        self._is_training = tf.placeholder(tf.bool)
        net = self._tfx
        current_dimension = self.n_dim
        if self.activations is None:
            self.activations = [None] * len(self.hidden_units)
        elif isinstance(self.activations, str):
            self.activations = [self.activations] * len(self.hidden_units)
        else:
            self.activations = self.activations
        for i, n_unit in enumerate(self.hidden_units):
            net = self._fully_connected_linear(net, [current_dimension, n_unit], i)
            net = self._build_layer(i, net)
            current_dimension = n_unit
        appendix = "_final_projection"
        fc_shape = self.hidden_units[-1] if self.hidden_units else current_dimension
        self._output = self._fully_connected_linear(net, [fc_shape, self.n_class], appendix)

    def _get_feed_dict(self, x, y=None, is_training=True):
        feed_dict = super(Basic, self)._get_feed_dict(x, y, is_training)
        feed_dict[self._is_training] = is_training
        return feed_dict
