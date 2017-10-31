import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import tensorflow as tf
from math import ceil

from Util.Timing import Timing


# Abstract Layers

class Layer:
    LayerTiming = Timing

    def __init__(self, shape, **kwargs):
        self.shape = shape
        self.is_fc = self.is_sub_layer = False
        self.apply_bias = kwargs.get("apply_bias", True)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return str(self)

    @property
    def root(self):
        return self

    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def activate(self, x, w, bias=None, predict=False):
        if self.is_fc:
            fc_shape = np.prod(x.get_shape()[1:])  # type: int
            x = tf.reshape(x, [-1, int(fc_shape)])
        if self.is_sub_layer:
            return self._activate(x, predict)
        if not self.apply_bias:
            return self._activate(tf.matmul(x, w), predict)
        return self._activate(tf.matmul(x, w) + bias, predict)

    def _activate(self, x, predict):
        pass


class SubLayer(Layer):
    def __init__(self, parent, shape):
        Layer.__init__(self, shape)
        self.parent = parent
        self.description = ""
        self.is_sub_layer = True

    @property
    def root(self):
        root = self.parent
        while root.parent:
            root = root.parent
        return root

    @property
    def info(self):
        return "Layer  :  {:<16s} - {} {}".format(self.name, self.shape[1], self.description)


class ConvLayer(Layer):
    LayerTiming = Timing()

    def __init__(self, shape, stride=1, padding=None, parent=None):
        """
        :param shape:    shape[0] = shape of previous layer           c x h x w
                         shape[1] = shape of current layer's weight   f x h x w
        :param stride:   stride
        :param padding:  zero-padding
        :param parent:   parent
        """
        if parent is not None:
            _parent = parent.root if parent.is_sub_layer else parent
            shape = _parent.shape
        Layer.__init__(self, shape)
        self.stride = stride
        if padding is None:
            padding = "SAME"
        if isinstance(padding, str):
            if padding.upper() == "VALID":
                self.padding = 0
                self.pad_flag = "VALID"
            else:
                self.padding = self.pad_flag = "SAME"
        elif isinstance(padding, int):
            self.padding = padding
            self.pad_flag = "VALID"
        else:
            raise ValueError("Padding should be 'SAME' or 'VALID' or integer")
        self.parent = parent
        if len(shape) == 1:
            self.n_channels = self.n_filters = self.out_h = self.out_w = None
        else:
            self.feed_shape(shape)

    def feed_shape(self, shape):
        self.shape = shape
        self.n_channels, height, width = shape[0]
        self.n_filters, filter_height, filter_width = shape[1]
        if self.pad_flag == "VALID":
            self.out_h = ceil((height - filter_height + 1) / self.stride)
            self.out_w = ceil((width - filter_width + 1) / self.stride)
        else:
            self.out_h = ceil(height / self.stride)
            self.out_w = ceil(width / self.stride)


class ConvPoolLayer(ConvLayer):
    LayerTiming = Timing()

    def feed_shape(self, shape):
        shape = (shape[0], (shape[0][0], *shape[1]))
        ConvLayer.feed_shape(self, shape)

    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def activate(self, x, w, bias=None, predict=False):
        pool_height, pool_width = self.shape[1][1:]
        if self.pad_flag == "VALID" and self.padding > 0:
            _pad = [self.padding] * 2
            x = tf.pad(x, [[0, 0], _pad, _pad, [0, 0]], "CONSTANT")
        return self._activate(None)(
            x, ksize=[1, pool_height, pool_width, 1],
            strides=[1, self.stride, self.stride, 1], padding=self.pad_flag)

    def _activate(self, x, *args):
        raise NotImplementedError("Please implement activation function for {}".format(str(self)))


class ConvLayerMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        conv_layer, layer = bases

        def __init__(self, shape, stride=1, padding="SAME"):
            conv_layer.__init__(self, shape, stride, padding)

        def _conv(self, x, w):
            return tf.nn.conv2d(x, w, strides=[1, self.stride, self.stride, 1], padding=self.pad_flag)

        def _activate(self, x, w, bias, predict):
            res = self._conv(x, w) + bias
            return layer._activate(self, res, predict)

        def activate(self, x, w, bias=None, predict=False):
            if self.pad_flag == "VALID" and self.padding > 0:
                _pad = [self.padding] * 2
                x = tf.pad(x, [[0, 0], _pad, _pad, [0, 0]], "CONSTANT")
            return _activate(self, x, w, bias, predict)

        for key, value in locals().items():
            if str(value).find("function") >= 0:
                attr[key] = value

        return type(name, bases, attr)


class ConvSubLayerMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        conv_layer, sub_layer = bases

        def __init__(self, parent, shape, *_args, **_kwargs):
            conv_layer.__init__(self, None, parent=parent)
            self.out_h, self.out_w = parent.out_h, parent.out_w
            sub_layer.__init__(self, parent, shape, *_args, **_kwargs)
            self.shape = ((shape[0][0], self.out_h, self.out_w), shape[0])
            if name == "ConvNorm":
                self.tf_gamma = tf.Variable(tf.ones(self.n_filters), name="norm_scale")
                self.tf_beta = tf.Variable(tf.zeros(self.n_filters), name="norm_beta")

        def _activate(self, x, predict):
            return sub_layer._activate(self, x, predict)

        # noinspection PyUnusedLocal
        def activate(self, x, w, bias=None, predict=False):
            return self.LayerTiming.timeit(level=1, func_name="activate", cls_name=name, prefix="[Core] ")(
                _activate)(self, x, predict)

        for key, value in locals().items():
            if str(value).find("function") >= 0 or str(value).find("property"):
                attr[key] = value

        return type(name, bases, attr)


# Activation Layers

class Tanh(Layer):
    def _activate(self, x, predict):
        return tf.tanh(x)


class Sigmoid(Layer):
    def _activate(self, x, predict):
        return tf.nn.sigmoid(x)


class ELU(Layer):
    def _activate(self, x, predict):
        return tf.nn.elu(x)


class ReLU(Layer):
    def _activate(self, x, predict):
        return tf.nn.relu(x)


class Softplus(Layer):
    def _activate(self, x, predict):
        return tf.nn.softplus(x)


class Identical(Layer):
    def _activate(self, x, predict):
        return x


# Convolution Layers

class ConvTanh(ConvLayer, Tanh, metaclass=ConvLayerMeta):
    pass


class ConvSigmoid(ConvLayer, Sigmoid, metaclass=ConvLayerMeta):
    pass


class ConvELU(ConvLayer, ELU, metaclass=ConvLayerMeta):
    pass


class ConvReLU(ConvLayer, ReLU, metaclass=ConvLayerMeta):
    pass


class ConvSoftplus(ConvLayer, Softplus, metaclass=ConvLayerMeta):
    pass


class ConvIdentical(ConvLayer, Identical, metaclass=ConvLayerMeta):
    pass


# Pooling Layers

class MaxPool(ConvPoolLayer):
    def _activate(self, x, *args):
        return tf.nn.max_pool


class AvgPool(ConvPoolLayer):
    def _activate(self, x, *args):
        return tf.nn.avg_pool


# Special Layers

class Dropout(SubLayer):
    def __init__(self, parent, shape, drop_prob=0.5):
        if drop_prob < 0 or drop_prob >= 1:
            raise ValueError("(Dropout) Probability of Dropout should be a positive float smaller than 1")
        SubLayer.__init__(self, parent, shape)
        self._prob = tf.constant(1 - drop_prob, dtype=tf.float32)
        self.description = "(Drop prob: {})".format(drop_prob)

    def _activate(self, x, predict):
        if not predict:
            return tf.nn.dropout(x, self._prob)
        return x


class Normalize(SubLayer):
    def __init__(self, parent, shape, activation="ReLU", eps=1e-8, momentum=0.9):
        SubLayer.__init__(self, parent, shape)
        self._eps, self._activation = eps, activation
        self.tf_rm = self.tf_rv = None
        self.tf_gamma = tf.Variable(tf.ones(self.shape[1]), name="norm_scale")
        self.tf_beta = tf.Variable(tf.zeros(self.shape[1]), name="norm_beta")
        self._momentum = momentum
        self.description = "(eps: {}, momentum: {})".format(eps, momentum)

    # noinspection PyTypeChecker
    def _activate(self, x, predict):
        if self.tf_rm is None or self.tf_rv is None:
            shape = x.get_shape()[-1]
            self.tf_rm = tf.Variable(tf.zeros(shape), trainable=False, name="norm_mean")
            self.tf_rv = tf.Variable(tf.ones(shape), trainable=False, name="norm_var")
        if not predict:
            sm, sv = tf.nn.moments(x, list(range(len(x.get_shape()) - 1)))
            rm = tf.assign(self.tf_rm, self._momentum * self.tf_rm + (1 - self._momentum) * sm)
            rv = tf.assign(self.tf_rv, self._momentum * self.tf_rv + (1 - self._momentum) * sv)
            with tf.control_dependencies([rm, rv]):
                norm = tf.nn.batch_normalization(x, sm, sv, self.tf_beta, self.tf_gamma, self._eps)
        else:
            norm = tf.nn.batch_normalization(x, self.tf_rm, self.tf_rv, self.tf_beta, self.tf_gamma, self._eps)
        if self._activation == "ReLU":
            return tf.nn.relu(norm)
        if self._activation == "Sigmoid":
            return tf.nn.sigmoid(norm)
        return norm


class ConvDrop(ConvLayer, Dropout, metaclass=ConvSubLayerMeta):
    pass


class ConvNorm(ConvLayer, Normalize, metaclass=ConvSubLayerMeta):
    pass


# Cost Layers

class CostLayer(Layer):
    def calculate(self, y, y_pred):
        return self._activate(y_pred, y)


class CrossEntropy(CostLayer):
    def _activate(self, x, y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y))


class MSE(CostLayer):
    def _activate(self, x, y):
        return tf.reduce_mean(tf.square(x - y))


# Factory

class LayerFactory:
    available_root_layers = {
        # Normal Layers
        "Tanh": Tanh, "Sigmoid": Sigmoid,
        "ELU": ELU, "ReLU": ReLU, "Softplus": Softplus,
        "Identical": Identical,

        # Cost Layers
        "CrossEntropy": CrossEntropy, "MSE": MSE,

        # Conv Layers
        "ConvTanh": ConvTanh, "ConvSigmoid": ConvSigmoid,
        "ConvELU": ConvELU, "ConvReLU": ConvReLU, "ConvSoftplus": ConvSoftplus,
        "ConvIdentical": ConvIdentical,
        "MaxPool": MaxPool, "AvgPool": AvgPool
    }
    available_special_layers = {
        "Dropout": Dropout,
        "Normalize": Normalize,
        "ConvDrop": ConvDrop,
        "ConvNorm": ConvNorm
    }
    special_layer_default_params = {
        "Dropout": (0.5,),
        "Normalize": ("Identical", 1e-8, 0.9),
        "ConvDrop": (0.5,),
        "ConvNorm": ("Identical", 1e-8, 0.9)
    }

    def get_root_layer_by_name(self, name, *args, **kwargs):
        if name not in self.available_special_layers:
            if name in self.available_root_layers:
                layer = self.available_root_layers[name]
            else:
                raise ValueError("Undefined layer '{}' found".format(name))
            return layer(*args, **kwargs)
        return None

    def get_layer_by_name(self, name, parent, current_dimension, *args, **kwargs):
        layer = self.get_root_layer_by_name(name, *args, **kwargs)
        if layer:
            return layer, None
        _current, _next = parent.shape[1], current_dimension
        layer_param = self.special_layer_default_params[name]
        layer = self.available_special_layers[name]
        if args or kwargs:
            layer = layer(parent, (_current, _next), *args, **kwargs)
        else:
            layer = layer(parent, (_current, _next), *layer_param)
        return layer, (_current, _next)

if __name__ == '__main__':
    with tf.Session().as_default() as sess:
        # NN Process
        nn_x = np.array([
            [ 0,  1,  2,  1,  0],
            [-1, -2,  0,  2,  1],
            [ 0,  1, -2, -1,  2],
            [ 1,  2, -1,  0, -2]
        ], dtype=np.float32)
        nn_w = np.array([
            [-2, -1, 0,  1,  2],
            [ 2,  1, 0, -1, -2]
        ], dtype=np.float32).T
        nn_b = 1.
        nn_id = Identical([nn_x.shape[1], 2])
        print(nn_id.activate(nn_x, nn_w, nn_b).eval())
        # CNN Process
        conv_x = np.array([
            [
                [ 0, 2,  1, 2],
                [-1, 0,  0, 1],
                [ 1, 1,  0, 1],
                [-2, 1, -1, 0]
            ]
        ], dtype=np.float32).reshape(1, 4, 4, 1)
        conv_w = np.array([
            [[ 1, 0,  1],
             [-1, 0,  1],
             [ 1, 0, -1]],
            [[0,  1,  0],
             [1,  0, -1],
             [0, -1,  1]]
        ], dtype=np.float32).transpose([1, 2, 0])[..., None, :]
        conv_b = np.array([1, -1], dtype=np.float32)
        # Using "VALID" Padding -> out_h = out_w = 2
        conv_id = ConvIdentical([(conv_x.shape[1:], [2, 3, 3])], padding="VALID")
        print(conv_id.activate(conv_x, conv_w, conv_b).eval())
        conv_x = np.array([
            [
                [ 1,  2,  1],
                [-1,  0, -2],
                [ 1, -1,  2]
            ]
        ], dtype=np.float32).reshape(1, 3, 3, 1)
        """
        Using "SAME" Padding -> out_h = out_w = 3 & input_x = 
            [ [ 0  0  0  0  0 ]
              [ 0  1  2  1  0 ]
              [ 0 -1  0 -2  0 ]
              [ 0  1 -1  2  0 ]
              [ 0  0  0  0  0 ] ]
        """
        conv_id = ConvIdentical([(conv_x.shape[1:], [2, 3, 3])], padding=1, stride=2)
        print(conv_id.activate(conv_x, conv_w, conv_b).eval())
