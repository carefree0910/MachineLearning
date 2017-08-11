import numpy as np
from math import ceil

from NN.Errors import *
from NN.TF.Optimizers import *


class Layer:
    LayerTiming = Timing()

    def __init__(self, shape, **kwargs):
        """
        :param shape: shape[0] = units of previous layer
                      shape[1] = units of current layer (self)
        """
        self._shape = shape
        self.parent = None
        self.is_fc = False
        self.is_sub_layer = False
        self.apply_bias = kwargs["apply_bias"]
        self.position = kwargs["position"]

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def init(self, sess):
        pass

    @property
    def name(self):
        return str(self)

    @property
    def root(self):
        return self

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def params(self):
        return self._shape,

    @property
    def info(self):
        return "Layer  :  {:<16s} - {}{}".format(
            self.name, self.shape[1], " (apply_bias: False)" if not self.apply_bias else "")

    def get_special_params(self, sess):
        pass

    def set_special_params(self, dic):
        for key, value in dic.items():
            setattr(self, key, value)

    # Core

    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def activate(self, x, w, bias=None, predict=False):
        if self.is_fc:
            fc_shape = np.prod(x.get_shape()[1:])  # type: int
            x = tf.reshape(x, [-1, int(fc_shape)])
        if self.is_sub_layer:
            if not self.apply_bias:
                return self._activate(x, predict)
            return self._activate(x + bias, predict)
        if not self.apply_bias:
            return self._activate(tf.matmul(x, w), predict)
        return self._activate(tf.matmul(x, w) + bias, predict)

    def _activate(self, x, predict):
        pass


class SubLayer(Layer):
    def __init__(self, parent, shape, **kwargs):
        Layer.__init__(self, shape, **kwargs)
        self.parent = parent
        self.description = ""
        self.is_sub_layer = True

    @property
    def root(self):
        _root = self.parent
        while _root.parent:
            _root = _root.parent
        return _root

    def get_params(self):
        pass

    @property
    def params(self):
        return self.get_params()

    @property
    def info(self):
        return "Layer  :  {:<16s} - {} {}".format(self.name, self.shape[1], self.description)

    def _activate(self, x, predict):
        raise NotImplementedError("Please implement activation function for " + self.name)


class ConvLayer(Layer):
    LayerTiming = Timing()

    def __init__(self, shape, stride=1, padding=None, parent=None, **kwargs):
        """
        :param shape:    shape[0] = shape of previous layer           c x h x w
                         shape[1] = shape of current layer's weight   f x c x h x w
        :param stride:   stride
        :param padding:  zero-padding
        """
        if parent is not None:
            _parent = parent.root if parent.is_sub_layer else parent
            shape = _parent.shape
        Layer.__init__(self, shape, **kwargs)
        self._stride = stride
        if padding is None:
            padding = "SAME"
        if isinstance(padding, str):
            if padding.upper() == "VALID":
                self._padding = 0
                self._pad_flag = "VALID"
            else:
                self._padding = self._pad_flag = "SAME"
        elif isinstance(padding, int):
            self._padding = padding
            self._pad_flag = "VALID"
        else:
            raise BuildLayerError("Padding should be 'SAME' or 'VALID' or integer")
        self.parent = parent
        if len(shape) == 1:
            self.n_channels, self.n_filters, self.out_h, self.out_w = None, None, None, None
        else:
            self.feed_shape(shape)

    def feed_shape(self, shape):
        self._shape = shape
        self.n_channels, height, width = shape[0]
        self.n_filters, filter_height, filter_width = shape[1]
        if self._pad_flag == "VALID":
            self.out_h = ceil((height - filter_height + 1) / self._stride)
            self.out_w = ceil((width - filter_width + 1) / self._stride)
        else:
            self.out_h = ceil(height / self._stride)
            self.out_w = ceil(width / self._stride)

    @property
    def params(self):
        return self._shape, self._stride, self._padding

    @property
    def stride(self):
        return self._stride

    @property
    def padding(self):
        return self._padding

    @property
    def pad_flag(self):
        return self._pad_flag

    @property
    def info(self):
        return "Layer  :  {:<16s} - {:<14s} - strides: {:2d} - padding: {:6s}{} - out: {}".format(
            self.name, str(self.shape[1]), self.stride, self.pad_flag,
            " " * 5 if self.pad_flag == "SAME" else " ({:2d})".format(self.padding),
            (self.n_filters, self.out_h, self.out_w)
        )


class ConvPoolLayer(ConvLayer):
    LayerTiming = Timing()

    @property
    def params(self):
        return (self._shape[0], self._shape[1][1:]), self._stride, self._padding

    def feed_shape(self, shape):
        shape = (shape[0], (shape[0][0], *shape[1]))
        ConvLayer.feed_shape(self, shape)

    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def activate(self, x, w, bias=None, predict=False):
        pool_height, pool_width = self._shape[1][1:]
        if self._pad_flag == "VALID" and self._padding > 0:
            _pad = [self._padding] * 2
            x = tf.pad(x, [[0, 0], _pad, _pad, [0, 0]], "CONSTANT")
        return self._activate(None)(
            x, ksize=[1, pool_height, pool_width, 1],
            strides=[1, self._stride, self._stride, 1], padding=self._pad_flag)

    def _activate(self, x, *args):
        raise NotImplementedError("Please implement activation function for " + self.name)


# noinspection PyProtectedMember
class ConvMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        conv_layer, layer = bases

        def __init__(self, shape, stride=1, padding="SAME", **_kwargs):
            conv_layer.__init__(self, shape, stride, padding, **_kwargs)

        def _conv(self, x, w):
            return tf.nn.conv2d(x, w, strides=[1, self.stride, self.stride, 1], padding=self._pad_flag)

        def _activate(self, x, w, bias, predict):
            res = self._conv(x, w) + bias if self.apply_bias else self._conv(x, w)
            return layer._activate(self, res, predict)

        def activate(self, x, w, bias=None, predict=False):
            if self._pad_flag == "VALID" and self._padding > 0:
                _pad = [self._padding] * 2
                x = tf.pad(x, [[0, 0], _pad, _pad, [0, 0]], "CONSTANT")
            return self.LayerTiming.timeit(level=1, func_name="activate", cls_name=name, prefix="[Core] ")(
                _activate)(self, x, w, bias, predict)

        for key, value in locals().items():
            if str(value).find("function") >= 0 or str(value).find("property"):
                attr[key] = value

        return type(name, bases, attr)


# noinspection PyProtectedMember
class ConvSubMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        conv_layer, sub_layer = bases

        def __init__(self, parent, shape, *_args, **_kwargs):
            conv_layer.__init__(self, None, parent=parent, **_kwargs)
            self.out_h, self.out_w = parent.out_h, parent.out_w
            sub_layer.__init__(self, parent, shape, *_args, **_kwargs)
            self._shape = ((shape[0][0], self.out_h, self.out_w), shape[0])
            if name == "ConvNorm":
                self.tf_gamma = tf.Variable(tf.ones(self.n_filters), name="norm_scale")
                self.tf_beta = tf.Variable(tf.zeros(self.n_filters), name="norm_beta")

        def _activate(self, x, predict):
            return sub_layer._activate(self, x, predict)

        # noinspection PyUnusedLocal
        def activate(self, x, w, bias=None, predict=False):
            return self.LayerTiming.timeit(level=1, func_name="activate", cls_name=name, prefix="[Core] ")(
                _activate)(self, x, predict)

        @property
        def params(self):
            return sub_layer.get_params(self)

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


class CF0910(Layer):
    def _activate(self, x, predict):
        return tf.minimum(tf.maximum(x, 0), 6)


# Convolution Layers

class ConvTanh(ConvLayer, Tanh, metaclass=ConvMeta):
    pass


class ConvSigmoid(ConvLayer, Sigmoid, metaclass=ConvMeta):
    pass


class ConvELU(ConvLayer, ELU, metaclass=ConvMeta):
    pass


class ConvReLU(ConvLayer, ReLU, metaclass=ConvMeta):
    pass


class ConvSoftplus(ConvLayer, Softplus, metaclass=ConvMeta):
    pass


class ConvIdentical(ConvLayer, Identical, metaclass=ConvMeta):
    pass


class ConvCF0910(ConvLayer, CF0910, metaclass=ConvMeta):
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
    def __init__(self, parent, shape, keep_prob=0.5, **kwargs):
        if keep_prob < 0 or keep_prob >= 1:
            raise BuildLayerError("(Dropout) Keep probability of Dropout should be a positive float smaller than 1")
        SubLayer.__init__(self, parent, shape, **kwargs)
        self._keep_prob = keep_prob
        self.description = "(Keep prob: {})".format(keep_prob)

    def get_params(self):
        return self._keep_prob,

    def _activate(self, x, predict):
        if not predict:
            return tf.nn.dropout(x, self._keep_prob)
        return x


class Normalize(SubLayer):
    def __init__(self, parent, shape, activation="Identical", eps=1e-8, momentum=0.9, **kwargs):
        SubLayer.__init__(self, parent, shape, **kwargs)
        self._eps, self._activation = eps, activation
        self.rm = self.rv = None
        self.tf_rm = self.tf_rv = None
        self.tf_gamma = tf.Variable(tf.ones(self.shape[1]), name="norm_scale")
        self.tf_beta = tf.Variable(tf.zeros(self.shape[1]), name="norm_beta")
        self._momentum = momentum
        self.description = "(eps: {}, momentum: {}, activation: {})".format(eps, momentum, activation)

    def init(self, sess):
        if self.rm is not None:
            self.tf_rm = tf.Variable(self.rm, trainable=False, name="norm_mean")
        if self.rv is not None:
            self.tf_rv = tf.Variable(self.rv, trainable=False, name="norm_var")
        sess.run(tf.variables_initializer([self.tf_rm, self.tf_rv]))

    def get_special_params(self, sess):
        with sess.as_default():
            return {
                "rm": self.tf_rm.eval(), "rv": self.tf_rv.eval(),
            }

    def get_params(self):
        return self._activation, self._eps, self._momentum

    # noinspection PyTypeChecker
    def _activate(self, x, predict):
        if self.tf_rm is None or self.tf_rv is None:
            shape = x.get_shape()[-1]
            self.tf_rm = tf.Variable(tf.zeros(shape), trainable=False, name="norm_mean")
            self.tf_rv = tf.Variable(tf.ones(shape), trainable=False, name="norm_var")
        if not predict:
            _sm, _sv = tf.nn.moments(x, list(range(len(x.get_shape()) - 1)))
            _rm = tf.assign(self.tf_rm, self._momentum * self.tf_rm + (1 - self._momentum) * _sm)
            _rv = tf.assign(self.tf_rv, self._momentum * self.tf_rv + (1 - self._momentum) * _sv)
            with tf.control_dependencies([_rm, _rv]):
                _norm = tf.nn.batch_normalization(x, _sm, _sv, self.tf_beta, self.tf_gamma, self._eps)
        else:
            _norm = tf.nn.batch_normalization(x, self.tf_rm, self.tf_rv, self.tf_beta, self.tf_gamma, self._eps)
        if self._activation == "ReLU":
            return tf.nn.relu(_norm)
        if self._activation == "Sigmoid":
            return tf.nn.sigmoid(_norm)
        return _norm


class ConvDrop(ConvLayer, Dropout, metaclass=ConvSubMeta):
    pass


class ConvNorm(ConvLayer, Normalize, metaclass=ConvSubMeta):
    pass


# Cost Layers

class CostLayer(Layer):
    @property
    def info(self):
        return "Cost   :  {:<16s}".format(self.name)

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
        "CF0910": CF0910,

        # Cost Layers
        "CrossEntropy": CrossEntropy, "MSE": MSE,

        # Conv Layers
        "ConvTanh": ConvTanh, "ConvSigmoid": ConvSigmoid,
        "ConvELU": ConvELU, "ConvReLU": ConvReLU, "ConvSoftplus": ConvSoftplus,
        "ConvIdentical": ConvIdentical,
        "ConvCF0910": ConvCF0910,
        "MaxPool": MaxPool, "AvgPool": AvgPool
    }
    available_special_layers = {
        "Dropout": Dropout,
        "Normalize": Normalize,
        "ConvDrop": ConvDrop,
        "ConvNorm": ConvNorm
    }

    def get_root_layer_by_name(self, name, *args, **kwargs):
        if name not in self.available_special_layers:
            if name in self.available_root_layers:
                name = self.available_root_layers[name]
            else:
                raise BuildNetworkError("Undefined layer '{}' found".format(name))
            return name(*args, **kwargs)
        return None

    def get_layer_by_name(self, name, parent, current_dimension, *args, **kwargs):
        _layer = self.get_root_layer_by_name(name, *args, **kwargs)
        if _layer:
            return _layer, None
        _current, _next = parent.shape[1], current_dimension
        _layer = self.available_special_layers[name]
        if args:
            _layer = _layer(parent, (_current, _next), *args, **kwargs)
        else:
            _layer = _layer(parent, (_current, _next), **kwargs)
        return _layer, (_current, _next)
