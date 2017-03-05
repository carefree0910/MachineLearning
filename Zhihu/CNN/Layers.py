import numpy as np
import tensorflow as tf
from math import ceil
from abc import ABCMeta, abstractmethod

from Util.Timing import Timing


class Layer(metaclass=ABCMeta):
    def __init__(self, shape):
        self.shape = shape
        self.is_fc = False
        self.is_fc_base = False

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def activate(self, x, w, bias=None, predict=False):
        if self.is_fc:
            x = tf.reshape(x, [-1, int(np.prod(x.get_shape()[1:]))])
        if bias is None:
            return self._activate(tf.matmul(x, w), predict)
        return self._activate(tf.matmul(x, w) + bias, predict)

    @abstractmethod
    def _activate(self, x, predict):
        pass


class ConvLayer(Layer):
    LayerTiming = Timing()

    def __init__(self, shape, stride=1, padding="SAME", parent=None):
        """
        :param shape:    shape[0] = shape of previous layer           c x h x w
                         shape[1] = shape of current layer's weight   f x c x h x w
        :param stride:   stride
        :param padding:  zero-padding
        :param parent:   parent
        """
        if parent is not None:
            shape, stride, padding = parent.shape, parent.stride, parent.padding
        Layer.__init__(self, shape)
        self._stride = stride
        if isinstance(padding, str):
            if padding.upper() == "VALID":
                self._padding = 0
                self._pad_flag = "VALID"
            else:
                self._padding = self._pad_flag = "SAME"
        else:
            self._padding = int(padding)
            self._pad_flag = "VALID"
        self.parent = parent
        if len(shape) == 1:
            self.n_channels = self.n_filters = self.out_h = self.out_w = None
        else:
            self.feed_shape(shape)

    def feed_shape(self, shape):
        self.shape = shape
        self.n_channels, height, width = shape[0]
        self.n_filters, filter_height, filter_width = shape[1]
        if self._pad_flag == "VALID":
            self.out_h = ceil((height - filter_height + 1) / self._stride)
            self.out_w = ceil((width - filter_width + 1) / self._stride)
        else:
            self.out_h = ceil(height / self._stride)
            self.out_w = ceil(width / self._stride)

    @property
    def stride(self):
        return self._stride

    @property
    def padding(self):
        return self._padding

    @property
    def pad_flag(self):
        return self._pad_flag


class ConvPoolLayer(ConvLayer):
    LayerTiming = Timing()

    def feed_shape(self, shape):
        shape = (shape[0], (shape[0][0], *shape[1]))
        ConvLayer.feed_shape(self, shape)

    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def activate(self, x, w, bias=None, predict=False):
        pool_height, pool_width = self.shape[1][1:]
        if self._pad_flag == "VALID" and self._padding > 0:
            _pad = [self._padding] * 2
            x = tf.pad(x, [[0, 0], _pad, _pad, [0, 0]], "CONSTANT")
        return self._activate(None)(
            x, ksize=[1, pool_height, pool_width, 1],
            strides=[1, self._stride, self._stride, 1], padding=self._pad_flag)

    def _activate(self, x, *args):
        raise NotImplementedError("Please implement activation function for {}".format(str(self)))


class ConvMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        conv_layer, layer = bases

        def __init__(self, shape, stride=1, padding="SAME"):
            conv_layer.__init__(self, shape, stride, padding)

        def _conv(self, x, w):
            return tf.nn.conv2d(x, w, strides=[self._stride] * 4, padding=self._pad_flag)

        def _activate(self, x, w, bias, predict):
            res = self._conv(x, w) + bias
            return layer._activate(self, res, predict)

        def activate(self, x, w, bias=None, predict=False):
            if self._pad_flag == "VALID" and self._padding > 0:
                _pad = [self._padding] * 2
                x = tf.pad(x, [[0, 0], _pad, _pad, [0, 0]], "CONSTANT")
            return _activate(self, x, w, bias, predict)

        for key, value in locals().items():
            if str(value).find("function") >= 0:
                attr[key] = value

        return type(name, bases, attr)


class ConvLayerMeta(ABCMeta, ConvMeta):
    pass


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


class ConvCF0910(ConvLayer, CF0910, metaclass=ConvLayerMeta):
    pass


# Pooling Layers

class MaxPool(ConvPoolLayer):
    def _activate(self, x, *args):
        return tf.nn.max_pool


class AvgPool(ConvPoolLayer):
    def _activate(self, x, *args):
        return tf.nn.avg_pool


# Cost Layers

class CostLayer(Layer):
    def _activate(self, x, y):
        pass

    def calculate(self, y, y_pred):
        return self._activate(y.astype(np.float32), y_pred)


class CrossEntropy(CostLayer):
    def _activate(self, x, y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(x, y))


class MSE(CostLayer):
    def _activate(self, x, y):
        return tf.reduce_mean(tf.square(x - y))
