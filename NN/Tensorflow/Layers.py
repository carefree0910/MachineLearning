import numpy as np
from abc import ABCMeta, abstractmethod

from Errors import *
from Tensorflow.Optimizers import *
from Util import Timing


class Layer(metaclass=ABCMeta):
    LayerTiming = Timing()

    def __init__(self, shape):
        """
        :param shape: shape[0] = units of previous layer
                      shape[1] = units of current layer (self)
        """
        self._shape = shape
        self.parent = None
        self.is_fc = False
        self.is_fc_base = False
        self.is_sub_layer = False

    def init(self):
        pass

    def feed_timing(self, timing):
        if isinstance(timing, Timing):
            self.LayerTiming = timing

    @property
    def name(self):
        return str(self)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def params(self):
        return self._shape,

    def get_special_params(self, sess):
        pass

    def set_special_params(self, dic):
        for key, value in dic.items():
            setattr(self, key, value)

    # Core

    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def activate(self, x, w, bias=None, predict=False):
        if self.is_fc:
            x = tf.reshape(x, [-1, int(np.prod(x.get_shape()[1:]))])
        if self.is_sub_layer:
            if bias is None:
                return self._activate(x, predict)
            return self._activate(x + bias, predict)
        if bias is None:
            return self._activate(tf.matmul(x, w), predict)
        return self._activate(tf.matmul(x, w) + bias, predict)

    @abstractmethod
    def _activate(self, x, predict):
        pass

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)


class SubLayer(Layer):
    def __init__(self, parent, shape):
        Layer.__init__(self, shape)
        self.parent = parent
        self.description = ""

    @abstractmethod
    def get_params(self):
        pass

    @property
    def params(self):
        return self.get_params()

    def _activate(self, x, predict):
        raise NotImplementedError("Please implement activation function for " + self.name)


class ConvLayer(Layer):
    LayerTiming = Timing()

    def __init__(self, shape, stride=1, padding=0, parent=None):
        """
        :param shape:    shape[0] = shape of previous layer           c x h x w
                         shape[1] = shape of current layer's weight   f x c x h x w
        :param stride:   stride
        :param padding:  zero-padding
        """
        if parent is not None:
            _parent = parent.root if parent.is_sub_layer else parent
            shape, stride, padding = _parent.shape, _parent.stride, _parent.padding
        Layer.__init__(self, shape)
        self._stride, self._padding = stride, padding
        self.parent = parent
        if len(shape) == 1:
            self.n_channels, self.n_filters, self.out_h, self.out_w = None, None, None, None
        else:
            self.feed_shape(shape)

    def feed_shape(self, shape):
        self._shape = shape
        self.n_channels, height, width = shape[0]
        self.n_filters, filter_height, filter_width = shape[1]
        full_height, full_width = height + 2 * self._padding, width + 2 * self._padding
        if (
            (full_height - filter_height) % self._stride != 0 or
            (full_width - filter_width) % self._stride != 0
        ):
            raise BuildLayerError(
                "({}) Weight shape does not work, "
                "shape: {} - stride: {} - padding: {} not compatible with {}".format(
                    self.name, self._shape[1][1:], self._stride, self._padding, (height, width)
                ))
        self.out_h = int((full_height - filter_height) / self._stride) + 1
        self.out_w = int((full_width - filter_width) / self._stride) + 1

    @property
    def params(self):
        return self._shape, self._stride, self._padding

    @property
    def stride(self):
        return self._stride

    @property
    def padding(self):
        return self._padding


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
        _pad = [self._padding] * 2
        x = tf.pad(x, [[0, 0], _pad, _pad, [0, 0]], "CONSTANT")
        return self._activate(x, w, bias, predict)

    def _activate(self, x, *args):
        raise NotImplementedError("Please implement activation function for " + self.name)


# noinspection PyUnusedLocal,PyProtectedMember
class ConvMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        conv_layer, layer = bases

        def __init__(self, shape, stride=1, padding=0):
            conv_layer.__init__(self, shape, stride, padding)

        def _conv(self, x, w):
            return tf.nn.conv2d(x, w, strides=[self._stride] * 4, padding='VALID')

        def _activate(self, x, w, bias, predict):
            res = self._conv(x, w) + bias
            return layer._activate(self, res, predict)

        def activate(self, x, w, bias=None, predict=False):
            _pad = [self._padding] * 2
            x = tf.pad(x, [[0, 0], _pad, _pad, [0, 0]], "CONSTANT")
            return self.LayerTiming.timeit(level=1, name="activate", cls_name=name, prefix="[Core] ")(
                _activate)(self, x, w, bias, predict)

        for key, value in locals().items():
            if str(value).find("function") >= 0 or str(value).find("property"):
                attr[key] = value

        return type(name, bases, attr)


# noinspection PyUnusedLocal,PyProtectedMember
class ConvSubMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        conv_layer, sub_layer = bases

        def __init__(self, parent, shape, *_args, **_kwargs):
            conv_layer.__init__(self, None, parent=parent)
            self.out_h, self.out_w = parent.out_h, parent.out_w
            sub_layer.__init__(self, parent, shape, *_args, **_kwargs)
            self._shape = ((shape[0][0], self.out_h, self.out_w), shape[0])
            if name == "ConvNorm":
                self.tf_gamma = tf.Variable(tf.ones(self.n_filters), name="norm_scale")
                self.tf_beta = tf.Variable(tf.zeros(self.n_filters), name="norm_beta")

        def _activate(self, x, predict):
            return sub_layer._activate(self, x, predict)

        def activate(self, x, w, bias=None, predict=False):
            return self.LayerTiming.timeit(level=1, name="activate", cls_name=name, prefix="[Core] ")(
                _activate)(self, x, predict)

        @property
        def params(self):
            return sub_layer.get_params(self)

        for key, value in locals().items():
            if str(value).find("function") >= 0 or str(value).find("property"):
                attr[key] = value

        return type(name, bases, attr)


class ConvLayerMeta(ABCMeta, ConvMeta):
    pass


class ConvSubLayerMeta(ABCMeta, ConvSubMeta):
    pass


# Activation Layers

class Tanh(Layer):
    def _activate(self, x, predict):
        return tf.tanh(x)


class Sigmoid(Layer):
    def _activate(self, x, predict):
        return 1 / (1 + tf.exp(-x))


class ELU(Layer):
    def _activate(self, x, predict):
        return tf.nn.elu(x)


class ReLU(Layer):
    def _activate(self, x, predict):
        return tf.maximum(x, 0)


class Softplus(Layer):
    def _activate(self, x, predict):
        return tf.log(1 + tf.exp(x))


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
        pool_height, pool_width = self._shape[1][1:]
        return tf.nn.max_pool(
            x, ksize=[1, pool_height, pool_width, 1],
            strides=[1, self._stride, self._stride, 1], padding="VALID")


# Special Layers

class Dropout(SubLayer):
    def __init__(self, parent, shape, drop_prob=None):
        if drop_prob < 0 or drop_prob >= 1:
            raise BuildLayerError("(Dropout) Probability of Dropout should be a positive float smaller than 1")
        SubLayer.__init__(self, parent, shape)
        if drop_prob is None:
            drop_prob = tf.constant(0.5, dtype=tf.float32)
        self._prob = 1 - drop_prob
        self._one = tf.constant(1, dtype=tf.float32)
        self.description = "(Drop prob: {})".format(drop_prob)

    def get_params(self):
        return 1 - self._prob,

    def _activate(self, x, predict):
        if not predict:
            return tf.nn.dropout(x, self._prob)
        return tf.nn.dropout(x, self._one)


class Normalize(SubLayer):
    def __init__(self, parent, shape, eps=1e-8, momentum=0.9):
        SubLayer.__init__(self, parent, shape)
        self._eps = eps
        self.rm = self.rv = None
        self.tf_rm = self.tf_rv = None
        self.tf_gamma = tf.Variable(tf.ones(self.shape[1]), name="norm_scale")
        self.tf_beta = tf.Variable(tf.zeros(self.shape[1]), name="norm_beta")
        self._momentum = momentum
        self.description = "(eps: {}, momentum: {})".format(eps, momentum)

    def init(self):
        if self.rm is not None:
            self.tf_rm = tf.Variable(self.rm, trainable=False, name="norm_mean")
        if self.rv is not None:
            self.tf_rv = tf.Variable(self.rv, trainable=False, name="norm_var")

    def get_special_params(self, sess):
        with sess.as_default():
            return {
                "rm": self.tf_rm.eval(), "rv": self.tf_rv.eval(),
            }

    def get_params(self):
        return self._eps, self._momentum

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
                return tf.nn.batch_normalization(x, _sm, _sv, self.tf_beta, self.tf_gamma, self._eps)
        return tf.nn.batch_normalization(x, self.tf_rm, self.tf_rv, self.tf_beta, self.tf_gamma, self._eps)


class ConvDrop(ConvLayer, Dropout, metaclass=ConvSubLayerMeta):
    pass


class ConvNorm(ConvLayer, Normalize, metaclass=ConvSubLayerMeta):
    pass


# Cost Layers

class CostLayer(Layer):
    def __init__(self, shape):
        Layer.__init__(self, shape)

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
        "MaxPool": MaxPool
    }
    available_special_layers = {
        "Dropout": Dropout,
        "Normalize": Normalize,
        "ConvDrop": ConvDrop,
        "ConvNorm": ConvNorm
    }
    special_layer_default_params = {
        "Dropout": (0.5,),
        "Normalize": (1e-8, 0.9),
        "ConvDrop": (0.5,),
        "ConvNorm": (1e-8, 0.9)
    }

    def handle_str_main_layers(self, name, *args, **kwargs):
        if name not in self.available_special_layers:
            if name in self.available_root_layers:
                name = self.available_root_layers[name]
            else:
                raise BuildNetworkError("Undefined layer '{}' found".format(name))
            return name(*args, **kwargs)
        return None

    def get_layer_by_name(self, name, parent, current_dimension, *args, **kwargs):
        _layer = self.handle_str_main_layers(name, *args, **kwargs)
        if _layer:
            return _layer, None
        _current, _next = parent.shape[1], current_dimension
        layer_param = self.special_layer_default_params[name]
        _layer = self.available_special_layers[name]
        if args or kwargs:
            _layer = _layer(parent, (_current, _next), *args, **kwargs)
        else:
            _layer = _layer(parent, (_current, _next), *layer_param)
        return _layer, (_current, _next)
