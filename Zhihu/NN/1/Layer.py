import tensorflow as tf
from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):
    def __init__(self, shape):
        self._shape = shape

    def activate(self, x, w, bias=None):
        if bias is None:
            return self._activate(tf.matmul(x, w))
        return self._activate(tf.matmul(x, w) + bias)

    @abstractmethod
    def _activate(self, x):
        pass


# Activation Layers

class Tanh(Layer):
    def _activate(self, x):
        return tf.tanh(x)


class Sigmoid(Layer):
    def _activate(self, x):
        return tf.nn.sigmoid(x)


class ELU(Layer):
    def _activate(self, x):
        return tf.nn.elu(x)


class ReLU(Layer):
    def _activate(self, x):
        return tf.nn.relu(x)


class Softplus(Layer):
    def _activate(self, x):
        return tf.nn.softplus(x)


class Identical(Layer):
    def _activate(self, x):
        return x


class CF0910(Layer):
    def _activate(self, x):
        return tf.minimum(tf.maximum(x, 0), 6)
