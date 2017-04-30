import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):
    def __init__(self, shape):
        self.shape = shape

    def activate(self, x, w, bias=None, predict=False):
        if bias is None:
            return self._activate(tf.matmul(x, w), predict)
        return self._activate(tf.matmul(x, w) + bias, predict)

    @abstractmethod
    def _activate(self, x, predict):
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


# Cost Layers

class CostLayer(Layer):
    def _activate(self, x, y):
        pass

    def calculate(self, y, y_pred):
        return self._activate(y.astype(np.float32), y_pred)


class CrossEntropy(CostLayer):
    def _activate(self, x, y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y))


class MSE(CostLayer):
    def _activate(self, x, y):
        return tf.reduce_mean(tf.square(x - y))
