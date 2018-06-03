import os
import sys
root_path = os.path.abspath("../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import tensorflow as tf

from _Dist.NeuralNetworks.DistBase import Base,  AutoBase, AutoMeta, DistMixin, DistMeta


class LinearSVM(Base):
    def __init__(self, *args, **kwargs):
        super(LinearSVM, self).__init__(*args, **kwargs)
        self._name_appendix = "LinearSVM"
        self.c = None

    def init_from_data(self, x, y, x_test, y_test, sample_weights, names):
        super(LinearSVM, self).init_from_data(x, y, x_test, y_test, sample_weights, names)
        metric = self.model_param_settings.setdefault("metric", "binary_acc")
        if metric == "acc":
            self.model_param_settings["metric"] = "binary_acc"
        self.n_class = 1

    def init_model_param_settings(self):
        self.model_param_settings.setdefault("lr", 0.01)
        self.model_param_settings.setdefault("n_epoch", 10 ** 3)
        self.model_param_settings.setdefault("max_epoch", 10 ** 6)
        super(LinearSVM, self).init_model_param_settings()
        self.c = self.model_param_settings.get("C", 1.)

    def _build_model(self, net=None):
        self._model_built = True
        if net is None:
            net = self._tfx
        current_dimension = net.shape[1].value
        self._output = self._fully_connected_linear(
            net, [current_dimension, 1], "_final_projection"
        )

    def _define_loss_and_train_step(self):
        self._loss = self.c * tf.reduce_sum(
            tf.maximum(0., 1 - self._tfy * self._output)
        ) + tf.nn.l2_loss(self._ws[0])
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self._train_step = self._optimizer.minimize(self._loss)

    def _get_feed_dict(self, x, y=None, weights=None, is_training=False):
        if y is not None:
            y[y == 0] = -1
        return super(LinearSVM, self)._get_feed_dict(x, y, weights, is_training)

    def predict_classes(self, x):
        return (self._calculate(x, tensor=self._output, is_training=False) >= 0).astype(np.int32)


class SVM(LinearSVM):
    def __init__(self, *args, **kwargs):
        super(SVM, self).__init__(*args, **kwargs)
        self._name_appendix = "SVM"
        self._p = self._gamma = None
        self._x = self._gram = self._kernel_name = None

    @property
    def kernel(self):
        if self._kernel_name == "linear":
            return self.linear
        if self._kernel_name == "poly":
            return lambda x, y: self.poly(x, y, self._p)
        if self._kernel_name == "rbf":
            return lambda x, y: self.rbf(x, y, self._gamma)
        raise NotImplementedError("Kernel '{}' is not implemented".format(self._kernel_name))

    @staticmethod
    def linear(x, y):
        return x.dot(y.T)

    @staticmethod
    def poly(x, y, p):
        return (x.dot(y.T) + 1) ** p

    @staticmethod
    def rbf(x, y, gamma):
        return np.exp(-gamma * np.sum((x[..., None, :] - y) ** 2, axis=2))

    def init_from_data(self, x, y, x_test, y_test, sample_weights, names):
        self._x, y = np.atleast_2d(x).astype(np.float32), np.asarray(y, np.float32)
        self._p = self.model_param_settings.setdefault("p", 3)
        self._gamma = self.model_param_settings.setdefault("gamma", 1 / self._x.shape[1])
        self._kernel_name = self.model_param_settings.setdefault("kernel_name", "rbf")
        self._gram, x_test = self.kernel(self._x, self._x), self.kernel(x_test, self._x)
        super(SVM, self).init_from_data(self._gram, y, x_test, y_test, sample_weights, names)

    def init_model_param_settings(self):
        super(SVM, self).init_model_param_settings()
        self._p = self.model_param_settings["p"]
        self._gamma = self.model_param_settings["gamma"]
        self._kernel_name = self.model_param_settings["kernel_name"]

    def _define_py_collections(self):
        super(SVM, self)._define_py_collections()
        self.py_collections += ["_x", "_gram"]

    def _define_loss_and_train_step(self):
        self._loss = self.c * tf.reduce_sum(tf.maximum(0., 1 - self._tfy * self._output)) + 0.5 * tf.matmul(
            self._ws[0], tf.matmul(self._gram, self._ws[0]), transpose_a=True
        )[0]
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self._train_step = self._optimizer.minimize(self._loss)

    def _evaluate(self, x=None, y=None, x_cv=None, y_cv=None, x_test=None, y_test=None, metric=None):
        n_sample = self._x.shape[0]
        cv_feat_dim = None if x_cv is None else x_cv.shape[1]
        test_feat_dim = None if x_test is None else x_test.shape[1]
        x_cv = None if x_cv is None else self.kernel(x_cv, self._x) if cv_feat_dim != n_sample else x_cv
        x_test = None if x_test is None else self.kernel(x_test, self._x) if test_feat_dim != n_sample else x_test
        return super(SVM, self)._evaluate(x, y, x_cv, y_cv, x_test, y_test)

    def predict(self, x):
        # noinspection PyTypeChecker
        return self._predict(self.kernel(x, self._x))

    def predict_classes(self, x):
        return (self.predict(x) >= 0).astype(np.int32)

    def evaluate(self, x, y, x_cv=None, y_cv=None, x_test=None, y_test=None, metric=None):
        return self._evaluate(self.kernel(x, self._x), y, x_cv, y_cv, x_test, y_test, metric)


class AutoLinearSVM(AutoBase, LinearSVM, metaclass=AutoMeta):
    pass


class DistLinearSVM(AutoLinearSVM, DistMixin, metaclass=DistMeta):
    pass
