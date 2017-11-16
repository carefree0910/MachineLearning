import os
import sys
root_path = os.path.abspath("../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import math
import numpy as np
import tensorflow as tf

from _Dist.NeuralNetworks.NNUtil import *
from _Dist.NeuralNetworks.c_BasicNN.NN import Basic


class Advanced(Basic):
    def __init__(self, *args, **kwargs):
        super(Advanced, self).__init__(*args, **kwargs)

        self.numerical_idx = self._kwargs.get("numerical_idx", None)
        if self.numerical_idx is None:
            raise ValueError("numerical_idx should be provided")
        if len(self.numerical_idx) != self.n_dim + 1:
            raise ValueError("Length of numerical_idx should be {}, {} found".format(
                self.n_dim + 1, len(self.numerical_idx)
            ))
        self.categorical_columns = self._kwargs.get("categorical_columns", None)
        if self.categorical_columns is None:
            raise ValueError("categorical_columns should be provided")

        self._deep_input = self._kwargs.get("deep_input", "embedding_concat")
        self._wide_input = self._kwargs.get("wide_input", "continuous")

        self.embedding_size = self._kwargs.get("embedding_size", 8)
        self._embedding = self._embedding_concat = None
        self._one_hot = self._one_hot_concat = None
        self._embedding_with_one_hot_concat = None

        self.dropout_keep_prob = self._kwargs.get("p_keep", 0.5)
        self.use_batch_norm = self._kwargs.get("use_batch_norm", False)

        self._tf_p_keep = tf.cond(
            self._is_training, lambda: self.dropout_keep_prob, lambda: 1.,
            name="p_keep"
        )
        self._n_batch_placeholder = tf.placeholder(tf.int32, name="n_batch")

        self._use_wide_network = self._kwargs.get("use_wide_network", True)
        self._dndf = DNDF(self.n_class) if self._kwargs.get("use_dndf", True) else None
        self._pruner = Pruner() if self._kwargs.get("use_pruner", True) else None

    @property
    def name(self):
        return "AdvancedNN" if self._name is None else self._name

    def _get_embedding(self, i, n):
        embedding_size = math.ceil(math.log2(n)) + 1 if self.embedding_size == "log" else self.embedding_size
        embedding = tf.get_variable("Embedding{}".format(i), [n, embedding_size])
        return tf.nn.embedding_lookup(embedding, self._categorical_xs[i], name="Embedded_X{}".format(i))

    def _define_input(self):
        super(Advanced, self)._define_input()
        if not self.categorical_columns:
            self._categorical_xs = []
            self._one_hot = self._one_hot_concat = self._tfx
            self._embedding = self._embedding_concat = self._tfx
            self._embedding_with_one_hot = self._embedding_with_one_hot_concat = self._tfx
        else:
            all_categorical = not np.any(self.numerical_idx)
            with tf.name_scope("Categorical_Xs"):
                self._categorical_xs = [
                    tf.placeholder(tf.int32, shape=[None], name="Categorical_X{}".format(i))
                    for i in range(len(self.categorical_columns))
                ]
            with tf.name_scope("One_hot"):
                one_hot_vars = [
                    tf.one_hot(self._categorical_xs[i], n)
                    for i, (_, n) in enumerate(self.categorical_columns)
                ]
                self._one_hot = self._one_hot_concat = tf.concat(one_hot_vars, 1, name="Raw")
                if not all_categorical:
                    self._one_hot_concat = tf.concat([self._tfx, self._one_hot], 1, name="Concat")
            with tf.name_scope("Embedding"):
                embeddings = [
                    self._get_embedding(i, n)
                    for i, (_, n) in enumerate(self.categorical_columns)
                ]
                self._embedding = self._embedding_concat = tf.concat(embeddings, 1, name="Raw")
                if not all_categorical:
                    self._embedding_concat = tf.concat([self._tfx, self._embedding], 1, name="Concat")
            with tf.name_scope("Embedding_with_one_hot"):
                self._embedding_with_one_hot = self._embedding_with_one_hot_concat = tf.concat(
                    embeddings + one_hot_vars, 1, name="Raw"
                )
                if not all_categorical:
                    self._embedding_with_one_hot_concat = tf.concat(
                        [self._tfx, self._embedding_with_one_hot], 1, name="Concat"
                    )
        if self._wide_input == "continuous":
            self._wide_input = self._tfx
        else:
            self._wide_input = getattr(self, "_" + self._wide_input)
        if self._deep_input == "continuous":
            self._deep_input = self._tfx
        else:
            self._deep_input = getattr(self, "_" + self._deep_input)

    def _fully_connected_linear(self, net, shape, appendix):
        w = init_w(shape, "W{}".format(appendix))
        if self._pruner is not None:
            w_abs = tf.abs(w)
            w_abs_mean, w_abs_var = tf.nn.moments(w_abs, None)
            w = self._pruner.prune_w(w, w_abs, w_abs_mean, tf.sqrt(w_abs_var))
        b = init_b([shape[1]], "b{}".format(appendix))
        self._ws.append(w)
        self._bs.append(b)
        return tf.nn.xw_plus_b(net, w, b, "linear{}".format(appendix))

    def _build_layer(self, i, net):
        if self.use_batch_norm:
            net = tf.layers.batch_normalization(net, training=self._is_training, name="BN{}".format(i))
        activation = self.activations[i]
        if activation is not None:
            net = getattr(Activations, activation)(net, "{}{}".format(activation, i))
        if self.dropout_keep_prob < 1:
            net = tf.nn.dropout(net, keep_prob=self._tf_p_keep)
        return net

    def _build_model(self):
        tfx = self._tfx
        self._tfx = self._deep_input
        super(Advanced, self)._build_model()
        self._tfx = tfx
        if self._use_wide_network:
            if self._dndf is None:
                wide_output = self._fully_connected_linear(
                    self._wide_input,
                    [self._wide_input.shape[1].value, self.n_class], "_wide_output"
                )
            else:
                wide_output = self._dndf(self._wide_input, self._n_batch_placeholder)
            self._output += wide_output

    def _get_feed_dict(self, x, y=None, is_training=True):
        feed_dict = super(Advanced, self)._get_feed_dict(x, y, is_training)
        if self._dndf is not None:
            feed_dict[self._n_batch_placeholder] = len(x)
        return feed_dict
