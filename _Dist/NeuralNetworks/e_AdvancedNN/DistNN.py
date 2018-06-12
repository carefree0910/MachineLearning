import os
import sys
root_path = os.path.abspath("../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import math
import numpy as np
import tensorflow as tf

from _Dist.NeuralNetworks.NNUtil import *
from _Dist.NeuralNetworks.c_BasicNN.DistNN import Basic


class Advanced(Basic):
    signature = "Advanced"

    def __init__(self, name=None, data_info=None, model_param_settings=None, model_structure_settings=None):
        self.tf_list_collections = None
        super(Advanced, self).__init__(name, model_param_settings, model_structure_settings)
        self._name_appendix = "Advanced"

        if data_info is None:
            self.data_info = {}
        else:
            assert_msg = "data_info should be a dictionary"
            assert isinstance(data_info, dict), assert_msg
            self.data_info = data_info
        self._data_info_initialized = False
        self.numerical_idx = self.categorical_columns = None

        self._deep_input = self._wide_input = None
        self._categorical_xs = None
        self.embedding_size = None
        self._embedding = self._one_hot = self._embedding_concat = self._one_hot_concat = None
        self._embedding_with_one_hot = self._embedding_with_one_hot_concat = None

        self.dropout_keep_prob = self.use_batch_norm = None
        self._use_wide_network = self._dndf = self._pruner = self._dndf_pruner = None

        self._tf_p_keep = None
        self._n_batch_placeholder = None

    @property
    def valid_numerical_idx(self):
        return np.array([
            is_numerical for is_numerical in self.numerical_idx
            if is_numerical is not None
        ])

    def init_data_info(self):
        if self._data_info_initialized:
            return
        self._data_info_initialized = True
        self.numerical_idx = self.data_info.get("numerical_idx", None)
        self.categorical_columns = self.data_info.get("categorical_columns", None)
        if self.numerical_idx is None:
            raise ValueError("numerical_idx should be provided")
        if self.categorical_columns is None:
            raise ValueError("categorical_columns should be provided")

    def init_from_data(self, x, y, x_test, y_test, sample_weights, names):
        self.init_data_info()
        super(Advanced, self).init_from_data(x, y, x_test, y_test, sample_weights, names)
        if len(self.valid_numerical_idx) != self.n_dim + 1:
            raise ValueError("Length of valid_numerical_idx should be {}, {} found".format(
                self.n_dim + 1, len(self.valid_numerical_idx)
            ))
        self.n_dim -= len(self.categorical_columns)
        self.model_structure_settings.setdefault("use_wide_network", self.n_dim > 0)

    def init_model_param_settings(self):
        super(Advanced, self).init_model_param_settings()
        self.dropout_keep_prob = float(self.model_param_settings.get("keep_prob", 0.5))
        self.use_batch_norm = self.model_param_settings.get("use_batch_norm", False)

    def init_model_structure_settings(self):
        self.hidden_units = self.model_structure_settings.get("hidden_units", None)
        self._deep_input = self.model_structure_settings.get("deep_input", "embedding_concat")
        self._wide_input = self.model_structure_settings.get("wide_input", "continuous")
        self.embedding_size = self.model_structure_settings.get("embedding_size", 8)

        self._use_wide_network = self.model_structure_settings["use_wide_network"]
        if not self._use_wide_network:
            self._dndf = None
        else:
            dndf_params = self.model_structure_settings.get("dndf_params", {})
            if self.model_structure_settings.get("use_dndf", True):
                self._dndf = DNDF(self.n_class, **dndf_params)
        if self.model_structure_settings.get("use_pruner", True):
            pruner_params = self.model_structure_settings.get("pruner_params", {})
            self._pruner = Pruner(**pruner_params)
        if self.model_structure_settings.get("use_dndf_pruner", False):
            dndf_pruner_params = self.model_structure_settings.get("dndf_pruner_params", {})
            self._dndf_pruner = Pruner(**dndf_pruner_params)

    def _get_embedding(self, i, n):
        embedding_size = math.ceil(math.log2(n)) + 1 if self.embedding_size == "log" else self.embedding_size
        embedding = tf.Variable(tf.truncated_normal(
            [n, embedding_size], mean=0, stddev=0.02
        ), name="Embedding{}".format(i))
        return tf.nn.embedding_lookup(embedding, self._categorical_xs[i], name="Embedded_X{}".format(i))

    def _define_hidden_units(self):
        n_data = len(self._train_generator)
        current_units = self._deep_input.shape[1].value
        if current_units > 512:
            self.hidden_units = [1024, 1024]
        elif current_units > 256:
            if n_data >= 10000:
                self.hidden_units = [1024, 1024]
            else:
                self.hidden_units = [2 * current_units, 2 * current_units]
        else:
            if n_data >= 100000:
                self.hidden_units = [768, 768]
            elif n_data >= 10000:
                self.hidden_units = [512, 512]
            else:
                self.hidden_units = [2 * current_units, 2 * current_units]

    def _fully_connected_linear(self, net, shape, appendix):
        with tf.name_scope("Linear{}".format(appendix)):
            w = init_w(shape, "W{}".format(appendix))
            if self._pruner is not None:
                w = self._pruner.prune_w(*self._pruner.get_w_info(w))
            b = init_b([shape[1]], "b{}".format(appendix))
            self._ws.append(w)
            self._bs.append(b)
            return tf.add(tf.matmul(net, w), b, name="Linear{}_Output".format(appendix))

    def _build_layer(self, i, net):
        if self.use_batch_norm:
            net = tf.layers.batch_normalization(net, training=self._is_training, name="BN{}".format(i))
        activation = self.activations[i]
        if activation is not None:
            net = getattr(Activations, activation)(net, "{}{}".format(activation, i))
        if self.dropout_keep_prob < 1:
            net = tf.nn.dropout(net, keep_prob=self._tf_p_keep)
        return net

    def _build_model(self, net=None):
        super(Advanced, self)._build_model(self._deep_input)
        if self._use_wide_network:
            if self._dndf is None:
                wide_output = self._fully_connected_linear(
                    self._wide_input, appendix="_wide_output",
                    shape=[self._wide_input.shape[1].value, self.n_class]
                )
            else:
                wide_output = self._dndf(
                    self._wide_input, self._n_batch_placeholder,
                    pruner=self._dndf_pruner
                )
            self._output += wide_output

    def _get_feed_dict(self, x, y=None, weights=None, is_training=True):
        continuous_x = x[..., self.valid_numerical_idx[:-1]] if self._categorical_xs else x
        feed_dict = super(Advanced, self)._get_feed_dict(continuous_x, y, weights, is_training)
        if self._dndf is not None:
            feed_dict[self._n_batch_placeholder] = len(x)
        if self._pruner is not None:
            cond_placeholder = self._pruner.cond_placeholder
            if cond_placeholder is not None:
                feed_dict[cond_placeholder] = True
        if self._dndf is not None and self._dndf_pruner is not None:
            cond_placeholder = self._dndf_pruner.cond_placeholder
            if cond_placeholder is not None:
                feed_dict[cond_placeholder] = True
        for (idx, _), categorical_x in zip(self.categorical_columns, self._categorical_xs):
            feed_dict.update({categorical_x: x[..., idx].astype(np.int32)})
        return feed_dict

    def _define_input_and_placeholder(self):
        super(Advanced, self)._define_input_and_placeholder()
        if not self.categorical_columns:
            self._categorical_xs = []
            self._one_hot = self._one_hot_concat = self._tfx
            self._embedding = self._embedding_concat = self._tfx
            self._embedding_with_one_hot = self._embedding_with_one_hot_concat = self._tfx
        else:
            all_categorical = self.n_dim == 0
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
        if self.hidden_units is None:
            self._define_hidden_units()
        self._tf_p_keep = tf.cond(
            self._is_training, lambda: self.dropout_keep_prob, lambda: 1.,
            name="keep_prob"
        )
        self._n_batch_placeholder = tf.placeholder(tf.int32, name="n_batch")

    def _define_py_collections(self):
        super(Advanced, self)._define_py_collections()
        self.py_collections += ["data_info", "numerical_idx", "categorical_columns"]

    def _define_tf_collections(self):
        super(Advanced, self)._define_tf_collections()
        self.tf_collections += [
            "_deep_input", "_wide_input", "_n_batch_placeholder",
            "_embedding", "_one_hot", "_embedding_with_one_hot",
            "_embedding_concat", "_one_hot_concat", "_embedding_with_one_hot_concat"
        ]
        self.tf_list_collections = ["_categorical_xs"]

    def add_tf_collections(self):
        super(Advanced, self).add_tf_collections()
        for tf_list in self.tf_list_collections:
            target_list = getattr(self, tf_list)
            if target_list is None:
                continue
            for tensor in target_list:
                tf.add_to_collection(tf_list, tensor)

    def restore_collections(self, folder):
        for tf_list in self.tf_list_collections:
            if tf_list is not None:
                setattr(self, tf_list, tf.get_collection(tf_list))
        super(Advanced, self).restore_collections(folder)

    def clear_tf_collections(self):
        super(Advanced, self).clear_tf_collections()
        for key in self.tf_list_collections:
            tf.get_collection_ref(key).clear()

    def print_settings(self, only_return=False):
        msg = "\n".join([
            "=" * 100, "This is a {}".format(
                "{}-classes problem".format(self.n_class) if not self.n_class == 1
                else "regression problem"
            ), "-" * 100,
            "Data     : {} training samples, {} test samples".format(
                len(self._train_generator), len(self._test_generator) if self._test_generator is not None else 0
            ),
            "Features : {} categorical, {} numerical".format(
                len(self.categorical_columns), np.sum(self.valid_numerical_idx)
            )
        ]) + "\n"

        msg += "=" * 100 + "\n"
        msg += "Deep model: DNN\n"
        msg += "Deep model input: {}\n".format(
            "Continuous features only" if not self.categorical_columns else
            "Continuous features with embeddings" if np.any(self.numerical_idx) else
            "Embeddings only"
        )
        msg += "-" * 100 + "\n"
        if self.categorical_columns:
            msg += "Embedding size: {}\n".format(self.embedding_size)
            msg += "Actual feature dimension: {}\n".format(self._embedding_concat.shape[1].value)
        msg += "-" * 100 + "\n"
        if self.dropout_keep_prob < 1:
            msg += "Using dropout with keep_prob = {}\n".format(self.dropout_keep_prob)
        else:
            msg += "Training without dropout\n"
        msg += "Training {} batch norm\n".format("with" if self.use_batch_norm else "without")
        msg += "Hidden units: {}\n".format(self.hidden_units)

        msg += "=" * 100 + "\n"
        if not self._use_wide_network:
            msg += "Wide model: None\n"
        else:
            msg += "Wide model: {}\n".format("logistic regression" if self._dndf is None else "DNDF")
            msg += "Wide model input: Continuous features only\n"
            msg += "-" * 100 + '\n'
            if self._dndf is not None:
                msg += "Using DNDF with n_tree = {}, tree_depth = {}\n".format(
                    self._dndf.n_tree, self._dndf.tree_depth
                )

        msg += "\n".join(["=" * 100, "Hyper parameters", "-" * 100, "{}".format(
            "This is a DNN model" if self._dndf is None and not self._use_wide_network else
            "This is a Wide & Deep model" if self._dndf is None else
            "This is a hybrid model"
        ), "-" * 100]) + "\n"
        msg += "Activation       : " + str(self.activations) + "\n"
        msg += "Batch size       : " + str(self.batch_size) + "\n"
        msg += "Epoch num        : " + str(self.n_epoch) + "\n"
        msg += "Optimizer        : " + self._optimizer_name + "\n"
        msg += "Metric           : " + self._metric_name + "\n"
        msg += "Loss             : " + self._loss_name + "\n"
        msg += "lr               : " + str(self.lr) + "\n"
        msg += "-" * 100 + "\n"
        msg += "Pruner           : {}".format("None" if self._pruner is None else "") + "\n"
        if self._pruner is not None:
            msg += "\n".join("-> {:14}: {}".format(key, value) for key, value in sorted(
                self._pruner.params.items()
            )) + "\n"
        msg += "-" * 100
        return msg if only_return else self.log_msg(
            "\n" + msg, logger=self.get_logger("print_settings", "general.log"))
