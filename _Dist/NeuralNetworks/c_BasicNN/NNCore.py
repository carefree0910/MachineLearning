import os
import math
import datetime
import numpy as np
import tensorflow as tf

import sys
sys.path.append("../../../")
from Util.ProgressBar import ProgressBar
from _Dist.NeuralNetworks.c_BasicNN.NNUtils import *


class NNCore:
    def __init__(self, numerical_idx, categorical_columns, n_classes,
                 model_param_settings=None, network_structure_settings=None, verbose_settings=None):
        tf.reset_default_graph()
        self.numerical_idx = [i for i, numerical in enumerate(numerical_idx) if numerical]
        if numerical_idx[-1]:
            self.numerical_idx.pop()
        self.categorical_columns = categorical_columns
        self.n_classes = n_classes
        self.train_data = self.test_data = None

        self.model_built = False
        self.settings_inited = False

        if model_param_settings is None:
            model_param_settings = {}
        else:
            assert_msg = "Model param settings must be a dictionary"
            assert isinstance(model_param_settings, dict), assert_msg
        self.model_param_settings = model_param_settings
        self.lb = self.lr = self.loss_name = self.n_epoch = self.n_batch = self.optimizer = None
        self.activation_names = None
        self.activation_collection = Activations()

        if network_structure_settings is None:
            network_structure_settings = {}
        else:
            assert_msg = "Network structure settings must be a dictionary"
            assert isinstance(network_structure_settings, dict), assert_msg
        self.network_structure_settings = network_structure_settings
        self.hidden_units = None
        self.use_dropout = self.dropout_keep_prob = self.use_batch_norm = None
        self.use_one_hot_for_deep = self.use_embedding_for_deep = None
        # Embedding
        self.embedding_size = self.embedding_initializer = self.embedding_params = None

        if verbose_settings is None:
            verbose_settings = {}
        else:
            assert_msg = "Network structure settings must be a dictionary"
            assert isinstance(network_structure_settings, dict), assert_msg
        self.verbose_settings = verbose_settings
        self.tensorboard_verbose = None
        self.snapshot_ratio = self.snapshot_step = None
        self.metric_name = self._metric_placeholder = None

        self._sess = tf.Session()
        self._tfx = self._tfy = None
        self._model_activations = None
        self._output = self._prob_output = self._indicators = self._loss = self._train_step = None
        self._one_hot = self._embedding = self._embedding_with_one_hot = None
        self._one_hot_concat = self._embedding_concat = self._embedding_with_one_hot_concat = None
        self._categorical_xs = self._deep_input = None
        self._embedding_tables = []
        self._central_bias = None
        self._w_names = []
        self._ws, self._bs, self._ws_abs, self._w_abs_means = [], [], [], []
        self._p_keep = self._is_training = None

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    @property
    def metric(self):
        return getattr(Metrics, self.metric_name)

    @property
    def is_regression(self):
        return self.n_classes == 1

    # Initialize

    def init_all_settings(self):
        self.settings_inited = True
        self.init_model_param_settings()
        self.init_network_structure_settings()
        self.init_verbose_settings()

    def init_model_param_settings(self):
        self.loss_name = self.model_param_settings.get("loss", "cross_entropy")
        activations = self.model_param_settings.get("activations", ["relu"])
        self.lr = self.model_param_settings.get("lr", 0.001)
        self.lb = self.model_param_settings.get("lb", 0.)
        self.n_epoch = self.model_param_settings.get("n_epoch", 20)
        self.n_batch = self.model_param_settings.get("n_batch", 128)
        self.optimizer = self.model_param_settings.get("optimizer", "Adam")
        # Sanity check
        if not isinstance(activations, list):
            assert isinstance(activations, str), "Activations should be either a list or a string"
            self.activation_names = [activations]
        else:
            self.activation_names = activations

    def init_network_structure_settings(self):
        # Deep
        self.use_one_hot_for_deep = self.network_structure_settings.get("use_one_hot_for_deep", False)
        self.use_embedding_for_deep = self.network_structure_settings.get("use_embedding_for_deep", True)
        self.hidden_units = self.network_structure_settings.get("hidden_units", None)
        self.use_dropout = self.network_structure_settings.get("use_dropout", True)
        self.dropout_keep_prob = self.network_structure_settings.get("dropout_keep_prob", 0.5)
        self.use_batch_norm = self.network_structure_settings.get("use_batch_norm", False)
        # Embeddings
        self.embedding_size = self.network_structure_settings.get("embedding_size", 8)
        self.embedding_initializer = self.network_structure_settings.get(
            "embedding_initializer", "truncated_normal"
        )
        self.embedding_params = self.network_structure_settings.get(
            "embedding_params", (0, 0.02)
        )
        # Sanity check
        if self.is_regression:
            if self.loss_name == "cross_entropy":
                self.loss_name = "mse"

    def init_verbose_settings(self):
        self.metric_name = self.verbose_settings.get("metric", None)
        self.snapshot_ratio = self.verbose_settings.get("snapshot_ratio", 3)
        self.snapshot_step = self.verbose_settings.get("snapshot_step", None)
        self.tensorboard_verbose = self.verbose_settings.get("tensorboard_verbose", 1)
        # Sanity check
        if self.metric_name is None:
            if self.is_regression:
                self.metric_name = "mse"
            else:
                self.metric_name = "auc" if self.n_classes == 2 else "multi_auc"

    # Util

    def prepare_data(self, x, y, x_test, y_test):
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        x_test, y_test = np.asarray(x_test, np.float32), np.asarray(y_test, np.float32)
        self.train_data = np.hstack([x, y.reshape([-1, 1])])
        self.test_data = np.hstack([x_test, y_test.reshape([-1, 1])])
        if not self.is_regression:
            y, y_test = Toolbox.get_one_hot(y, self.n_classes), Toolbox.get_one_hot(y_test, self.n_classes)
        self.init_with_data(x)
        return x, y, x_test, y_test

    def get_feed_dict(self, data, include_label=True, predict=False, count=None):
        feed_dict = {}
        self._update_special_cases(feed_dict, count)
        if predict:
            feed_dict.update({self._p_keep: 1, self._is_training: False})
        else:
            feed_dict.update({self._p_keep: self.dropout_keep_prob, self._is_training: True})
        if self.use_embedding_for_deep or self._categorical_xs:
            feed_dict.update({self._tfx: data[..., self.numerical_idx]})
        if not self.use_embedding_for_deep:
            if include_label:
                feed_dict.update({self._deep_input: data[..., :-1]})
            else:
                feed_dict.update({self._deep_input: data})
        for (idx, _), embedding in zip(self.categorical_columns, self._categorical_xs):
            feed_dict.update({embedding: data[..., idx].astype(np.int)})
        if include_label:
            y = data[..., -1]
            y = Toolbox.get_one_hot(y, self.n_classes) if not self.is_regression else y.reshape([-1, 1])
            feed_dict.update({self._tfy: y})
        return feed_dict

    def gen_batches(self, data, shuffle=True, n_batch=None):
        n_batch = self.n_batch if n_batch is None else int(n_batch)
        n_repeat = len(data) // n_batch
        if len(data) > n_repeat * n_batch:
            n_repeat += 1
        if shuffle:
            data = data.copy()
            np.random.shuffle(data)
        for i in range(n_repeat):
            start = i * n_batch
            end = start + n_batch
            batch = data[start:end]
            yield batch

    def gen_dicts(self, data, n_batch=None, include_label=True, predict=False, add_noises=False,
                  shuffle=True, name=None, count=None):
        n_batch = self.n_batch if n_batch is None else int(n_batch)
        if name is not None:
            bar = ProgressBar(max_value=len(data) // n_batch, name=name)
        else:
            bar = None
        for batch in self.gen_batches(data, shuffle, n_batch):
            if bar is not None:
                bar.update()
            yield self.get_feed_dict(batch, include_label, predict, count=count)

    def get_embedding(self, i, n):
        embedding_size = math.ceil(math.log2(n)) + 1 if self.embedding_size == "log" else self.embedding_size
        embedding = tf.Variable(getattr(tf, self.embedding_initializer)(
            [n, embedding_size], *self.embedding_params
        ))
        self._embedding_tables.append(embedding)
        return tf.nn.embedding_lookup(embedding, self._categorical_xs[i], name="Embedded_X{}".format(i))

    def prepare_categorical_inputs(self):
        if not self.use_embedding_for_deep and not self.use_one_hot_for_deep:
            self._categorical_xs = []
        else:
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
                if self.numerical_idx:
                    self._one_hot_concat = tf.concat([self._tfx, self._one_hot], 1, name="Concat")
            with tf.name_scope("Embedding"):
                embeddings = [
                    self.get_embedding(i, n)
                    for i, (_, n) in enumerate(self.categorical_columns)
                ]
                self._embedding = self._embedding_concat = tf.concat(embeddings, 1, name="Raw")
                if self.numerical_idx:
                    self._embedding_concat = tf.concat([self._tfx, self._embedding], 1, name="Concat")
            with tf.name_scope("Embedding_with_one_hot"):
                self._embedding_with_one_hot = self._embedding_with_one_hot_concat = tf.concat(
                    embeddings + one_hot_vars, 1, name="Raw"
                )
                if self.numerical_idx:
                    self._embedding_with_one_hot_concat = tf.concat(
                        [self._tfx, self._embedding_with_one_hot], 1, name="Concat"
                    )

    # Build

    def init_w(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape, stddev=math.sqrt(2/sum(shape))), name=name)

    def init_b(self, shape, name):
        return tf.Variable(tf.zeros(shape), name=name)

    def init_placeholders(self):
        self._tfx = tf.placeholder("float", [None, len(self.numerical_idx)], name="Continuous_X")
        self._tfy = tf.placeholder("float", [None, self.n_classes], name="Y")
        self._p_keep = tf.placeholder("float", (), name="p_keep")
        self._is_training = tf.placeholder(tf.bool, (), name="is_training")
        self._metric_placeholder = tf.placeholder("float", (), name=self.metric_name)

    def init_hidden_units(self, current_units):
        if current_units > 512:
            self.hidden_units = [1024, 1024]
        else:
            self.hidden_units = [512, 512]

    def init_with_data(self, x):
        if self.snapshot_step is None:
            self.snapshot_step = (len(x) // self.n_batch) // self.snapshot_ratio

    def init_variables(self):
        self._sess.run(tf.global_variables_initializer())

    def feed_weights(self, ws):
        for i, w in enumerate(ws):
            if w is not None:
                self._sess.run(self._ws[i].assign(w))

    def feed_biases(self, bs):
        for i, b in enumerate(bs):
            if b is not None:
                self._sess.run(self._bs[i].assign(b))

    def fully_connected_linear(self, i, net, shape, bias=True):
        with tf.name_scope("Linear{}".format(i)):
            w_name = "W{}".format(i)
            self._w_names.append(w_name)
            w = self.init_w(shape, w_name)
            w_abs = tf.abs(w)
            w_abs_mean, w_abs_std = tf.nn.moments(w_abs, None, name="W_abs_mean{}".format(i))
            self._ws.append(w)
            self._ws_abs.append(w_abs)
            self._w_abs_means.append(w_abs_mean)
            if bias:
                b = self.init_b(shape[1], "b{}".format(i))
                self._bs.append(b)
                return tf.add(tf.matmul(net, w), b, name="Linear{}".format(i))
            return tf.matmul(net, w, name="Linear{}_without_bias".format(i))

    def build_layer(self, i, net, activation=None, use_dropout=None):
        if self.use_batch_norm:
            net = tf.layers.batch_normalization(net, training=self._is_training, name="Batch_Norm{}".format(i))
        if activation is None:
            activation = self.activation_names[0] if len(self.activation_names) == 1 else self.activation_names[i]
        net = self._activate(activation)(net, name="Activation{}_{}".format(i, activation))
        if use_dropout or (use_dropout is None and self.use_dropout):
            net = tf.nn.dropout(net, self._p_keep, name="Dropout{}".format(i))
        return net

    def build_deep_input(self):
        if not self.categorical_columns:
            self._deep_input = self._tfx
        elif self.use_one_hot_for_deep and self.use_embedding_for_deep:
            self._deep_input = self._embedding_with_one_hot_concat
        elif self.use_one_hot_for_deep:
            self._deep_input = self._one_hot_concat
        elif self.use_embedding_for_deep:
            self._deep_input = self._embedding_concat
        else:
            self._deep_input = tf.placeholder(
                "float", name="X",
                shape=[None, len(self.numerical_idx) + len(self.categorical_columns)]
            )
        net = self._deep_input
        current_units = net.get_shape().as_list()[1]
        if self.hidden_units is None:
            self.init_hidden_units(current_units)
        return net, current_units

    def build_deep(self, x, y, x_test, y_test):
        self._model_activations = []
        net, current_units = self.build_deep_input()
        with tf.name_scope("Deep_model"):
            for i, n_units in enumerate(self.hidden_units):
                net = self.fully_connected_linear(i, net, [current_units, n_units])
                net = self.build_layer(i, net)
                self._model_activations.append(net)
                current_units = n_units
            with tf.name_scope("DNN"):
                dnn_output = self.fully_connected_linear(
                    len(self.hidden_units), net,
                    [current_units, self.n_classes], bias=False
                )
        return dnn_output

    def build_wide(self):
        pass

    def build_loss(self):
        with tf.name_scope("Loss"):
            if self.lb > 0:
                with tf.name_scope("L2_loss"):
                    l2_loss = self.lb * tf.reduce_sum([tf.nn.l2_loss(w) for w in self._ws])
            else:
                l2_loss = None
            self._loss = getattr(Losses, self.loss_name)(self._tfy, self._indicators[-1], False)
            if l2_loss is not None:
                self._loss += l2_loss
        with tf.name_scope("Train_step"):
            self._train_step = getattr(tf.train, self.optimizer + "Optimizer")(self.lr).minimize(self._loss)

    def build_model(self, x, y, x_test, y_test, print_settings):
        # Placeholders
        self.init_placeholders()

        # Prepare net
        if self.categorical_columns:
            self.prepare_categorical_inputs()
        else:
            self._categorical_xs = []
            self._embedding_concat = self._one_hot_concat = self._tfx
        self._indicators = []
        self._central_bias = [tf.Variable(np.zeros(self.n_classes, dtype=np.float32), name="Central_bias")]

        # Deep
        deep_output = self.build_deep(x, y, x_test, y_test)

        # Wide
        wide_output = self.build_wide()

        # Output
        with tf.name_scope("Output"):
            self.build_output(deep_output, wide_output)
        self._indicators.append(self._output)
        self._model_activations.append(self._prob_output)

        # Loss
        self.build_loss()

        self.model_built = True
        if print_settings:
            self.print_settings()

        # Initialize
        self.init_variables()

    def build_output(self, deep_output, _):
        self._output = tf.add(deep_output, self._central_bias[0], name="Raw_output")
        self._prob_output = tf.nn.softmax(self._output, name="Prob_output")

    # Internal methods

    def _activate(self, activation):
        return getattr(self.activation_collection, activation)

    def _update_special_cases(self, dic, count):
        pass

    def _calculate(self, data, tensor, name, include_label=False, predict=True, n_elem=1e7, verbose=False):
        n_batch = n_elem // data.shape[1]
        if isinstance(tensor, list):
            return [self._calculate(data, t, name, include_label, predict, n_elem, verbose) for t in tensor]
        target = getattr(self, tensor) if isinstance(tensor, str) else tensor
        if not verbose:
            name = None
        if not isinstance(target, list):
            return np.vstack([self._sess.run(
                target, local_dict
            ) for local_dict in self.gen_dicts(
                data, n_batch, include_label=include_label, predict=predict, add_noises=False,
                shuffle=False, name=name
            )])
        results = [self._sess.run(
            target, local_dict
        ) for local_dict in self.gen_dicts(
            data, n_batch, include_label=include_label, predict=predict, add_noises=False,
            shuffle=False, name=name
        )]
        return [np.vstack([result[i] for result in results]) for i in range(len(target))]

    def _calculate_loss(self, train_losses, test_losses, return_pred=False):
        predictions = []
        for data, losses in zip([self.train_data, self.test_data], [train_losses, test_losses]):
            if data is not None:
                local_indicators, local_prob_pred = self._calculate(
                    data[..., :-1], ["_indicators", "_prob_output"], "Predict"
                )
                y = data[..., -1]
                if not self.is_regression:
                    y = Toolbox.get_one_hot(y, self.n_classes)
                else:
                    y = y.reshape([-1, 1])
                local_dict = {self._tfy: y}
                local_dict.update(dict(zip(self._indicators, local_indicators)))
                self._update_special_cases(local_dict, None)
                losses.append(self._sess.run(self._loss, local_dict))
                if return_pred:
                    predictions.append(local_prob_pred)
                else:
                    predictions.append(None)
            else:
                predictions.append(None)
        return predictions

    def _get_metrics(self, x, y, x_test, y_test):
        train_pred = self.predict(x, verbose=False)
        test_pred = self.predict(x_test, verbose=False) if x_test is not None else None
        train_metric = self.metric(y, train_pred)
        test_metric = self.metric(y_test, test_pred) if y_test is not None else None
        return train_metric, test_metric

    def _summary_var(self, var, abs_var, abs_mean, summary_sparsity):
        mean, std = tf.nn.moments(var, None)
        tf.summary.scalar("mean", mean)
        tf.summary.scalar("std", std)
        if self.tensorboard_verbose > 2:
            tf.summary.scalar("abs_mean", abs_mean)
        if self.tensorboard_verbose > 3:
            if summary_sparsity:
                tf.summary.scalar("sparsity", tf.reduce_mean(
                    tf.cast(abs_var < 1e-12, tf.float32)
                ))

    def _get_tb_collections(self):
        return [(self._ws, self._ws_abs, self._w_abs_means, self._w_names)]

    def _prepare_special_tb_verbose(self):
        pass

    def _prepare_tensorboard_verbose(self, sess):
        if self.tensorboard_verbose > 0:
            tb_log_folder = os.path.join(
                os.path.sep, "tmp", "tbLogs",
                str(datetime.datetime.now())[:19].replace(":", "-")
            )
            train_dir = os.path.join(tb_log_folder, "train")
            test_dir = os.path.join(tb_log_folder, "test")
            for tmp_dir in (train_dir, test_dir):
                if not os.path.isdir(tmp_dir):
                    os.makedirs(tmp_dir)
            if self.tensorboard_verbose > 1:
                collections = self._get_tb_collections()
                with tf.name_scope("VarSummaries"):
                    for weights, abs_weights, abs_means, names in collections:
                        for w, w_abs, w_abs_mean, name in zip(weights, abs_weights, abs_means, names):
                            with tf.name_scope(name):
                                self._summary_var(w, w_abs, w_abs_mean, "pruned" in name.lower())
            test_summary_ops = []
            with tf.name_scope("GlobalSummaries"):
                test_summary_ops.append(tf.summary.scalar("Loss", self._loss))
                test_summary_ops.append(tf.summary.scalar(self.metric_name, self._metric_placeholder))
            self._prepare_special_tb_verbose()
            train_merge_op = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(train_dir, sess.graph)
            test_writer = tf.summary.FileWriter(test_dir)
            test_merge_op = tf.summary.merge(test_summary_ops)
        else:
            train_writer = test_writer = train_merge_op = test_merge_op = None
        return train_writer, test_writer, train_merge_op, test_merge_op

    def _do_tensorboard_verbose(self, count, train_info, test_info, train_metric, test_metric):
        train_merge_op, train_losses, train_writer = train_info
        test_merge_op, test_losses, test_writer = test_info
        local_dict = {
            self._loss: train_losses[-1],
            self._metric_placeholder: train_metric
        }
        self._update_special_cases(local_dict, count)
        train_summary = self._sess.run(train_merge_op, local_dict)
        train_writer.add_summary(train_summary, count)
        if test_metric is not None:
            test_summary = self._sess.run(test_merge_op, {
                self._metric_placeholder: test_metric,
                self._loss: test_losses[-1]
            })
            test_writer.add_summary(test_summary, count)

    # API

    def print_settings(self):
        print("\n".join([
            "=" * 60, "This is a {}".format(
                "{}-classes problem".format(self.n_classes) if not self.is_regression
                else "regression problem"
            ), "-" * 60,
            "Data     : {} training samples, {} test samples".format(
                len(self.train_data), len(self.test_data) if self.test_data is not None else 0
            ),
            "Features : {} categorical, {} numerical".format(
                len(self.categorical_columns), len(self.numerical_idx)
            )
        ]))

        # Deep
        print("=" * 60)
        print("Deep model input: {}".format(
            "Continuous features only" if not self.categorical_columns
            else "Continuous features and raw categorical features"
            if not self.use_embedding_for_deep and not self.use_one_hot_for_deep
            else "One_hot encodings and embeddings" if self.use_one_hot_for_deep and self.use_embedding_for_deep
            else "One_hot encodings" if self.use_one_hot_for_deep
            else "Embeddings"
        ))
        print("-" * 60)
        if self.categorical_columns and self.use_embedding_for_deep:
            print("Embedding initializer: {}".format(self.embedding_initializer))
            print("Embedding params: {}".format(
                "mean={}, std={}".format(*self.embedding_params) if "normal" in self.embedding_initializer
                else "minval={}, maxval={}".format(*self.embedding_params)
            ))
            print("Embedding size: {}".format(self.embedding_size))
            print("Actual feature dimension: {}".format(self._embedding_concat.get_shape().as_list()[1]))
        elif not self.use_embedding_for_deep:
            print("Using raw values in categorical columns without embedding")
        print("-" * 60)
        if self.use_dropout:
            print("Using dropout with keep_prob = {}".format(self.dropout_keep_prob))
        else:
            print("Training without dropout")
        print("Training {} batch norm".format("with" if self.use_batch_norm else "without"))
        print("Hidden units: {}".format(self.hidden_units))

        # Params
        print("\n".join(["=" * 60, "Hyper parameters", "-" * 60]))
        if len(self.activation_names) == 1:
            activation_verbose = self.activation_names[0]
        else:
            activation_verbose = str(self.activation_names)
        print("Activation   : " + activation_verbose)
        print("Batch size   : " + str(self.n_batch))
        print("Epoch num    : " + str(self.n_epoch))
        print("Optimizer    : " + self.optimizer)
        print("Metric       : " + self.metric_name)
        print("Loss         : " + self.loss_name)
        print("lr           : " + str(self.lr))
        print("lb           : " + str(self.lb))
        print("-" * 60)

    def fit(self, x, y, x_test, y_test, n_epoch=None, n_batch=None, print_settings=True):
        if not self.settings_inited:
            self.init_all_settings()
        if n_epoch is not None:
            self.n_epoch = n_epoch
        if n_batch is not None:
            self.n_batch = n_batch
        x, y, x_test, y_test = self.prepare_data(x, y, x_test, y_test)
        if not self.model_built:
            self.build_model(x, y, x_test, y_test, print_settings)
        count = 0
        train_losses, test_losses = [], []
        with self._sess.as_default() as sess:
            # Prepare
            i = 0
            train_writer, test_writer, train_merge_op, test_merge_op = self._prepare_tensorboard_verbose(sess)
            bar = ProgressBar(max_value=self.n_epoch, name="Main")
            train_info = [train_merge_op, train_losses, train_writer]
            test_info = [test_merge_op, test_losses, test_writer]
            self._calculate_loss(train_losses, test_losses)
            train_metric, test_metric = self._get_metrics(x, y, x_test, y_test)
            if self.tensorboard_verbose > 0:
                self._do_tensorboard_verbose(count, train_info, test_info, train_metric, test_metric)
            # Train
            while i < self.n_epoch:
                for local_dict in self.gen_dicts(self.train_data, count=count):
                    count += 1
                    self._sess.run(self._train_step, local_dict)
                    if self.snapshot_step > 0 and count % self.snapshot_step == 0:
                        if self.tensorboard_verbose > 0:
                            train_metric, test_metric = self._get_metrics(x, y, x_test, y_test)
                            self._do_tensorboard_verbose(count, train_info, test_info, train_metric, test_metric)
                i += 1
                if self.tensorboard_verbose > 0:
                    if train_metric is None:
                        y_pred, y_test_pred = self._calculate_loss(train_losses, test_losses, return_pred=True)
                        train_metric = self.metric(y, y_pred)
                        test_metric = self.metric(y_test, y_test_pred) if y_test is not None else None
                    else:
                        self._calculate_loss(train_losses, test_losses)
                    self._do_tensorboard_verbose(count, train_info, test_info, train_metric, test_metric)
                else:
                    self._calculate_loss(train_losses, test_losses)
                if bar is not None:
                    bar.update()
        return train_losses, test_losses

    def predict(self, x, get_raw=False, verbose=True):
        tensor = "_output" if self.is_regression or get_raw else "_prob_output"
        return self._calculate(x, tensor, "Predict", verbose=verbose)

    def predict_classes(self, x, get_raw=False, verbose=True):
        pred = self.predict(x, get_raw, verbose)
        return np.argmax(pred, axis=1)

    def evaluate(self, x, y, metric_name=None, verbose=False):
        if metric_name is None:
            metric_name = self.metric_name
            metric = self.metric
        else:
            metric = getattr(Metrics, metric_name)
        if Metrics.require_prob[metric_name]:
            pred = self.predict(x, verbose=verbose)
        else:
            pred = self.predict_classes(x, verbose=verbose)
        score = metric(y, pred)
        print("{}: {:8.6f}".format(metric_name, score))
        return score
