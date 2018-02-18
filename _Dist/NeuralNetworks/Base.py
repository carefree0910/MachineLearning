import os
import sys
root_path = os.path.abspath("../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import time
import math
import random
import pickle
import shutil
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from _Dist.NeuralNetworks.NNUtil import *


class Generator:
    def __init__(self, x, y, name="Generator", weights=None, n_class=None, shuffle=True):
        self._cache = {}
        self._x, self._y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        if weights is None:
            self._sample_weights = None
        else:
            self._sample_weights = np.asarray(weights, np.float32)
        if n_class is not None:
            self.n_class = n_class
        else:
            y_int = self._y.astype(np.int32)
            if np.allclose(self._y, y_int):
                assert y_int.min() == 0, "Labels should start from 0"
                self.n_class = y_int.max() + 1
            else:
                self.n_class = 1
        self._name = name
        self._do_shuffle = shuffle
        self._all_valid_data = self._generate_all_valid_data()
        self._valid_indices = np.arange(len(self._all_valid_data))
        self._random_indices = self._valid_indices.copy()
        np.random.shuffle(self._random_indices)
        self._batch_cursor = -1

    def __enter__(self):
        self._cache_current_status()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._restore_cache()

    def __getitem__(self, item):
        return getattr(self, "_" + item)

    def __len__(self):
        return self.n_valid

    def __str__(self):
        return "{}_{}".format(self._name, self.shape)

    __repr__ = __str__

    @property
    def n_valid(self):
        return len(self._valid_indices)

    @property
    def n_dim(self):
        return self._x.shape[-1]

    @property
    def shape(self):
        return self.n_valid, self.n_dim

    def _generate_all_valid_data(self):
        return np.hstack([self._x, self._y.reshape([-1, 1])])

    def _cache_current_status(self):
        self._cache["_valid_indices"] = self._valid_indices
        self._cache["_random_indices"] = self._random_indices

    def _restore_cache(self):
        self._valid_indices = self._cache["_valid_indices"]
        self._random_indices = self._cache["_random_indices"]
        self._cache = {}

    def set_indices(self, indices):
        indices = np.asarray(indices, np.int)
        self._valid_indices = self._valid_indices[indices]
        self._random_indices = self._random_indices[indices]

    def set_range(self, start, end=None):
        if end is None:
            self._valid_indices = self._valid_indices[start:]
            self._random_indices = self._random_indices[start:]
        else:
            self._valid_indices = self._valid_indices[start:end]
            self._random_indices = self._random_indices[start:end]

    def get_indices(self, indices):
        return self._get_data(np.asarray(indices, np.int))

    def get_range(self, start, end=None):
        if end is None:
            return self._get_data(self._valid_indices[start:])
        return self._get_data(self._valid_indices[start:end])

    def _get_data(self, indices, return_weights=True):
        data = self._all_valid_data[indices]
        if not return_weights:
            return data
        weights = None if self._sample_weights is None else self._sample_weights[indices]
        return data, weights

    def gen_batch(self, n_batch, re_shuffle=True):
        n_batch = min(n_batch, self.n_valid)
        logger = logging.getLogger("DataReader")
        if n_batch == -1:
            n_batch = self.n_valid
        if self._batch_cursor < 0:
            self._batch_cursor = 0
        if self._do_shuffle:
            if self._batch_cursor == 0 and re_shuffle:
                logger.debug("Re-shuffling random indices")
                np.random.shuffle(self._random_indices)
            indices = self._random_indices
        else:
            indices = self._valid_indices
        logger.debug("Generating batch with size={}".format(n_batch))
        end = False
        next_cursor = self._batch_cursor + n_batch
        if next_cursor >= self.n_valid:
            next_cursor = self.n_valid
            end = True
        data, w = self._get_data(indices[self._batch_cursor:next_cursor])
        self._batch_cursor = -1 if end else next_cursor
        logger.debug("Done")
        return data, w

    def gen_random_subset(self, n):
        n = min(n, self.n_valid)
        logger = logging.getLogger("DataReader")
        logger.debug("Generating random subset with size={}".format(n))
        start = random.randint(0, self.n_valid - n)
        subset, weights = self._get_data(self._random_indices[start:start + n])
        logger.debug("Done")
        return subset, weights

    def get_all_data(self, return_weights=True):
        if self._all_valid_data is not None:
            if return_weights:
                return self._all_valid_data, self._sample_weights
            return self._all_valid_data
        return self._get_data(self._valid_indices, return_weights)


class Generator3d(Generator):
    @property
    def n_time_step(self):
        return self._x.shape[1]

    @property
    def shape(self):
        return self.n_valid, self.n_time_step, self.n_dim

    def _generate_all_valid_data(self):
        return np.array([(x, y) for x, y in zip(self._x, self._y)])


class Generator4d(Generator3d):
    @property
    def height(self):
        return self._x.shape[1]

    @property
    def width(self):
        return self._x.shape[2]

    @property
    def shape(self):
        return self.n_valid, self.height, self.width, self.n_dim


class Base:
    signature = "Base"

    def __init__(self, name=None, model_param_settings=None, model_structure_settings=None):
        self.log = {}
        self._name = name
        self._name_appendix = ""
        self._settings_initialized = False

        self._generator_base = Generator
        self._train_generator = self._test_generator = None
        self._sample_weights = self._tf_sample_weights = None
        self.n_dim = self.n_class = None
        self.n_random_train_subset = self.n_random_test_subset = None

        if model_param_settings is None:
            self.model_param_settings = {}
        else:
            assert_msg = "model_param_settings should be a dictionary"
            assert isinstance(model_param_settings, dict), assert_msg
            self.model_param_settings = model_param_settings
        self.lr = None
        self._loss = self._loss_name = self._metric_name = None
        self._optimizer_name = self._optimizer = None
        self.n_epoch = self.max_epoch = self.n_iter = self.batch_size = None

        if model_structure_settings is None:
            self.model_structure_settings = {}
        else:
            assert_msg = "model_structure_settings should be a dictionary"
            assert isinstance(model_structure_settings, dict), assert_msg
            self.model_structure_settings = model_structure_settings

        self._model_built = False
        self.py_collections = self.tf_collections = None
        self._define_py_collections()
        self._define_tf_collections()

        self._ws, self._bs = [], []
        self._is_training = None
        self._loss = self._train_step = None
        self._tfx = self._tfy = self._output = self._prob_output = None

        self._sess = None
        self._graph = tf.Graph()
        self._sess_config = self.model_param_settings.pop("sess_config", None)

    def __str__(self):
        return self.model_saving_name

    __repr__ = __str__

    @property
    def name(self):
        return "Base" if self._name is None else self._name

    @property
    def metric(self):
        return getattr(Metrics, self._metric_name)

    @property
    def model_saving_name(self):
        return "{}_{}".format(self.name, self._name_appendix)

    @property
    def model_saving_path(self):
        return os.path.join(os.getcwd(), "_Models", self.model_saving_name)

    # Settings

    def init_from_data(self, x, y, x_test, y_test, sample_weights, names):
        self._sample_weights = sample_weights
        if self._sample_weights is None:
            self._tf_sample_weights = None
        else:
            self._tf_sample_weights = tf.placeholder(tf.float32, name="sample_weights")

        self._train_generator = self._generator_base(x, y, "TrainGenerator", self._sample_weights, self.n_class)
        if x_test is not None and y_test is not None:
            self._test_generator = self._generator_base(x_test, y_test, "TestGenerator", n_class=self.n_class)
        else:
            self._test_generator = None
        self.n_random_train_subset = int(len(self._train_generator) * 0.1)
        if self._test_generator is None:
            self.n_random_test_subset = -1
        else:
            self.n_random_test_subset = len(self._test_generator)

        self.n_dim = self._train_generator.shape[-1]
        self.n_class = self._train_generator.n_class

        batch_size = self.model_param_settings.setdefault("batch_size", 128)
        self.model_param_settings["batch_size"] = min(batch_size, len(self._train_generator))
        n_iter = self.model_param_settings.setdefault("n_iter", -1)
        if n_iter < 0:
            self.model_param_settings["n_iter"] = int(len(self._train_generator) / batch_size)

    def init_all_settings(self):
        self.init_model_param_settings()
        self.init_model_structure_settings()

    def init_model_param_settings(self):
        loss = self.model_param_settings.get("loss", None)
        if loss is None:
            self._loss_name = "correlation" if self.n_class == 1 else "cross_entropy"
        else:
            self._loss_name = loss
        metric = self.model_param_settings.get("metric", None)
        if metric is None:
            if self.n_class == 1:
                self._metric_name = "correlation"
            elif self.n_class == 2:
                self._metric_name = "auc"
            else:
                self._metric_name = "multi_auc"
        else:
            self._metric_name = metric
        self.n_epoch = self.model_param_settings.get("n_epoch", 32)
        self.max_epoch = self.model_param_settings.get("max_epoch", 256)
        self.max_epoch = max(self.max_epoch, self.n_epoch)

        self.batch_size = self.model_param_settings["batch_size"]
        self.n_iter = self.model_param_settings["n_iter"]

        self._optimizer_name = self.model_param_settings.get("optimizer", "Adam")
        self.lr = self.model_param_settings.get("lr", 1e-3)
        self._optimizer = getattr(tf.train, "{}Optimizer".format(self._optimizer_name))(self.lr)

    def init_model_structure_settings(self):
        pass

    # Core

    def _fully_connected_linear(self, net, shape, appendix):
        with tf.name_scope("Linear{}".format(appendix)):
            w = init_w(shape, "W{}".format(appendix))
            b = init_b([shape[1]], "b{}".format(appendix))
            self._ws.append(w)
            self._bs.append(b)
            return tf.add(tf.matmul(net, w), b, name="Linear{}_Output".format(appendix))

    def _build_model(self, net=None):
        pass

    def _gen_batch(self, generator, n_batch, gen_random_subset=False, one_hot=False):
        if gen_random_subset:
            data, weights = generator.gen_random_subset(n_batch)
        else:
            data, weights = generator.gen_batch(n_batch)
        x, y = data[..., :-1], data[..., -1]
        if not one_hot:
            return x, y, weights
        if self.n_class == 1:
            y = y.reshape([-1, 1])
        else:
            y = Toolbox.get_one_hot(y, self.n_class)
        return x, y, weights

    def _get_feed_dict(self, x, y=None, weights=None, is_training=False):
        feed_dict = {self._tfx: x, self._is_training: is_training}
        if y is not None:
            feed_dict[self._tfy] = y
        if self._tf_sample_weights is not None:
            if weights is None:
                weights = np.ones(len(x))
            feed_dict[self._tf_sample_weights] = weights
        return feed_dict

    def _define_loss_and_train_step(self):
        self._loss = getattr(Losses, self._loss_name)(self._tfy, self._output, False, self._tf_sample_weights)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self._train_step = self._optimizer.minimize(self._loss)

    def _initialize_session(self):
        self._sess = tf.Session(graph=self._graph, config=self._sess_config)

    def _initialize_variables(self):
        self._sess.run(tf.global_variables_initializer())

    def _snapshot(self, i_epoch, i_iter, snapshot_cursor):
        x_train, y_train, sw_train = self._gen_batch(
            self._train_generator, self.n_random_train_subset,
            gen_random_subset=True
        )
        if self._test_generator is not None:
            x_test, y_test, sw_test = self._gen_batch(
                self._test_generator, self.n_random_test_subset,
                gen_random_subset=True
            )
            if self.n_class == 1:
                y_test = y_test.reshape([-1, 1])
            else:
                y_test = Toolbox.get_one_hot(y_test, self.n_class)
        else:
            x_test = y_test = sw_test = None
        y_train_pred = self._predict(x_train)
        if x_test is not None:
            tensor = self._output if self.n_class == 1 else self._prob_output
            y_test_pred, test_snapshot_loss = self._calculate(
                x_test, y_test, sw_test,
                [tensor, self._loss], is_training=False
            )
            y_test_pred, test_snapshot_loss = y_test_pred[0], test_snapshot_loss[0]
        else:
            y_test_pred = test_snapshot_loss = None
        train_metric = self.metric(y_train, y_train_pred)
        if y_test is not None and y_test_pred is not None:
            test_metric = self.metric(y_test, y_test_pred)
            if i_epoch >= 0 and i_iter >= 0 and snapshot_cursor >= 0:
                self.log["test_snapshot_loss"].append(test_snapshot_loss)
                self.log["test_{}".format(self._metric_name)].append(test_metric)
                self.log["train_{}".format(self._metric_name)].append(train_metric)
        else:
            test_metric = None
        print("\rEpoch {:6}   Iter {:8}   Snapshot {:6} ({})  -  Train : {:8.6f}   Test : {}".format(
            i_epoch, i_iter, snapshot_cursor, self._metric_name, train_metric,
            "None" if test_metric is None else "{:8.6f}".format(test_metric)
        ), end="")
        if i_epoch == i_iter == snapshot_cursor == 0:
            print()
        return train_metric, test_metric

    def _calculate(self, x, y=None, weights=None, tensor=None, n_elem=1e7, is_training=False):
        n_batch = int(n_elem / x.shape[1])
        n_repeat = int(len(x) / n_batch)
        if n_repeat * n_batch < len(x):
            n_repeat += 1
        cursors = [0]
        if tensor is None:
            target = self._prob_output
        elif isinstance(tensor, list):
            target = []
            for t in tensor:
                if isinstance(t, str):
                    t = getattr(self, t)
                if isinstance(t, list):
                    target += t
                    cursors.append(len(t))
                else:
                    target.append(t)
                    cursors.append(cursors[-1] + 1)
        else:
            target = getattr(self, tensor) if isinstance(tensor, str) else tensor
        results = [self._sess.run(
            target, self._get_feed_dict(
                x[i * n_batch:(i + 1) * n_batch],
                None if y is None else y[i * n_batch:(i + 1) * n_batch],
                None if weights is None else weights[i * n_batch:(i + 1) * n_batch],
                is_training=is_training
            )
        ) for i in range(n_repeat)]
        if not isinstance(target, list):
            if len(results) == 1:
                return results[0]
            return np.vstack(results)
        if n_repeat > 1:
            results = [
                np.vstack([result[i] for result in results]) if target[i].shape.ndims else
                np.mean([result[i] for result in results]) for i in range(len(target))
            ]
        else:
            results = results[0]
        if len(cursors) == 1:
            return results
        return [results[cursor:cursors[i + 1]] for i, cursor in enumerate(cursors[:-1])]

    def _predict(self, x):
        tensor = self._output if self.n_class == 1 else self._prob_output
        output = self._calculate(x, tensor=tensor, is_training=False)
        if self.n_class == 1:
            return output.ravel()
        return output

    def _evaluate(self, x=None, y=None, x_cv=None, y_cv=None, x_test=None, y_test=None, metric=None):
        if isinstance(metric, str):
            metric_name, metric = metric, getattr(Metrics, metric)
        else:
            metric_name, metric = self._metric_name, self.metric
        pred = self._predict(x) if x is not None else None
        cv_pred = self._predict(x_cv) if x_cv is not None else None
        test_pred = self._predict(x_test) if x_test is not None else None
        train_metric = None if y is None else metric(y, pred)
        cv_metric = None if y_cv is None else metric(y_cv, cv_pred)
        test_metric = None if y_test is None else metric(y_test, test_pred)
        self._print_metrics(metric_name, train_metric, cv_metric, test_metric)
        return train_metric, cv_metric, test_metric

    @staticmethod
    def _print_metrics(metric_name, train_metric=None, cv_metric=None, test_metric=None, only_return=False):
        msg = "{}  -  Train : {}   CV : {}   Test : {}".format(
            metric_name,
            "None" if train_metric is None else "{:10.8f}".format(train_metric),
            "None" if cv_metric is None else "{:10.8f}".format(cv_metric),
            "None" if test_metric is None else "{:10.8f}".format(test_metric)
        )
        return msg if only_return else print(msg)

    def _define_input_and_placeholder(self):
        self._is_training = tf.placeholder(tf.bool, name="is_training")
        self._tfx = tf.placeholder(tf.float32, [None, self.n_dim], name="X")
        self._tfy = tf.placeholder(tf.float32, [None, self.n_class], name="Y")

    def _define_py_collections(self):
        self.py_collections = [
            "_name", "n_class",
            "model_param_settings", "model_structure_settings"
        ]

    def _define_tf_collections(self):
        self.tf_collections = [
            "_tfx", "_tfy", "_output", "_prob_output",
            "_loss", "_train_step", "_is_training"
        ]

    # Save & Load

    def add_tf_collections(self):
        for tensor in self.tf_collections:
            target = getattr(self, tensor)
            if target is not None:
                tf.add_to_collection(tensor, target)

    def clear_tf_collections(self):
        for key in self.tf_collections:
            tf.get_collection_ref(key).clear()

    def save_collections(self, folder):
        with open(os.path.join(folder, "py.core"), "wb") as file:
            param_dict = {name: getattr(self, name) for name in self.py_collections}
            pickle.dump(param_dict, file)
        self.add_tf_collections()

    def restore_collections(self, folder):
        with open(os.path.join(folder, "py.core"), "rb") as file:
            param_dict = pickle.load(file)
            for name, value in param_dict.items():
                setattr(self, name, value)
        for tensor in self.tf_collections:
            target = tf.get_collection(tensor)
            if target is None:
                continue
            assert len(target) == 1, "{} available '{}' found".format(len(target), tensor)
            setattr(self, tensor, target[0])
        self.clear_tf_collections()

    @staticmethod
    def get_model_name(path, idx):
        targets = os.listdir(path)
        if idx is None:
            idx = max([int(target) for target in targets if target.isnumeric()])
        return os.path.join(path, "{:06}".format(idx))

    def save(self, run_id=0, path=None):
        if path is None:
            path = self.model_saving_path
        folder = os.path.join(path, "{:06}".format(run_id))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        print("Saving model")
        with self._graph.as_default():
            saver = tf.train.Saver()
            self.save_collections(folder)
            saver.save(self._sess, os.path.join(folder, "Model"))
            print("Model saved to " + folder)
            return self

    def load(self, run_id=None, clear_devices=False, path=None):
        self._model_built = True
        if path is None:
            path = self.model_saving_path
        folder = self.get_model_name(path, run_id)
        path = os.path.join(folder, "Model")
        print("Restoring model")
        with self._graph.as_default():
            if self._sess is None:
                self._initialize_session()
            saver = tf.train.import_meta_graph("{}.meta".format(path), clear_devices)
            saver.restore(self._sess, os.path.join(folder, "Model"))
            self.restore_collections(folder)
            self.init_all_settings()
            print("Model restored from " + folder)
            return self

    def save_checkpoint(self, folder):
        if not os.path.isdir(folder):
            os.makedirs(folder)
        with self._graph.as_default():
            tf.train.Saver().save(self._sess, os.path.join(folder, "Model"))

    def restore_checkpoint(self, folder):
        with self._graph.as_default():
            tf.train.Saver().restore(self._sess, os.path.join(folder, "Model"))

    # API

    def print_settings(self):
        pass

    def fit(self, x, y, x_test=None, y_test=None, sample_weights=None, names=("train", "test"),
            timeit=True, time_limit=-1, snapshot_ratio=3, print_settings=True, verbose=1):
        t = None
        if timeit:
            t = time.time()

        self.init_from_data(x, y, x_test, y_test, sample_weights, names)
        if not self._settings_initialized:
            self.init_all_settings()
            self._settings_initialized = True

        if not self._model_built:
            with self._graph.as_default():
                self._initialize_session()
                with tf.name_scope("Input"):
                    self._define_input_and_placeholder()
                with tf.name_scope("Model"):
                    self._build_model()
                    self._prob_output = tf.nn.softmax(self._output, name="Prob_Output")
                with tf.name_scope("LossAndTrainStep"):
                    self._define_loss_and_train_step()
                with tf.name_scope("InitializeVariables"):
                    self._initialize_variables()

        i_epoch = i_iter = j = snapshot_cursor = 0
        if snapshot_ratio == 0 or x_test is None or y_test is None:
            use_monitor = False
            snapshot_step = self.n_iter
        else:
            use_monitor = True
            snapshot_ratio = min(snapshot_ratio, self.n_iter)
            snapshot_step = int(self.n_iter / snapshot_ratio)

        terminate = False
        over_fitting_flag = 0
        n_epoch = self.n_epoch
        tmp_checkpoint_folder = os.path.join(self.model_saving_path, "tmp")
        if time_limit > 0:
            time_limit -= time.time() - t
            if time_limit <= 0:
                print("Time limit exceeded before training process started")
                return self
        monitor = TrainMonitor(Metrics.sign_dict[self._metric_name], snapshot_ratio)

        if verbose >= 2:
            prepare_tensorboard_verbose(self._sess)

        if print_settings:
            self.print_settings()

        self.log["iter_loss"] = []
        self.log["epoch_loss"] = []
        self.log["test_snapshot_loss"] = []
        self.log["train_{}".format(self._metric_name)] = []
        self.log["test_{}".format(self._metric_name)] = []
        self._snapshot(0, 0, 0)

        while i_epoch < n_epoch:
            i_epoch += 1
            epoch_loss = 0
            for j in range(self.n_iter):
                i_iter += 1
                x_batch, y_batch, sw_batch = self._gen_batch(self._train_generator, self.batch_size, one_hot=True)
                iter_loss = self._sess.run(
                    [self._loss, self._train_step],
                    self._get_feed_dict(x_batch, y_batch, sw_batch, is_training=True)
                )[0]
                self.log["iter_loss"].append(iter_loss)
                epoch_loss += iter_loss
                if i_iter % snapshot_step == 0 and verbose >= 1:
                    snapshot_cursor += 1
                    train_metric, test_metric = self._snapshot(i_epoch, i_iter, snapshot_cursor)
                    if use_monitor:
                        check_rs = monitor.check(test_metric)
                        over_fitting_flag = monitor.over_fitting_flag
                        if check_rs["terminate"]:
                            n_epoch = i_epoch
                            print("  -  Early stopped at n_epoch={} due to '{}'".format(
                                n_epoch, check_rs["info"]
                            ))
                            terminate = True
                            break
                        if check_rs["save_checkpoint"]:
                            print("  -  {}".format(check_rs["info"]))
                            self.save_checkpoint(tmp_checkpoint_folder)
                if 0 < time_limit <= time.time() - t:
                    print("  -  Early stopped at n_epoch={} "
                          "due to 'Time limit exceeded'".format(i_epoch))
                    terminate = True
                    break
            self.log["epoch_loss"].append(epoch_loss / (j + 1))
            if use_monitor:
                if i_epoch == n_epoch and i_epoch < self.max_epoch and not monitor.info["terminate"]:
                    monitor.flat_flag = True
                    monitor.punish_extension()
                    n_epoch = min(n_epoch + monitor.extension, self.max_epoch)
                    print("  -  Extending n_epoch to {}".format(n_epoch))
                if i_epoch == self.max_epoch:
                    terminate = True
                    if not monitor.info["terminate"]:
                        if not over_fitting_flag:
                            print(
                                "  -  Model seems to be under-fitting but max_epoch reached. "
                                "Increasing max_epoch may improve performance"
                            )
                        else:
                            print("  -  max_epoch reached")
            elif i_epoch == n_epoch:
                terminate = True
            if terminate:
                if os.path.isdir(tmp_checkpoint_folder):
                    print("  -  Rolling back to the best checkpoint")
                    self.restore_checkpoint(tmp_checkpoint_folder)
                    shutil.rmtree(tmp_checkpoint_folder)
                break
        self._snapshot(-1, -1, -1)

        if timeit:
            print("  -  Time Cost: {}".format(time.time() - t))

        return self

    def predict(self, x):
        return self._predict(x)

    def predict_classes(self, x):
        if self.n_class == 1:
            raise ValueError("Predicting classes is not permitted in regression problem")
        return self._predict(x).argmax(1).astype(np.int32)

    def evaluate(self, x, y, x_cv=None, y_cv=None, x_test=None, y_test=None, metric=None):
        return self._evaluate(x, y, x_cv, y_cv, x_test, y_test, metric)

    # Visualization

    def draw_losses(self):
        el, il = self.log["epoch_loss"], self.log["iter_loss"]
        ee_base = np.arange(len(el))
        ie_base = np.linspace(0, len(el) - 1, len(il))
        plt.figure()
        plt.plot(ie_base, il, label="Iter loss")
        plt.plot(ee_base, el, linewidth=3, label="Epoch loss")
        plt.legend()
        plt.show()
        return self

    def scatter2d(self, x, y, padding=0.5, title=None):
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        if labels.ndim == 1:
            plot_label_dict = {c: i for i, c in enumerate(set(labels))}
            n_label = len(plot_label_dict)
            labels = np.array([plot_label_dict[label] for label in labels])
        else:
            n_label = labels.shape[1]
            labels = np.argmax(labels, axis=1)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]

        if title is None:
            title = self.model_saving_name

        indices = [labels == i for i in range(np.max(labels) + 1)]
        scatters = []
        plt.figure()
        plt.title(title)
        for idx in indices:
            scatters.append(plt.scatter(axis[0][idx], axis[1][idx], c=colors[idx]))
        plt.legend(scatters, ["$c_{}$".format("{" + str(i) + "}") for i in range(len(scatters))],
                   ncol=math.ceil(math.sqrt(len(scatters))), fontsize=8)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()
        return self

    def scatter3d(self, x, y, padding=0.1, title=None):
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        z_min, z_max = np.min(axis[2]), np.max(axis[2])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        z_padding = max(abs(z_min), abs(z_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        z_min -= z_padding
        z_max += z_padding

        def transform_arr(arr):
            if arr.ndim == 1:
                dic = {c: i for i, c in enumerate(set(arr))}
                n_dim = len(dic)
                arr = np.array([dic[label] for label in arr])
            else:
                n_dim = arr.shape[1]
                arr = np.argmax(arr, axis=1)
            return arr, n_dim

        if title is None:
            title = self.model_saving_name

        labels, n_label = transform_arr(labels)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]
        indices = [labels == i for i in range(n_label)]
        scatters = []
        fig = plt.figure()
        plt.title(title)
        ax = fig.add_subplot(111, projection='3d')
        for _index in indices:
            scatters.append(ax.scatter(axis[0][_index], axis[1][_index], axis[2][_index], c=colors[_index]))
        ax.legend(scatters, ["$c_{}$".format("{" + str(i) + "}") for i in range(len(scatters))],
                  ncol=math.ceil(math.sqrt(len(scatters))), fontsize=8)
        plt.show()
        return self

    def visualize2d(self, x, y, padding=0.1, dense=200, title=None,
                    scatter=True, show_org=False, draw_background=True, emphasize=None, extra=None):
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))
        nx, ny, padding = dense, dense, padding
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        def get_base(_nx, _ny):
            _xf = np.linspace(x_min, x_max, _nx)
            _yf = np.linspace(y_min, y_max, _ny)
            n_xf, n_yf = np.meshgrid(_xf, _yf)
            return _xf, _yf, np.c_[n_xf.ravel(), n_yf.ravel()]

        xf, yf, base_matrix = get_base(nx, ny)

        t = time.time()
        z = self.predict_classes(base_matrix).reshape((nx, ny))
        print("Decision Time: {:8.6f} s".format(time.time() - t))

        print("Drawing figures...")
        xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
        if labels.ndim == 1:
            plot_label_dict = {c: i for i, c in enumerate(set(labels))}
            n_label = len(plot_label_dict)
            labels = np.array([plot_label_dict[label] for label in labels])
        else:
            n_label = labels.shape[1]
            labels = np.argmax(labels, axis=1)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]

        if title is None:
            title = self.model_saving_name

        if show_org:
            plt.figure()
            plt.scatter(axis[0], axis[1], c=colors)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.show()

        plt.figure()
        plt.title(title)
        if draw_background:
            plt.pcolormesh(xy_xf, xy_yf, z, cmap=plt.cm.Pastel1)
        else:
            plt.contour(xf, yf, z, c='k-', levels=[0])
        if scatter:
            plt.scatter(axis[0], axis[1], c=colors)
        if emphasize is not None:
            indices = np.array([False] * len(axis[0]))
            indices[np.asarray(emphasize)] = True
            plt.scatter(axis[0][indices], axis[1][indices], s=80,
                        facecolors="None", zorder=10)
        if extra is not None:
            plt.scatter(*np.asarray(extra).T, s=80, zorder=25, facecolors="red")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()
        print("Done.")
        return self

    def visualize3d(self, x, y, padding=0.1, dense=100, title=None,
                    show_org=False, draw_background=True, emphasize=None, extra=None):
        if False:
            print(Axes3D.add_artist)
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))

        nx, ny, nz, padding = dense, dense, dense, padding
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        z_min, z_max = np.min(axis[2]), np.max(axis[2])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        z_padding = max(abs(z_min), abs(z_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        z_min -= z_padding
        z_max += z_padding

        def get_base(_nx, _ny, _nz):
            _xf = np.linspace(x_min, x_max, _nx)
            _yf = np.linspace(y_min, y_max, _ny)
            _zf = np.linspace(z_min, z_max, _nz)
            n_xf, n_yf, n_zf = np.meshgrid(_xf, _yf, _zf)
            return _xf, _yf, _zf, np.c_[n_xf.ravel(), n_yf.ravel(), n_zf.ravel()]

        xf, yf, zf, base_matrix = get_base(nx, ny, nz)

        t = time.time()
        z_xyz = self.predict_classes(base_matrix).reshape((nx, ny, nz))
        p_classes = self.predict_classes(x).astype(np.int8)
        _, _, _, base_matrix = get_base(10, 10, 10)
        z_classes = self.predict_classes(base_matrix).astype(np.int8)
        print("Decision Time: {:8.6f} s".format(time.time() - t))

        print("Drawing figures...")
        z_xy = np.average(z_xyz, axis=2)
        z_yz = np.average(z_xyz, axis=1)
        z_xz = np.average(z_xyz, axis=0)

        xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
        yz_xf, yz_yf = np.meshgrid(yf, zf, sparse=True)
        xz_xf, xz_yf = np.meshgrid(xf, zf, sparse=True)

        def transform_arr(arr):
            if arr.ndim == 1:
                dic = {c: i for i, c in enumerate(set(arr))}
                n_dim = len(dic)
                arr = np.array([dic[label] for label in arr])
            else:
                n_dim = arr.shape[1]
                arr = np.argmax(arr, axis=1)
            return arr, n_dim

        labels, n_label = transform_arr(labels)
        p_classes, _ = transform_arr(p_classes)
        z_classes, _ = transform_arr(z_classes)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])
        if extra is not None:
            ex0, ex1, ex2 = np.asarray(extra).T
        else:
            ex0 = ex1 = ex2 = None

        if title is None:
            title = self.model_saving_name

        if show_org:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(axis[0], axis[1], axis[2], c=colors[labels])
            plt.show()

        fig = plt.figure(figsize=(16, 4), dpi=100)
        plt.title(title)
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        ax1.set_title("Org")
        ax2.set_title("Pred")
        ax3.set_title("Boundary")

        ax1.scatter(axis[0], axis[1], axis[2], c=colors[labels])
        ax2.scatter(axis[0], axis[1], axis[2], c=colors[p_classes], s=15)
        if extra is not None:
            ax2.scatter(ex0, ex1, ex2, s=80, zorder=25, facecolors="red")
        xyz_xf, xyz_yf, xyz_zf = base_matrix[..., 0], base_matrix[..., 1], base_matrix[..., 2]
        ax3.scatter(xyz_xf, xyz_yf, xyz_zf, c=colors[z_classes], s=15)

        plt.show()
        plt.close()

        fig = plt.figure(figsize=(16, 4), dpi=100)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        def _draw(_ax, _x, _xf, _y, _yf, _z):
            if draw_background:
                _ax.pcolormesh(_x, _y, _z > 0, cmap=plt.cm.Pastel1)
            else:
                _ax.contour(_xf, _yf, _z, c='k-', levels=[0])

        def _emphasize(_ax, axis0, axis1, _c):
            _ax.scatter(axis0, axis1, c=_c)
            if emphasize is not None:
                indices = np.array([False] * len(axis[0]))
                indices[np.asarray(emphasize)] = True
                _ax.scatter(axis0[indices], axis1[indices], s=80,
                            facecolors="None", zorder=10)

        def _extra(_ax, axis0, axis1, _c, _ex0, _ex1):
            _emphasize(_ax, axis0, axis1, _c)
            if extra is not None:
                _ax.scatter(_ex0, _ex1, s=80, zorder=25, facecolors="red")

        colors = colors[labels]

        ax1.set_title("xy figure")
        _draw(ax1, xy_xf, xf, xy_yf, yf, z_xy)
        _extra(ax1, axis[0], axis[1], colors, ex0, ex1)

        ax2.set_title("yz figure")
        _draw(ax2, yz_xf, yf, yz_yf, zf, z_yz)
        _extra(ax2, axis[1], axis[2], colors, ex1, ex2)

        ax3.set_title("xz figure")
        _draw(ax3, xz_xf, xf, xz_yf, zf, z_xz)
        _extra(ax3, axis[0], axis[2], colors, ex0, ex2)

        plt.show()
        print("Done.")
        return self


class AutoBase:
    # noinspection PyUnusedLocal
    def __init__(self, name=None, data_info=None, pre_process_settings=None, nan_handler_settings=None,
                 *args, **kwargs):
        if name is None:
            raise ValueError("name should be provided when using AutoBase")
        self._name = name

        self._data_folder = None
        self.whether_redundant = None
        self.feature_sets = self.sparsity = self.class_prior = None
        self.n_features = self.all_num_idx = self.transform_dicts = None

        self.py_collections = []

        if data_info is None:
            data_info = {}
        else:
            assert_msg = "data_info should be a dictionary"
            assert isinstance(data_info, dict), assert_msg
        self.data_info = data_info
        self._data_info_initialized = False
        self.numerical_idx = self.categorical_columns = None

        if pre_process_settings is None:
            pre_process_settings = {}
        else:
            assert_msg = "pre_process_settings should be a dictionary"
            assert isinstance(pre_process_settings, dict), assert_msg
        self.pre_process_settings = pre_process_settings
        self._pre_processors = None
        self.pre_process_method = self.scale_method = self.reuse_mean_and_std = None

        if nan_handler_settings is None:
            nan_handler_settings = {}
        else:
            assert_msg = "nan_handler_settings should be a dictionary"
            assert isinstance(nan_handler_settings, dict), assert_msg
        self.nan_handler_settings = nan_handler_settings
        self._nan_handler = None
        self.nan_handler_method = self.reuse_nan_handler_values = None

        self.init_pre_process_settings()
        self.init_nan_handler_settings()

    @property
    def label2num_dict(self):
        return None if not self.transform_dicts[-1] else self.transform_dicts[-1]

    @property
    def num2label_dict(self):
        label2num_dict = self.label2num_dict
        if label2num_dict is None:
            return
        num_label_list = sorted([(i, c) for c, i in label2num_dict.items()])
        return np.array([label for _, label in num_label_list])

    @property
    def valid_numerical_idx(self):
        return np.array([
            is_numerical for is_numerical in self.numerical_idx
            if is_numerical is not None
        ])

    @property
    def valid_n_features(self):
        return np.array([
            n_feature for i, n_feature in enumerate(self.n_features)
            if self.numerical_idx[i] is not None
        ])

    def init_data_info(self):
        if self._data_info_initialized:
            return
        self._data_info_initialized = True
        self.numerical_idx = self.data_info.get("numerical_idx", None)
        self.categorical_columns = self.data_info.get("categorical_columns", None)
        self.feature_sets = self.data_info.get("feature_sets", None)
        self.sparsity = self.data_info.get("sparsity", None)
        self.class_prior = self.data_info.get("class_prior", None)
        if self.feature_sets is not None and self.numerical_idx is not None:
            self.n_features = [len(feature_set) for feature_set in self.feature_sets]
            self._gen_categorical_columns()
        self._data_folder = self.data_info.get("data_folder", "_Data")
        self.data_info.setdefault("file_type", "txt")
        self.data_info.setdefault("shuffle", True)
        self.data_info.setdefault("test_rate", 0.1)
        self.data_info.setdefault("stage", 3)

    def init_pre_process_settings(self):
        self.pre_process_method = self.pre_process_settings.setdefault("pre_process_method", "normalize")
        self.scale_method = self.pre_process_settings.setdefault("scale_method", "truncate")
        self.reuse_mean_and_std = self.pre_process_settings.setdefault("reuse_mean_and_std", False)
        if self.pre_process_method is not None and self._pre_processors is None:
            self._pre_processors = {}

    def init_nan_handler_settings(self):
        self.nan_handler_method = self.nan_handler_settings.setdefault("nan_handler_method", "median")
        self.reuse_nan_handler_values = self.nan_handler_settings.setdefault("reuse_nan_handler_values", True)

    def _auto_init_from_data(self, x, y, x_test, y_test, names):
        stage = self.data_info["stage"]
        shuffle = self.data_info["shuffle"]
        file_type = self.data_info["file_type"]
        test_rate = self.data_info["test_rate"]
        args = (self.numerical_idx, file_type, names, shuffle, test_rate, stage)
        if x is None or y is None:
            x, y, x_test, y_test = self._load_data(None, *args)
        else:
            data = np.hstack([x, y.reshape([-1, 1])])
            if x_test is not None and y_test is not None:
                data = (data, np.hstack([x_test, y_test.reshape([-1, 1])]))
            x, y, x_test, y_test = self._load_data(data, *args)
        self._handle_unbalance(y)
        self._handle_sparsity()
        return x, y, x_test, y_test

    def _handle_unbalance(self, y):
        if self.n_class == 1:
            return
        class_ratio = self.class_prior.min() / self.class_prior.max()
        if class_ratio < 0.1:
            warn_msg = "Sample weights will be used since class_ratio < 0.1 ({:8.6f})".format(class_ratio)
            print(warn_msg)
            if self._sample_weights is None:
                print("Sample weights are not provided, they'll be generated automatically")
                self._sample_weights = np.ones(len(y)) / self.class_prior[y.astype(np.int)]
                self._sample_weights /= self._sample_weights.sum()
                self._sample_weights *= len(y)

    def _handle_sparsity(self):
        if self.sparsity >= 0.75:
            warn_msg = "Dropout will be disabled since data sparsity >= 0.75 ({:8.6f})".format(self.sparsity)
            print(warn_msg)
            self.dropout_keep_prob = 1.

    def _gen_categorical_columns(self):
        self.categorical_columns = [
            (i, value) for i, value in enumerate(self.valid_n_features)
            if not self.valid_numerical_idx[i] and self.valid_numerical_idx[i] is not None
        ]
        if not self.valid_numerical_idx[-1]:
            self.categorical_columns.pop()

    def _transform_data(self, data, name, train_name="train",
                        include_label=False, refresh_redundant_info=False, stage=3):
        print("Transforming {0}data{2} at stage {1}".format(
            "{} ".format(name), stage,
            "" if name == train_name or not self.reuse_mean_and_std else
            " with {} data".format(train_name),
        ))
        is_ndarray = isinstance(data, np.ndarray)
        if refresh_redundant_info or self.whether_redundant is None:
            self.whether_redundant = np.array([
                True if local_dict is None else False
                for local_dict in self.transform_dicts
            ])
        targets = [
            (i, local_dict) for i, (idx, local_dict) in enumerate(
                zip(self.numerical_idx, self.transform_dicts)
            ) if not idx and local_dict and not self.whether_redundant[i]
        ]
        if targets and targets[-1][0] == len(self.numerical_idx) - 1 and not include_label:
            targets = targets[:-1]
        if stage == 1 or stage == 3:
            # Transform data & Handle redundant
            n_redundant = np.sum(self.whether_redundant)
            if n_redundant == 0:
                whether_redundant = None
            else:
                whether_redundant = self.whether_redundant
                if not include_label:
                    whether_redundant = whether_redundant[:-1]
                if refresh_redundant_info:
                    warn_msg = "{} redundant: {}{}".format(
                        "These {} columns are".format(n_redundant) if n_redundant > 1 else "One column is",
                        [i for i, redundant in enumerate(whether_redundant) if redundant],
                        ", {} will be removed".format("it" if n_redundant == 1 else "they")
                    )
                    print(warn_msg)
            valid_indices = [
                i for i, redundant in enumerate(self.whether_redundant)
                if not redundant
            ]
            if not include_label:
                valid_indices = valid_indices[:-1]
            for i, line in enumerate(data):
                for j, local_dict in targets:
                    elem = line[j]
                    if isinstance(elem, str):
                        line[j] = local_dict.get(elem, local_dict.get("nan", len(local_dict)))
                    elif math.isnan(elem):
                        line[j] = local_dict["nan"]
                    else:
                        line[j] = local_dict.get(elem, local_dict.get("nan", len(local_dict)))
                if not is_ndarray and whether_redundant is not None:
                    data[i] = [line[j] for j in valid_indices]
            if is_ndarray and whether_redundant is not None:
                data = data[..., valid_indices].astype(np.float32)
            else:
                data = np.array(data, dtype=np.float32)
        if stage == 2 or stage == 3:
            data = np.asarray(data, dtype=np.float32)
            # Handle nan
            if self._nan_handler is None:
                self._nan_handler = NanHandler(
                    method=self.nan_handler_method,
                    reuse_values=self.reuse_nan_handler_values
                )
            data = self._nan_handler.transform(data, self.valid_numerical_idx[:-1])
            # Pre-process data
            if self._pre_processors is not None:
                pre_processor_name = train_name if self.reuse_mean_and_std else name
                pre_processor = self._pre_processors.setdefault(
                    pre_processor_name, PreProcessor(
                        self.pre_process_method, self.scale_method
                    )
                )
                if not include_label:
                    data = pre_processor.transform(data, self.valid_numerical_idx[:-1])
                else:
                    data[..., :-1] = pre_processor.transform(data[..., :-1], self.valid_numerical_idx[:-1])
        return data

    def _get_label_dict(self):
        labels = self.feature_sets[-1]
        sorted_labels = sorted(labels)
        if not all(Toolbox.is_number(str(label)) for label in labels):
            return {key: i for i, key in enumerate(sorted_labels)}
        if not sorted_labels:
            return {}
        numerical_labels = np.array(sorted_labels, np.float32)
        if numerical_labels.max() - numerical_labels.min() != self.n_class - 1:
            return {key: i for i, key in enumerate(sorted_labels)}
        return {}

    def _get_transform_dicts(self):
        self.transform_dicts = [
            None if is_numerical is None else
            {key: i for i, key in enumerate(sorted(feature_set))}
            if not is_numerical and (not all_num or not np.allclose(
                np.sort(np.array(list(feature_set), np.float32).astype(np.int32)),
                np.arange(0, len(feature_set))
            )) else {} for is_numerical, feature_set, all_num in zip(
                self.numerical_idx[:-1], self.feature_sets[:-1], self.all_num_idx[:-1]
            )
        ]
        if self.n_class == 1:
            self.transform_dicts.append({})
        else:
            self.transform_dicts.append(self._get_label_dict())

    def _get_data_from_file(self, file_type, test_rate, target=None):
        if file_type == "txt":
            sep, include_header = " ", False
        elif file_type == "csv":
            sep, include_header = ",", True
        else:
            raise NotImplementedError("File type '{}' not recognized".format(file_type))
        if target is None:
            target = os.path.join(self._data_folder, self._name)
        if not os.path.isdir(target):
            with open(target + ".{}".format(file_type), "r") as file:
                data = Toolbox.get_data(file, sep, include_header)
        else:
            with open(os.path.join(target, "train.{}".format(file_type)), "r") as file:
                train_data = Toolbox.get_data(file, sep, include_header)
            test_rate = 0
            test_file = os.path.join(target, "test.{}".format(file_type))
            if not os.path.isfile(test_file):
                data = train_data
            else:
                with open(test_file, "r") as file:
                    test_data = Toolbox.get_data(file, sep, include_header)
                data = (train_data, test_data)
        return data, test_rate

    def _load_data(self, data=None, numerical_idx=None, file_type="txt", names=("train", "test"),
                   shuffle=True, test_rate=0.1, stage=3):
        use_cached_data = False
        train_data = test_data = None
        data_cache_folder = os.path.join(self._data_folder, "_Cache", self._name)
        data_info_folder = os.path.join(self._data_folder, "_DataInfo")
        data_info_file = os.path.join(data_info_folder, "{}.info".format(self._name))
        train_data_file = os.path.join(data_cache_folder, "train.npy")
        test_data_file = os.path.join(data_cache_folder, "test.npy")

        if data is None and stage >= 2 and os.path.isfile(train_data_file):
            print("Restoring data")
            use_cached_data = True
            train_data = np.load(train_data_file)
            if not os.path.isfile(test_data_file):
                test_data = None
                data = train_data
            else:
                test_data = np.load(test_data_file)
                data = (train_data, test_data)
        if use_cached_data:
            n_train = None
        else:
            if data is None:
                is_ndarray = False
                data, test_rate = self._get_data_from_file(file_type, test_rate)
            else:
                is_ndarray = True
                if not isinstance(data, tuple):
                    test_rate = 0
                    data = np.asarray(data, dtype=np.float32)
                else:
                    data = tuple(
                        arr if isinstance(arr, list) else
                        np.asarray(arr, np.float32) for arr in data
                    )
            if isinstance(data, tuple):
                if shuffle:
                    np.random.shuffle(data[0]) if is_ndarray else random.shuffle(data[0])
                n_train = len(data[0])
                data = np.vstack(data) if is_ndarray else data[0] + data[1]
            else:
                if shuffle:
                    np.random.shuffle(data) if is_ndarray else random.shuffle(data)
                n_train = int(len(data) * (1 - test_rate)) if test_rate > 0 else -1

        if not os.path.isdir(data_info_folder):
            os.makedirs(data_info_folder)
        if not os.path.isfile(data_info_file) or stage == 1:
            print("Generating data info")
            if numerical_idx is not None:
                self.numerical_idx = numerical_idx
            elif self.numerical_idx is not None:
                numerical_idx = self.numerical_idx
            if not self.feature_sets or not self.n_features or not self.all_num_idx:
                is_regression = self.data_info.pop(
                    "is_regression",
                    numerical_idx is not None and numerical_idx[-1]
                )
                self.feature_sets, self.n_features, self.all_num_idx, self.numerical_idx = (
                    Toolbox.get_feature_info(data, numerical_idx, is_regression)
                )
            self.n_class = 1 if self.numerical_idx[-1] else self.n_features[-1]
            self._get_transform_dicts()
            with open(data_info_file, "wb") as file:
                pickle.dump([
                    self.n_features, self.numerical_idx, self.transform_dicts
                ], file)
        elif stage == 3:
            print("Restoring data info")
            with open(data_info_file, "rb") as file:
                info = pickle.load(file)
                self.n_features, self.numerical_idx, self.transform_dicts = info
            self.n_class = 1 if self.numerical_idx[-1] else self.n_features[-1]

        if not use_cached_data:
            if n_train > 0:
                train_data, test_data = data[:n_train], data[n_train:]
            else:
                train_data, test_data = data, None
            train_name, test_name = names
            train_data = self._transform_data(train_data, train_name, train_name, True, True, stage)
            if test_data is not None:
                test_data = self._transform_data(test_data, test_name, train_name, True, stage=stage)
        self._gen_categorical_columns()
        if not use_cached_data and stage == 3:
            print("Caching data...")
            if not os.path.isdir(data_cache_folder):
                os.makedirs(data_cache_folder)
            np.save(train_data_file, train_data)
            if test_data is not None:
                np.save(test_data_file, test_data)

        x, y = train_data[..., :-1], train_data[..., -1]
        if test_data is not None:
            x_test, y_test = test_data[..., :-1], test_data[..., -1]
        else:
            x_test = y_test = None
        self.sparsity = ((x == 0).sum() + np.isnan(x).sum()) / np.prod(x.shape)
        _, class_counts = np.unique(y, return_counts=True)
        self.class_prior = class_counts / class_counts.sum()

        self.data_info["numerical_idx"] = self.numerical_idx
        self.data_info["categorical_columns"] = self.categorical_columns

        return x, y, x_test, y_test

    def _pop_preprocessor(self, name):
        if isinstance(self._pre_processors, dict) and name in self._pre_processors:
            self._pre_processors.pop(name)

    def get_transformed_data_from_file(self, file, file_type="txt", include_label=False):
        x, _ = self._get_data_from_file(file_type, 0, file)
        x = self._transform_data(x, "new", include_label=include_label)
        self._pop_preprocessor("new")
        return x

    def get_labels_from_classes(self, classes):
        num2label_dict = self.num2label_dict
        if num2label_dict is None:
            return classes
        return num2label_dict[classes]

    def predict_labels(self, x):
        return self.get_labels_from_classes(self.predict_classes(x))

    # Signatures

    def fit(self, x=None, y=None, x_test=None, y_test=None, sample_weights=None, names=("train", "test"),
            timeit=True, time_limit=-1, snapshot_ratio=3, print_settings=True, verbose=1):
        raise ValueError

    def predict_classes(self, x):
        raise ValueError

    def predict_from_file(self, file, file_type="txt", include_label=False):
        raise ValueError

    def predict_classes_from_file(self, file, file_type="txt", include_label=False):
        raise ValueError

    def predict_labels_from_file(self, file, file_type="txt", include_label=False):
        raise ValueError

    def evaluate_from_file(self, file, file_type="txt"):
        raise ValueError


class AutoMeta(type):
    def __new__(mcs, *args, **kwargs):
        name_, bases, attr = args[:3]
        auto_base, model = bases

        def __init__(self, name=None, data_info=None, model_param_settings=None, model_structure_settings=None,
                     pre_process_settings=None, nan_handler_settings=None):
            auto_base.__init__(self, name, data_info, pre_process_settings, nan_handler_settings)
            if model.signature != "Advanced":
                model.__init__(self, name, model_param_settings, model_structure_settings)
            else:
                model.__init__(self, name, data_info, model_param_settings, model_structure_settings)

        def _define_py_collections(self):
            model._define_py_collections(self)
            self.py_collections += [
                "pre_process_settings", "nan_handler_settings",
                "_pre_processors", "_nan_handler", "transform_dicts",
                "numerical_idx", "categorical_columns", "transform_dicts"
            ]

        def init_data_info(self):
            auto_base.init_data_info(self)

        def init_from_data(self, x, y, x_test, y_test, sample_weights, names):
            self.init_data_info()
            x, y, x_test, y_test = self._auto_init_from_data(x, y, x_test, y_test, names)
            model.init_from_data(self, x, y, x_test, y_test, sample_weights, names)

        def fit(self, x=None, y=None, x_test=None, y_test=None, sample_weights=None, names=("train", "test"),
                timeit=True, time_limit=-1, snapshot_ratio=3, print_settings=True, verbose=1):
            return model.fit(
                self, x, y, x_test, y_test, sample_weights, names,
                timeit, time_limit, snapshot_ratio, print_settings, verbose
            )

        def predict(self, x):
            rs = self._predict(self._transform_data(x, "new", include_label=False))
            self._pop_preprocessor("new")
            return rs

        def predict_classes(self, x):
            if self.n_class == 1:
                raise ValueError("Predicting classes is not permitted in regression problem")
            return self.predict(x).argmax(1).astype(np.int32)

        def predict_target_prob(self, x, target):
            prob = self.predict(x)
            label2num_dict = self.label2num_dict
            if label2num_dict is not None:
                target = label2num_dict[target]
            return prob[..., target]

        def predict_from_file(self, file, file_type="txt", include_label=False):
            x = self.get_transformed_data_from_file(file, file_type, include_label)
            if include_label:
                x = x[..., :-1]
            return self._predict(x)

        def predict_classes_from_file(self, file, file_type="txt", include_label=False):
            if self.numerical_idx[-1]:
                raise ValueError("Predicting classes is not permitted in regression problem")
            x = self.get_transformed_data_from_file(file, file_type, include_label)
            if include_label:
                x = x[..., :-1]
            return self._predict(x).argmax(1).astype(np.int32)

        def predict_labels_from_file(self, file, file_type="txt", include_label=False):
            classes = self.predict_classes_from_file(file, file_type, include_label)
            return self.get_labels_from_classes(classes)

        def evaluate(self, x, y, x_cv=None, y_cv=None, x_test=None, y_test=None, metric=None):
            x = self._transform_data(x, "train")
            cv_name = "cv" if "cv" in self._pre_processors else "tmp_cv"
            test_name = "test" if "test" in self._pre_processors else "tmp_test"
            if x_cv is not None:
                x_cv = self._transform_data(x_cv, cv_name)
            if x_test is not None:
                x_test = self._transform_data(x_test, test_name)
            if cv_name == "tmp_cv":
                self._pop_preprocessor("tmp_cv")
            if test_name == "tmp_test":
                self._pop_preprocessor("tmp_test")
            return self._evaluate(x, y, x_cv, y_cv, x_test, y_test, metric)

        def print_settings(self):
            model.print_settings(self)
            msg = "NanHandler       : {}".format("None" if not self._nan_handler else "") + "\n"
            if self._nan_handler:
                msg += "\n".join("-> {:14}: {}".format(k, v) for k, v in sorted(
                    self.nan_handler_settings.items()
                )) + "\n"
            msg += "-" * 100 + "\n\n"
            msg += "PreProcessor     : {}".format("None" if not self._pre_processors else "") + "\n"
            if self._pre_processors:
                msg += "\n".join("-> {:14}: {}".format(k, v) for k, v in sorted(
                    self.pre_process_settings.items()
                )) + "\n"
            msg += "-" * 100 + "\n"
            print(msg)

        for key, value in locals().items():
            if str(value).find("function") >= 0:
                attr[key] = value

        return type(name_, bases, attr)
