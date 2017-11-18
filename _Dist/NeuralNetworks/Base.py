import os
import sys
root_path = os.path.abspath("../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import os
import time
import random
import pickle
import shutil
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from _Dist.NeuralNetworks.NNUtil import *


class Generator:
    def __init__(self, x, y, weights=None, name="Generator", shuffle=True):
        self._x, self._y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        if weights is None:
            self._sample_weights = None
        else:
            self._sample_weights = np.asarray(weights, np.float32)
        if len(self._y.shape) == 1:
            y_int = self._y.astype(np.int32)
            if np.allclose(self._y, y_int):
                assert y_int.min() == 0, "Labels should start from 0"
                self.n_class = y_int.max() + 1
            else:
                self.n_class = 1
        self._name = name
        self._do_shuffle = shuffle
        self._all_valid_data = np.hstack([self._x, self._y.reshape([-1, 1])])
        self._n_valid, self._n_dim = len(self._all_valid_data), self._x.shape[1]
        self._valid_indices = np.arange(len(self._all_valid_data))
        self._random_indices = self._valid_indices.copy()
        np.random.shuffle(self._random_indices)
        self._batch_cursor = -1

    def __getitem__(self, item):
        return getattr(self, "_" + item)

    def __len__(self):
        return self._n_valid

    def __str__(self):
        return "{}_{}".format(self._name, self.shape)

    __repr__ = __str__

    @property
    def shape(self):
        return self._n_valid, self._n_dim

    def _get_data(self, indices):
        weights = None if self._sample_weights is None else self._sample_weights[indices]
        return self._all_valid_data[indices], weights

    def _gen_batch_with_cache(self, logger, n_batch):
        logger.debug("Generating batch with cached data & size={}".format(n_batch))
        end = False
        next_cursor = self._batch_cursor + n_batch
        if next_cursor >= self._n_valid:
            next_cursor = self._n_valid
            end = True
        rs = self._all_valid_data[self._batch_cursor:next_cursor]
        if self._sample_weights is None:
            w = None
        else:
            w = self._sample_weights[self._batch_cursor:next_cursor]
        return rs, w, end, next_cursor

    def _gen_batch_without_cache(self, logger, n_batch, re_shuffle):
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
        if next_cursor >= self._n_valid:
            next_cursor = self._n_valid
            end = True
        rs, w = self._get_data(indices[self._batch_cursor:next_cursor])
        return rs, w, end, next_cursor

    def gen_batch(self, n_batch, re_shuffle=True):
        n_batch = min(n_batch, self._n_valid)
        logger = logging.getLogger("DataReader")
        if n_batch == -1:
            n_batch = self._n_valid
        if self._batch_cursor < 0:
            self._batch_cursor = 0
        if self._all_valid_data is None:
            rs, w, end, next_cursor = self._gen_batch_without_cache(logger, n_batch, re_shuffle)
        else:
            rs, w, end, next_cursor = self._gen_batch_with_cache(logger, n_batch)
        if end:
            self._batch_cursor = -1
        else:
            self._batch_cursor = next_cursor
        logger.debug("Done")
        return rs, w

    def gen_random_subset(self, n):
        n = min(n, self._n_valid)
        logger = logging.getLogger("DataReader")
        logger.debug("Generating random subset with size={}".format(n))
        start = random.randint(0, self._n_valid - n)
        if self._all_valid_data is None:
            subset, weights = self._get_data(self._random_indices[start:start + n])
        else:
            subset = self._all_valid_data[start:start + n]
            if self._sample_weights is None:
                weights = None
            else:
                weights = self._sample_weights[start:start + n]
        logger.debug("Done")
        return subset, weights

    def get_all_data(self):
        if self._all_valid_data is not None:
            return self._all_valid_data, self._sample_weights
        return self._get_data(self._valid_indices)

    def yield_all_data(self, n_batch):
        n_batch = min(n_batch, self._n_valid)
        logger = logging.getLogger("DataReader")
        logger.debug("Yielding all data with n_batch={}".format(n_batch))
        n_repeat = self._n_valid // n_batch
        if n_repeat * n_batch < self._n_valid:
            n_repeat += 1
        if self._all_valid_data is None:
            for i in range(n_repeat):
                yield self._get_data(self._valid_indices[i * n_batch:(i + 1) * n_batch])
        else:
            for i in range(n_repeat):
                if self._sample_weights is None:
                    weights = None
                else:
                    weights = self._sample_weights[i * n_batch:(i + 1) * n_batch]
                yield self._all_valid_data[i * n_batch:(i + 1) * n_batch], weights
        logger.debug("Done")


class Base:
    def __init__(self, x, y, x_test=None, y_test=None, name=None, loss=None, metric=None,
                 n_epoch=32, max_epoch=256, n_iter=128, batch_size=128, optimizer="Adam", lr=1e-3, **kwargs):
        tf.reset_default_graph()
        self.log = {}
        self._name = name
        self._kwargs = kwargs
        self._settings = ""

        self._sample_weights = kwargs.pop("sample_weights", None)
        if self._sample_weights is None:
            self._tf_sample_weights = None
        else:
            self._tf_sample_weights = tf.placeholder(tf.float32, name="sample_weights")

        self._train_generator = Generator(x, y, self._sample_weights)
        if x_test is not None and y_test is not None:
            self._test_generator = Generator(x_test, y_test)
        else:
            self._test_generator = None
        self.n_random_train_subset = int(len(self._train_generator) * 0.1)
        if self._test_generator is None:
            self.n_random_test_subset = -1
        else:
            self.n_random_test_subset = int(len(self._test_generator))

        self.n_dim = self._train_generator.shape[-1]
        self.n_class = self._train_generator.n_class

        self.n_epoch, self.max_epoch, self.n_iter = n_epoch, max_epoch, n_iter
        self.batch_size, self.lr = batch_size, lr

        if loss is None:
            self._loss_name = "correlation" if self.n_class == 1 else "cross_entropy"
        else:
            self._loss_name = loss

        if metric is None:
            if self.n_class == 1:
                self._metric, self._metric_name = Metrics.correlation, "correlation"
            else:
                self._metric, self._metric_name = Metrics.acc, "acc"
        else:
            self._metric, self._metric_name = getattr(Metrics, metric), metric

        self._model_built = False
        self.py_collections = None
        self.tf_collections = ["_tfx", "_tfy", "_output", "_n_batch_placeholder", "_is_training"]
        self._define_py_collections()

        self._ws, self._bs = [], []
        self._loss = self._train_step = None
        self._tfx = self._tfy = self._output = self._prob_output = None

        self._is_training = tf.placeholder(tf.bool, name="is_training")

        self._sess = tf.Session()
        self._optimizer_name = optimizer
        self._optimizer = getattr(tf.train, "{}Optimizer".format(optimizer))(lr)

    def __str__(self):
        return self.model_saving_name

    __repr__ = __str__

    @property
    def name(self):
        return "Base" if self._name is None else self._name

    @property
    def model_saving_name(self):
        return "{}_{}".format(self.name, self._settings)

    @property
    def model_saving_path(self):
        return os.path.join("_Models", self.model_saving_name)

    # Core

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

    def _define_py_collections(self):
        pass

    def _define_input(self):
        self._tfx = tf.placeholder(tf.float32, [None, self.n_dim], name="X")
        self._tfy = tf.placeholder(tf.float32, [None, self.n_class], name="Y")

    def _fully_connected_linear(self, net, shape, appendix):
        with tf.name_scope("Linear{}".format(appendix)):
            w = init_w(shape, "W{}".format(appendix))
            b = init_b([shape[1]], "b{}".format(appendix))
            self._ws.append(w)
            self._bs.append(b)
            return tf.add(tf.matmul(net, w), b, name="Linear{}_Output".format(appendix))

    def _build_model(self, net=None):
        pass

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

    def _initialize(self):
        self._sess.run(tf.global_variables_initializer())

    def _snapshot(self, i_epoch, i_iter, snapshot_cursor):
        x_train, y_train, _ = self._gen_batch(
            self._train_generator, self.n_random_train_subset, gen_random_subset=True
        )
        if self._test_generator is not None:
            x_test, y_test, sw_test = self._gen_batch(
                self._test_generator, self.n_random_test_subset, gen_random_subset=True
            )
        else:
            x_test = y_test = sw_test = None
        y_train_pred = self._predict(x_train)
        if x_test is not None:
            y_test_pred, test_snapshot_loss = self._calculate(
                x_test, Toolbox.get_one_hot(y_test, self.n_class), sw_test,
                [self._output, self._loss], is_training=False
            )
            y_test_pred, test_snapshot_loss = y_test_pred[0], test_snapshot_loss[0]
            self.log["test_snapshot_loss"].append(test_snapshot_loss)
        else:
            y_test_pred = None
        train_metric = self._metric(y_train, y_train_pred)
        if y_test is not None and y_test_pred is not None:
            test_metric = self._metric(y_test, y_test_pred)
        else:
            test_metric = None
        print("\rEpoch {:4}   Iter {:8}   Snapshot {:6} ({})  -  Train : {:8.6}   Test : {:8.6}".format(
            i_epoch, i_iter, snapshot_cursor, self._metric_name, train_metric, test_metric
        ), end="")
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
        output = self._calculate(x, is_training=False)
        if self.n_class == 1:
            return output.ravel()
        return output

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
            if not target:
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
        if not os.path.exists(folder):
            os.makedirs(folder)
        print("Saving model")
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
        saver = tf.train.import_meta_graph("{}.meta".format(path), clear_devices)
        saver.restore(self._sess, tf.train.latest_checkpoint(folder))
        self.restore_collections(folder)
        print("Model restored from " + folder)
        return self

    def save_checkpoint(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        tf.train.Saver().save(self._sess, os.path.join(folder, "Model"))

    def restore_checkpoint(self, folder):
        tf.train.Saver().restore(self._sess, tf.train.latest_checkpoint(folder))

    # API

    def feed_weights(self, ws):
        for i, w in enumerate(ws):
            if w is not None:
                self._sess.run(self._ws[i].assign(w))

    def feed_biases(self, bs):
        for i, b in enumerate(bs):
            if b is not None:
                self._sess.run(self._bs[i].assign(b))

    def print_settings(self):
        pass

    def fit(self, timeit=True, snapshot_ratio=3, print_settings=True, verbose=1):
        t = None
        if timeit:
            t = time.time()

        if not self._model_built:
            with tf.name_scope("Input"):
                self._define_input()
            with tf.name_scope("Model"):
                self._build_model()
                self._prob_output = tf.nn.softmax(self._output, name="Prob_Output")
            with tf.name_scope("LossAndTrainStep"):
                self._define_loss_and_train_step()
            with tf.name_scope("Initialize"):
                self._initialize()

        i_epoch = i_iter = snapshot_cursor = 0
        if snapshot_ratio == 0:
            use_monitor = False
            snapshot_step = self.n_iter
        else:
            use_monitor = True
            snapshot_step = self.n_iter // snapshot_ratio

        terminate = False
        over_fitting_flag = 0
        n_epoch = self.n_epoch
        tmp_checkpoint_folder = os.path.join(self.model_saving_path, "tmp")
        monitor = TrainMonitor(Metrics.sign_dict[self._metric_name], snapshot_ratio).start_new_run()

        if verbose >= 2:
            prepare_tensorboard_verbose(self._sess)

        if print_settings:
            self.print_settings()

        self.log["iter_loss"] = []
        self.log["epoch_loss"] = []
        self.log["test_snapshot_loss"] = []

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
                    if use_monitor and test_metric is not None:
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
            self.log["epoch_loss"].append(epoch_loss / self.n_iter)
            if use_monitor:
                if i_epoch == n_epoch and i_epoch < self.max_epoch and not monitor.rs["terminate"]:
                    monitor.flat_flag = True
                    n_epoch = min(n_epoch + monitor.extension, self.max_epoch)
                    print("  -  Extending n_epoch to {}".format(n_epoch))
                if i_epoch == self.max_epoch:
                    terminate = True
                    if not monitor.rs["terminate"]:
                        print(
                            "  -  Model seems to be under-fitting but max_epoch reached. "
                            "Increasing max_epoch may improve performance."
                        )
            if terminate:
                if over_fitting_flag and os.path.exists(tmp_checkpoint_folder):
                    print("  -  Rolling back to the best checkpoint")
                    self.restore_checkpoint(tmp_checkpoint_folder)
                    shutil.rmtree(tmp_checkpoint_folder)
                break

        if timeit:
            print("  -  Time Cost: {}".format(time.time() - t))

        return self

    def predict(self, x):
        return self._predict(x)

    def predict_classes(self, x):
        if self.n_class == 1:
            raise ValueError("Predicting classes is not permitted in regression problem")
        return self._predict(x).argmax(1)

    def evaluate(self, x, y, x_cv=None, y_cv=None, x_test=None, y_test=None):
        pred = self.predict(x)
        cv_pred = self.predict(x_cv) if x_cv is not None else None
        test_pred = self.predict(x_test) if x_test is not None else None
        print("{}  -  Train : {:8.6}   CV : {}   Test : {}".format(
            self._metric_name, self._metric(y, pred),
            "None" if y_cv is None else "{:8.6}".format(self._metric(y_cv, cv_pred)),
            "None" if y_test is None else "{:8.6}".format(self._metric(y_test, test_pred))
        ))
        return self

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
