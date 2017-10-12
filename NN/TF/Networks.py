import os
import cv2
import time
import pickle
import datetime
import matplotlib.pyplot as plt
from math import floor, sqrt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph

from NN.TF.Layers import *

from Util.Util import Util, VisUtil
from Util.Bases import TFClassifierBase
from Util.ProgressBar import ProgressBar


# TODO: Saving NNPipe; Add 'idx' param to 'get_rs' method


class NNVerbose:
    NONE = 0
    EPOCH = 1
    ITER = 1.5
    METRICS = 2
    METRICS_DETAIL = 3
    DETAIL = 4
    DEBUG = 5


class NNConfig:
    BOOST_LESS_SAMPLES = False
    TRAINING_SCALE = 5 / 6
    BATCH_SIZE = 1e6


# Neural Network

class NNBase(TFClassifierBase):
    NNTiming = Timing()

    def __init__(self, **kwargs):
        super(NNBase, self).__init__(**kwargs)
        self._layers = []
        self._optimizer = None
        self._w_stds, self._b_inits = [], []
        self._layer_names, self._layer_params = [], []
        self._lr = 0
        self.verbose = 1
        self._current_dimension = 0

        self._logs = {}
        self._metrics, self._metric_names, self._metric_rs = [], [], []

        self._loaded = False
        self._x_min = self._x_max = self._y_min = self._y_max = 0
        self._transferred_flags = {"train": False, "test": False}

        self._activations = None
        self._loss = self._train_step = None
        self._layer_factory = LayerFactory()
        self._tf_weights, self._tf_bias = [], []

    @property
    def name(self):
        return (
            "-".join([str(_layer.shape[1]) for _layer in self._layers]) +
            " at {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        )

    @NNTiming.timeit(level=4)
    def _get_w(self, shape):
        if self._w_stds[-1] is None:
            self._w_stds[-1] = sqrt(2 / sum(shape))
        initial = tf.truncated_normal(shape, stddev=self._w_stds[-1])
        return tf.Variable(initial, name="w")

    @NNTiming.timeit(level=4)
    def _get_b(self, shape):
        return tf.Variable(np.zeros(shape, dtype=np.float32) + self._b_inits[-1], name="b")

    @NNTiming.timeit(level=4)
    def _get_tb_name(self, layer):
        return "{}_{}".format(layer.position, layer.name)

    @staticmethod
    @NNTiming.timeit(level=4)
    def _summary_var(var):
        with tf.name_scope("summaries"):
            mean = tf.reduce_mean(var)
            tf.summary.scalar("mean", mean)
            with tf.name_scope("std"):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar("std", stddev)

    # noinspection PyTypeChecker
    @NNTiming.timeit(level=4)
    def _add_params(self, layer, shape, conv_channel=None, fc_shape=None, apply_bias=True):
        if fc_shape is not None:
            w_shape = (fc_shape, shape[1])
            b_shape = shape[1],
        elif conv_channel is not None:
            if len(shape[1]) <= 2:
                w_shape = shape[1][0], shape[1][1], conv_channel, conv_channel
            else:
                w_shape = (shape[1][1], shape[1][2], conv_channel, shape[1][0])
            b_shape = shape[1][0],
        else:
            w_shape = shape
            b_shape = shape[1],
        _new_w = self._get_w(w_shape)
        _new_b = self._get_b(b_shape) if apply_bias else None
        self._tf_weights.append(_new_w)
        if apply_bias:
            self._tf_bias.append(_new_b)
        else:
            self._tf_bias.append(None)
        with tf.name_scope(self._get_tb_name(layer)):
            with tf.name_scope("weight"):
                NNBase._summary_var(_new_w)
            if layer.apply_bias:
                with tf.name_scope("bias"):
                    NNBase._summary_var(_new_b)

    @NNTiming.timeit(level=4)
    def _add_param_placeholder(self):
        self._tf_weights.append(tf.constant([.0]))
        self._tf_bias.append(tf.constant([.0]))

    @NNTiming.timeit(level=4)
    def _add_layer(self, layer, *args, **kwargs):
        if not self._layers and isinstance(layer, str):
            if layer.lower() == "pipe":
                self._layers.append(NNPipe(args[0]))
                self._add_param_placeholder()
                return
            _layer = self._layer_factory.get_root_layer_by_name(layer, *args, **kwargs)
            if _layer:
                self.add(_layer, pop_last_init=True)
                return
        _parent = self._layers[-1]
        if isinstance(_parent, CostLayer):
            raise BuildLayerError("Adding layer after CostLayer is not permitted")
        if isinstance(_parent, NNPipe):
            self._current_dimension = _parent.shape[1]
        if isinstance(layer, str):
            if layer.lower() == "pipe":
                self._layers.append(NNPipe(args[0]))
                self._add_param_placeholder()
                return
            layer, shape = self._layer_factory.get_layer_by_name(
                layer, _parent, self._current_dimension, *args, **kwargs
            )
            if shape is None:
                self.add(layer, pop_last_init=True)
                return
            _current, _next = shape
        else:
            _current, _next = args
        if isinstance(layer, SubLayer):
            if _current != _parent.shape[1]:
                raise BuildLayerError("Output shape should be identical with input shape "
                                      "if chosen SubLayer is not a CostLayer")
            self.parent = _parent
            self._layers.append(layer)
            self._add_param_placeholder()
            self._current_dimension = _next
        else:
            fc_shape, conv_channel, last_layer = None, None, self._layers[-1]
            if NNBase._is_conv(last_layer):
                if NNBase._is_conv(layer):
                    conv_channel = last_layer.n_filters
                    _current = (conv_channel, last_layer.out_h, last_layer.out_w)
                    layer.feed_shape((_current, _next))
                else:
                    layer.is_fc = True
                    fc_shape = last_layer.out_h * last_layer.out_w * last_layer.n_filters
            self._layers.append(layer)
            if isinstance(layer, ConvPoolLayer):
                self._add_param_placeholder()
            else:
                self._add_params(layer, (_current, _next), conv_channel, fc_shape)
            self._current_dimension = _next
        self._update_layer_information(layer)

    @NNTiming.timeit(level=4)
    def _update_layer_information(self, layer):
        self._layer_params.append(layer.params)
        if len(self._layer_params) > 1 and not layer.is_sub_layer:
            self._layer_params[-1] = ((self._layer_params[-1][0][1],), *self._layer_params[-1][1:])

    @staticmethod
    @NNTiming.timeit(level=4)
    def _is_conv(layer):
        return isinstance(layer, ConvLayer) or isinstance(layer, NNPipe)

    @NNTiming.timeit(level=1, prefix="[API] ")
    def get_rs(self, x, predict=True, pipe=False):
        if not isinstance(self._layers[0], NNPipe):
            cache = self._layers[0].activate(x, self._tf_weights[0], self._tf_bias[0], predict)
        else:
            cache = self._layers[0].get_rs(x, predict)
        for i, layer in enumerate(self._layers[1:]):
            if i == len(self._layers) - 2:
                if not pipe:
                    if NNDist._is_conv(self._layers[i]):
                        fc_shape = np.prod(cache.get_shape()[1:])  # type: int
                        cache = tf.reshape(cache, [-1, int(fc_shape)])
                    if self._tf_bias[-1] is not None:
                        return tf.matmul(cache, self._tf_weights[-1]) + self._tf_bias[-1]
                    return tf.matmul(cache, self._tf_weights[-1])
                else:
                    if not isinstance(layer, NNPipe):
                        return layer.activate(cache, self._tf_weights[i + 1], self._tf_bias[i + 1], predict)
                    return layer.get_rs(cache, predict)
            if not isinstance(layer, NNPipe):
                cache = layer.activate(cache, self._tf_weights[i + 1], self._tf_bias[i + 1], predict)
            else:
                cache = layer.get_rs(cache, predict)

    @NNTiming.timeit(level=4, prefix="[API] ")
    def add(self, layer, *args, **kwargs):
        # Init kwargs
        kwargs["apply_bias"] = kwargs.get("apply_bias", True)
        kwargs["position"] = kwargs.get("position", len(self._layers) + 1)

        self._w_stds.append(kwargs.pop("w_std", None))
        self._b_inits.append(kwargs.pop("b_init", 0.1))
        if kwargs.pop("pop_last_init", False):
            self._w_stds.pop()
            self._b_inits.pop()
        if isinstance(layer, str):
            # noinspection PyTypeChecker
            self._add_layer(layer, *args, **kwargs)
        else:
            if not isinstance(layer, Layer):
                raise BuildLayerError("Invalid Layer provided (should be subclass of Layer)")
            if not self._layers:
                if isinstance(layer, SubLayer):
                    raise BuildLayerError("Invalid Layer provided (first layer should not be subclass of SubLayer)")
                if len(layer.shape) != 2:
                    raise BuildLayerError("Invalid input Layer provided (shape should be {}, {} found)".format(
                        2, len(layer.shape)
                    ))
                self._layers, self._current_dimension = [layer], layer.shape[1]
                self._update_layer_information(layer)
                if isinstance(layer, ConvLayer):
                    self._add_params(layer, layer.shape, layer.n_channels)
                else:
                    self._add_params(layer, layer.shape)
            else:
                if len(layer.shape) > 2:
                    raise BuildLayerError("Invalid Layer provided (shape should be {}, {} found)".format(
                        2, len(layer.shape)
                    ))
                if len(layer.shape) == 2:
                    _current, _next = layer.shape
                    if isinstance(layer, SubLayer):
                        if _next != self._current_dimension:
                            raise BuildLayerError("Invalid SubLayer provided (shape[1] should be {}, {} found)".format(
                                self._current_dimension, _next
                            ))
                    elif not NNDist._is_conv(layer) and _current != self._current_dimension:
                        raise BuildLayerError("Invalid Layer provided (shape[0] should be {}, {} found)".format(
                            self._current_dimension, _current
                        ))
                    self._add_layer(layer, _current, _next)
                elif len(layer.shape) == 1:
                    _next = layer.shape[0]
                    layer.shape = (self._current_dimension, _next)
                    self._add_layer(layer, self._current_dimension, _next)
                else:
                    raise LayerError("Invalid Layer provided (invalid shape '{}' found)".format(layer.shape))

    @NNTiming.timeit(level=4, prefix="[API] ")
    def add_pipe_layer(self, idx, layer, shape=None, *args, **kwargs):
        _last_layer = self._layers[-1]
        if len(self._layers) == 1:
            _last_parent = None
        else:
            _last_parent = self._layers[-2]
        if not isinstance(_last_layer, NNPipe):
            raise BuildLayerError("Adding pipe layers to a non-NNPipe object is not allowed")
        if not _last_layer.initialized[idx] and len(shape) == 1:
            if _last_parent is None:
                raise BuildLayerError("Adding pipe layers at first without input shape is not allowed")
            _dim = (_last_parent.n_filters, _last_parent.out_h, _last_parent.out_w)
            shape = (_dim, shape[0])
        _last_layer.add(idx, layer, shape, *args, **kwargs)

    @NNTiming.timeit(level=4, prefix="[API] ")
    def preview(self, verbose=0):
        if not self._layers:
            rs = "None"
        else:
            rs = (
                "Input  :  {:<16s} - {}\n".format("Dimension", self._layers[0].shape[0]) +
                "\n".join([_layer.info for _layer in self._layers]))
        print("=" * 30 + "\n" + "Structure\n" + "-" * 30 + "\n" + rs + "\n" + "-" * 30)
        if verbose >= 1:
            print("Initial Values\n" + "-" * 30)
            print("\n".join(["({:^16s}) w_std: {:8.6} ; b_init: {:8.6}".format(
                batch[0].name, float(batch[1]), float(batch[2])) if not isinstance(
                batch[0], NNPipe) else "({:^16s}) ({:^3d})".format(
                "Pipe", len(batch[0]["nn_lst"])
            ) for batch in zip(self._layers, self._w_stds, self._b_inits) if not isinstance(
                batch[0], SubLayer) and not isinstance(
                batch[0], CostLayer) and not isinstance(
                batch[0], ConvPoolLayer)])
                  )
        if verbose >= 2:
            for layer in self._layers:
                if isinstance(layer, NNPipe):
                    layer.preview()
        print("-" * 30)


class NNDist(NNBase):
    NNTiming = Timing()

    def __init__(self, **kwargs):
        super(NNDist, self).__init__(**kwargs)

        self._sess = tf.Session()
        self._optimizer_factory = OptFactory()

        self._available_metrics = {
            "acc": self.acc, "_acc": self.acc,
            "f1": self.f1_score, "_f1_score": self.f1_score
        }

    @NNTiming.timeit(level=4, prefix="[Initialize] ")
    def initialize(self):
        self._layers = []
        self._optimizer = None
        self._w_stds, self._b_inits = [], []
        self._layer_names, self._layer_params = [], []
        self._lr = 0
        self.verbose = 1
        self._current_dimension = 0

        self._logs = {}
        self._metrics, self._metric_names, self._metric_rs = [], [], []

        self._loaded = False
        self._x_min = self._x_max = self._y_min = self._y_max = 0
        self._transferred_flags = {"train": False, "test": False}

        self._activations = None
        self._loss = self._train_step = None
        self._layer_factory = LayerFactory()
        self._tf_weights, self._tf_bias = [], []

        self._sess = tf.Session()

    # Property

    @property
    def layer_names(self):
        return [layer.name for layer in self._layers]

    @layer_names.setter
    def layer_names(self, value):
        self._layer_names = value

    @property
    def layer_special_params(self):
        return [layer.get_special_params(self._sess) for layer in self._layers]

    @layer_special_params.setter
    def layer_special_params(self, value):
        for layer, sp_param in zip(self._layers, value):
            if sp_param is not None:
                layer.set_special_params(sp_param)

    @property
    def optimizer(self):
        return self._optimizer.name

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    # Utils

    @staticmethod
    @NNTiming.timeit(level=4, prefix="[Private StaticMethod] ")
    def _transfer_x(x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(x.shape) == 4:
            x = x.transpose(0, 2, 3, 1)
        return x.astype(np.float32)

    @NNTiming.timeit(level=4)
    def _feed_data(self, x, y):
        if not self._transferred_flags["train"]:
            x = NNDist._transfer_x(x)
            self._transferred_flags["train"] = True
        y = np.asarray(y, dtype=np.float32)
        if len(x) != len(y):
            raise BuildNetworkError("Data fed to network should be identical in length, x: {} and y: {} found".format(
                len(x), len(y)
            ))
        self._x_min, self._x_max = np.min(x), np.max(x)
        self._y_min, self._y_max = np.min(y), np.max(y)
        return x, y

    @NNTiming.timeit(level=2)
    def _get_prediction(self, x, name=None, verbose=None):
        if verbose is None:
            verbose = self.verbose
        fc_shape = np.prod(x.shape[1:])  # type: int
        single_batch = int(NNConfig.BATCH_SIZE / fc_shape)
        if not single_batch:
            single_batch = 1
        if not single_batch:
            single_batch = 1
        if single_batch >= len(x):
            return self._sess.run(self._y_pred, {self._tfx: x})
        epoch = int(len(x) / single_batch)
        if not len(x) % single_batch:
            epoch += 1
        name = "Prediction" if name is None else "Prediction ({})".format(name)
        sub_bar = ProgressBar(max_value=epoch, name=name, start=False)
        if verbose >= NNVerbose.METRICS:
            sub_bar.start()
        rs, count = [], 0
        while count < len(x):
            count += single_batch
            if count >= len(x):
                rs.append(self._sess.run(self._y_pred, {self._tfx: x[count - single_batch:]}))
            else:
                rs.append(self._sess.run(self._y_pred, {self._tfx: x[count - single_batch:count]}))
            if verbose >= NNVerbose.METRICS:
                sub_bar.update()
        return np.vstack(rs)

    @NNTiming.timeit(level=4)
    def _get_activations(self, x, predict=False):
        if not isinstance(self._layers[0], NNPipe):
            activations = [self._layers[0].activate(x, self._tf_weights[0], self._tf_bias[0], predict)]
        else:
            activations = [self._layers[0].get_rs(x, predict)]
        for i, layer in enumerate(self._layers[1:]):
            if i == len(self._layers) - 2:
                if NNDist._is_conv(self._layers[i]):
                    fc_shape = np.prod(activations[-1].get_shape()[1:])  # type: int
                    activations[-1] = tf.reshape(activations[-1], [-1, int(fc_shape)])
                if self._tf_bias[-1] is not None:
                    activations.append(tf.matmul(activations[-1], self._tf_weights[-1]) + self._tf_bias[-1])
                else:
                    activations.append(tf.matmul(activations[-1], self._tf_weights[-1]))
            else:
                if not isinstance(layer, NNPipe):
                    activations.append(layer.activate(
                        activations[-1], self._tf_weights[i + 1], self._tf_bias[i + 1], predict))
                else:
                    activations.append(layer.get_rs(activations[-1], predict))
        return activations

    @NNTiming.timeit(level=1)
    def _get_l2_losses(self, lb):
        if lb <= 0:
            return 0.
        return [lb * tf.nn.l2_loss(w) for l, w in zip(self._layers, self._tf_weights)
                if not isinstance(l, SubLayer) and not isinstance(l, ConvPoolLayer)]

    @NNTiming.timeit(level=1)
    def _get_acts(self, x):
        with self._sess.as_default():
            activations = [_ac.eval() for _ac in self._get_activations(x, True)]
        return activations

    @NNTiming.timeit(level=3)
    def _append_log(self, x, y, y_pred, name, get_loss):
        if y_pred is None:
            y_pred = self._get_prediction(x, name)
        for i, metric_rs in enumerate(self._metric_rs):
            self._logs[name][i].append(metric_rs.eval({
                self._tfy: y, self._y_pred: y_pred
            }))
        if get_loss:
            self._logs[name][-1].append(
                self._loss.eval({self._tfy: y, self._y_pred: y_pred})
            )

    @NNTiming.timeit(level=3)
    def _print_metric_logs(self, name, show_loss):
        print()
        print("=" * 47)
        for i, metric in enumerate(self._metric_names):
            print("{:<16s} {:<16s}: {:12.8}".format(
                name, metric, self._logs[name][i][-1]))
        if show_loss:
            print("{:<16s} {:<16s}: {:12.8}".format(
                name, "loss", self._logs[name][-1][-1]))
        print("=" * 47)

    @NNTiming.timeit(level=1)
    def _draw_2d_network(self, radius=6, width=1200, height=800, padding=0.2,
                         plot_scale=2, plot_precision=0.03,
                         sub_layer_height_scale=0, **kwargs):
        if not kwargs["show"] and not kwargs["mp4"]:
            return
        layers = len(self._layers) + 1
        units = [layer.shape[0] for layer in self._layers] + [self._layers[-1].shape[1]]
        whether_sub_layers = np.array([False] + [isinstance(layer, SubLayer) for layer in self._layers])
        n_sub_layers = np.sum(whether_sub_layers)  # type: int

        plot_num = int(1 / plot_precision)
        if plot_num % 2 == 1:
            plot_num += 1
        half_plot_num = int(plot_num * 0.5)
        xf = np.linspace(self._x_min * plot_scale, self._x_max * plot_scale, plot_num)
        yf = np.linspace(self._x_min * plot_scale, self._x_max * plot_scale, plot_num) * -1
        input_x, input_y = np.meshgrid(xf, yf)
        input_xs = np.c_[input_x.ravel().astype(np.float32), input_y.ravel().astype(np.float32)]

        activations = self._sess.run(self._activations, {self._tfx: input_xs})
        activations = [activation.T.reshape(units[i + 1], plot_num, plot_num)
                       for i, activation in enumerate(activations)]
        graphs = []
        for j, activation in enumerate(activations):
            graph_group = []
            if j == len(activations) - 1:
                classes = np.argmax(activation, axis=0)
            else:
                classes = None
            for k, ac in enumerate(activation):
                data = np.zeros((plot_num, plot_num, 3), np.uint8)
                if j != len(activations) - 1:
                    mask = ac >= np.average(ac)
                else:
                    mask = classes == k
                data[mask], data[~mask] = [0, 165, 255], [255, 165, 0]
                graph_group.append(data)
            graphs.append(graph_group)

        img = np.full([height, width, 3], 255, dtype=np.uint8)
        axis0_padding = int(height / (layers - 1 + 2 * padding)) * padding + plot_num
        axis0_step = (height - 2 * axis0_padding) / layers
        sub_layer_decrease = int((1 - sub_layer_height_scale) * axis0_step)
        axis0 = np.linspace(
            axis0_padding,
            height + n_sub_layers * sub_layer_decrease - axis0_padding,
            layers, dtype=np.int)
        axis0 -= sub_layer_decrease * np.cumsum(whether_sub_layers)
        axis1_padding = plot_num
        axis1 = [np.linspace(axis1_padding, width - axis1_padding, unit + 2, dtype=np.int)
                 for unit in units]
        axis1 = [axis[1:-1] for axis in axis1]

        colors, thicknesses, masks = [], [], []
        for weight in self._tf_weights:
            line_info = VisUtil.get_line_info(weight.eval())
            colors.append(line_info[0])
            thicknesses.append(line_info[1])
            masks.append(line_info[2])

        for i, (y, xs) in enumerate(zip(axis0, axis1)):
            for j, x in enumerate(xs):
                if i == 0:
                    cv2.circle(img, (x, y), radius, (20, 215, 20), int(radius / 2))
                else:
                    graph = graphs[i - 1][j]
                    img[y - half_plot_num:y + half_plot_num, x - half_plot_num:x + half_plot_num] = graph
            if i > 0:
                cv2.putText(img, self._layers[i - 1].name, (12, y - 36), cv2.LINE_AA, 0.6, (0, 0, 0), 1)
        for i, y in enumerate(axis0):
            if i == len(axis0) - 1:
                break
            for j, x in enumerate(axis1[i]):
                new_y = axis0[i + 1]
                whether_sub_layer = isinstance(self._layers[i], SubLayer)
                for k, new_x in enumerate(axis1[i + 1]):
                    if whether_sub_layer and j != k:
                        continue
                    if masks[i][j][k]:
                        cv2.line(img, (x, y + half_plot_num), (new_x, new_y - half_plot_num),
                                 colors[i][j][k], thicknesses[i][j][k])

        return img

    # Init

    @NNTiming.timeit(level=4)
    def _init_optimizer(self, optimizer=None):
        if optimizer is None:
            if isinstance(self._optimizer, str):
                optimizer = self._optimizer
            else:
                if self._optimizer is None:
                    self._optimizer = Adam(self._lr)
                if isinstance(self._optimizer, Optimizer):
                    return
                raise BuildNetworkError("Invalid optimizer '{}' provided".format(self._optimizer))
        if isinstance(optimizer, str):
            self._optimizer = self._optimizer_factory.get_optimizer_by_name(optimizer, self._lr)
        elif isinstance(optimizer, Optimizer):
            self._optimizer = optimizer
        else:
            raise BuildNetworkError("Invalid optimizer '{}' provided".format(optimizer))

    @NNTiming.timeit(level=4)
    def _init_layers(self):
        for _layer in self._layers:
            _layer.init(self._sess)

    @NNTiming.timeit(level=4)
    def _init_structure(self, verbose):
        x_shape = self._layers[0].shape[0]
        if isinstance(x_shape, int):
            x_shape = x_shape,
        y_shape = self._layers[-1].shape[1]
        x_placeholder, y_placeholder = np.zeros((1, *x_shape)), np.zeros((1, y_shape))
        self.fit(x_placeholder, y_placeholder, epoch=0, train_only=True, verbose=verbose)
        self._transferred_flags["train"] = False

    @NNTiming.timeit(level=4)
    def _init_train_step(self, sess):
        if not self._loaded:
            self._train_step = self._optimizer.minimize(self._loss)
            sess.run(tf.global_variables_initializer())
        else:
            _var_cache = set(tf.global_variables())
            self._train_step = self._optimizer.minimize(self._loss)
            sess.run(tf.variables_initializer(set(tf.global_variables()) - _var_cache))

    # Batch Work

    @NNTiming.timeit(level=2)
    def _batch_work(self, i, bar, counter, x_train, y_train, x_test, y_test, show_loss, condition,
                    tensorboard_verbose, train_repeat, sess, train_merge_op, test_merge_op,
                    train_writer, test_writer):
        if tensorboard_verbose > 0:
            count = counter * train_repeat + i
            y_train_pred = self.predict(x_train, get_raw_results=True, transfer_x=False)
            y_test_pred = self.predict(x_test, get_raw_results=True, transfer_x=False)
            train_summary = sess.run(train_merge_op, feed_dict={
                self._tfy: y_train, self._y_pred: y_train_pred
            })
            test_summary = sess.run(test_merge_op, feed_dict={
                self._tfy: y_test, self._y_pred: y_test_pred
            })
            train_writer.add_summary(train_summary, count)
            test_writer.add_summary(test_summary, count)
        else:
            y_train_pred = y_test_pred = None
        if bar is not None:
            condition = bar.update() and condition
        if condition:
            self._append_log(x_train, y_train, y_train_pred, "Train", show_loss)
            self._append_log(x_test, y_test, y_test_pred, "Test", show_loss)
            self._print_metric_logs("Train", show_loss)
            self._print_metric_logs("Test", show_loss)

    # API

    @NNTiming.timeit(level=4, prefix="[API] ")
    def get_current_pipe(self, idx):
        _last_layer = self._layers[-1]
        if not isinstance(_last_layer, NNPipe):
            return
        return _last_layer["nn_lst"][idx]

    @NNTiming.timeit(level=4, prefix="[API] ")
    def build(self, units="load"):
        if isinstance(units, str):
            if units == "load":
                for name, param in zip(self._layer_names, self._layer_params):
                    self.add(name, *param)
            else:
                raise NotImplementedError("Invalid param '{}' provided to 'build' method".format(units))
        else:
            try:
                units = np.asarray(units).flatten().astype(np.int)
            except ValueError as err:
                raise BuildLayerError(err)
            if len(units) < 2:
                raise BuildLayerError("At least 2 layers are needed")
            _input_shape = (units[0], units[1])
            self.initialize()
            self.add("ReLU", _input_shape)
            for unit_num in units[2:-1]:
                self.add("ReLU", (unit_num,))
            self.add("CrossEntropy", (units[-1],))

    @NNTiming.timeit(level=4, prefix="[API] ")
    def split_data(self, x, y, x_test, y_test,
                   train_only, training_scale=NNConfig.TRAINING_SCALE):
        if train_only:
            if x_test is not None and y_test is not None:
                if not self._transferred_flags["test"]:
                    x, y = np.vstack((x, NNDist._transfer_x(np.asarray(x_test)))), np.vstack((y, y_test))
                    self._transferred_flags["test"] = True
            x_train = x_test = x.astype(np.float32)
            y_train = y_test = y.astype(np.float32)
        else:
            shuffle_suffix = np.random.permutation(len(x))
            x, y = x[shuffle_suffix], y[shuffle_suffix]
            if x_test is None or y_test is None:
                train_len = int(len(x) * training_scale)
                x_train, y_train = x[:train_len], y[:train_len]
                x_test, y_test = x[train_len:], y[train_len:]
            else:
                x_train, y_train = x, y
                if not self._transferred_flags["test"]:
                    x_test, y_test = NNDist._transfer_x(np.asarray(x_test)), np.asarray(y_test, dtype=np.float32)
                    self._transferred_flags["test"] = True
        if NNConfig.BOOST_LESS_SAMPLES:
            if y_train.shape[1] != 2:
                raise BuildNetworkError("It is not permitted to boost less samples in multiple classification")
            y_train_arg = np.argmax(y_train, axis=1)
            y0 = y_train_arg == 0
            y1 = ~y0
            y_len, y0_len = len(y_train), np.sum(y0)  # type: int
            if y0_len > int(0.5 * y_len):
                y0, y1 = y1, y0
                y0_len = y_len - y0_len
            boost_suffix = np.random.randint(y0_len, size=y_len - y0_len)
            x_train = np.vstack((x_train[y1], x_train[y0][boost_suffix]))
            y_train = np.vstack((y_train[y1], y_train[y0][boost_suffix]))
            shuffle_suffix = np.random.permutation(len(x_train))
            x_train, y_train = x_train[shuffle_suffix], y_train[shuffle_suffix]
        return (x_train, x_test), (y_train, y_test)

    @NNTiming.timeit(level=1, prefix="[API] ")
    def fit(self,
            x, y, x_test=None, y_test=None,
            lr=0.001, lb=0.001, epoch=10, weight_scale=1,
            batch_size=128, record_period=1, train_only=False, optimizer=None,
            show_loss=True, metrics=None, do_log=False, verbose=None,
            tensorboard_verbose=0, animation_params=None):
        x, y = self._feed_data(x, y)
        self._lr = lr
        self._init_optimizer(optimizer)
        print("Optimizer: ", self._optimizer.name)
        print("-" * 30)

        if not self._layers:
            raise BuildNetworkError("Please provide layers before fitting data")

        if y.shape[1] != self._current_dimension:
            raise BuildNetworkError("Output layer's shape should be {}, {} found".format(
                self._current_dimension, y.shape[1]))

        (x_train, x_test), (y_train, y_test) = self.split_data(x, y, x_test, y_test, train_only)
        train_len, test_len = len(x_train), len(x_test)
        batch_size = min(batch_size, train_len)
        do_random_batch = train_len > batch_size
        train_repeat = 1 if not do_random_batch else int(train_len / batch_size) + 1

        with tf.name_scope("Entry"):
            self._tfx = tf.placeholder(tf.float32, shape=[None, *x.shape[1:]])
        self._tfy = tf.placeholder(tf.float32, shape=[None, y.shape[1]])
        if epoch <= 0:
            return

        self._metrics = ["acc"] if metrics is None else metrics
        for i, metric in enumerate(self._metrics):
            if isinstance(metric, str):
                if metric not in self._available_metrics:
                    raise BuildNetworkError("Metric '{}' is not implemented".format(metric))
                self._metrics[i] = self._available_metrics[metric]
        self._metric_names = [_m.__name__ for _m in self._metrics]

        self._logs = {
            name: [[] for _ in range(len(self._metrics) + 1)] for name in ("Train", "Test")
        }
        if verbose is not None:
            self.verbose = verbose

        bar = ProgressBar(max_value=max(1, epoch // record_period), name="Epoch", start=False)
        if self.verbose >= NNVerbose.EPOCH:
            bar.start()
        img = None

        *animation_properties, animation_params = self._get_animation_params(animation_params)
        with self._sess.as_default() as sess:
            with tf.name_scope("ActivationFlow"):
                self._activations = self._get_activations(self._tfx)
            self._y_pred = self._activations[-1]
            l2_losses = self._get_l2_losses(lb)  # type: list
            self._loss = self._layers[-1].calculate(self._tfy, self._y_pred) + tf.reduce_sum(l2_losses)
            self._metric_rs = [metric(self._tfy, self._y_pred) for metric in self._metrics]
            self._init_train_step(sess)
            for weight in self._tf_weights:
                weight *= weight_scale

            if tensorboard_verbose > 0:
                log_dir = os.path.join("tbLogs", str(datetime.datetime.now())[:19].replace(":", "-"))
                train_dir = os.path.join(log_dir, "train")
                test_dir = os.path.join(log_dir, "test")
                for _dir in (log_dir, train_dir, test_dir):
                    if not os.path.isdir(_dir):
                        os.makedirs(_dir)
                test_summary_ops = []
                with tf.name_scope("l2_loss"):
                    layer_names = [
                        self._get_tb_name(layer) for layer in self._layers
                        if not isinstance(layer, SubLayer) and not isinstance(layer, ConvPoolLayer)
                    ]
                    for name, l2_loss in zip(layer_names, l2_losses):
                        tf.summary.scalar(name, l2_loss)
                with tf.name_scope("GlobalSummaries"):
                    test_summary_ops.append(tf.summary.scalar("loss", self._loss))
                    for name, metric_rs in zip(self._metric_names, self._metric_rs):
                        test_summary_ops.append(tf.summary.scalar(name, metric_rs))
                train_merge_op = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(train_dir, sess.graph)
                train_writer.add_graph(sess.graph)
                test_writer = tf.summary.FileWriter(test_dir)
                test_merge_op = tf.summary.merge(test_summary_ops)
            else:
                train_writer = test_writer = train_merge_op = test_merge_op = None

            args = (
                x_train, y_train, x_test, y_test, show_loss,
                self.verbose >= NNVerbose.METRICS_DETAIL,
                tensorboard_verbose, train_repeat, sess, train_merge_op, test_merge_op,
                train_writer, test_writer
            )
            ims = []
            for counter in range(epoch):
                if self.verbose >= NNVerbose.ITER and counter % record_period == 0:
                    sub_bar = ProgressBar(max_value=train_repeat * record_period - 1, name="Iteration")
                else:
                    sub_bar = None
                self._batch_training(
                    x_train, y_train, batch_size, train_repeat,
                    self._loss, self._train_step, sub_bar, counter, *args)
                self._handle_animation(
                    counter, x, y, ims, animation_params, *animation_properties,
                    img=self._draw_2d_network(**animation_params), name="Neural Network"
                )
                if (counter + 1) % record_period == 0:
                    if do_log:
                        self._append_log(x_train, y_train, None, "Train", show_loss)
                        self._append_log(x_test, y_test, None,  "Test", show_loss)
                        if self.verbose >= NNVerbose.METRICS:
                            self._print_metric_logs("Train", show_loss)
                            self._print_metric_logs("Test", show_loss)
                    if self.verbose >= NNVerbose.EPOCH:
                        bar.update(counter // record_period + 1)
        if img is not None:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        self._handle_mp4(ims, animation_properties, "NN")

    @NNTiming.timeit(level=2, prefix="[API] ")
    def save(self, path=None, name=None, overwrite=True):
        path = "Models" if path is None else path
        name = "Cache" if name is None else name
        folder = os.path.join(path, name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        _dir = os.path.join(folder, "Model")
        if os.path.isfile(_dir):
            if not overwrite:
                _count = 1
                _new_dir = _dir + "({})".format(_count)
                while os.path.isfile(_new_dir):
                    _count += 1
                    _new_dir = _dir + "({})".format(_count)
                _dir = _new_dir
            else:
                os.remove(_dir)

        print()
        print("=" * 60)
        print("Saving Model to {}...".format(folder))
        print("-" * 60)

        with open(_dir + ".nn", "wb") as file:
            # We don't need w_stds & b_inits when we load a model
            _dic = {
                "structures": {
                    "_lr": self._lr,
                    "_layer_names": self.layer_names,
                    "_layer_params": self._layer_params,
                    "_next_dimension": self._current_dimension
                },
                "params": {
                    "_logs": self._logs,
                    "_metric_names": self._metric_names,
                    "_optimizer": self._optimizer.name,
                    "layer_special_params": self.layer_special_params
                }
            }
            pickle.dump(_dic, file)
        saver = tf.train.Saver()
        saver.save(self._sess, _dir)
        graph_io.write_graph(self._sess.graph, os.path.join(path, name), "Model.pb", False)
        with tf.name_scope("OutputFlow"):
            self.get_rs(self._tfx)
        _output = ""
        for op in self._sess.graph.get_operations()[::-1]:
            if "OutputFlow" in op.name:
                _output = op.name
                break
        with open(os.path.join(path, name, "IO.txt"), "w") as file:
            file.write("\n".join([
                "Input  : Entry/Placeholder:0",
                "Output : {}:0".format(_output)
            ]))
        graph_io.write_graph(self._sess.graph, os.path.join(path, name), "Cache.pb", False)
        freeze_graph.freeze_graph(
            os.path.join(path, name, "Cache.pb"),
            "", True, os.path.join(path, name, "Model"),
            _output, "save/restore_all", "save/Const:0",
            os.path.join(path, name, "Frozen.pb"), True, ""
        )
        os.remove(os.path.join(path, name, "Cache.pb"))

        print("Done")
        print("=" * 60)

    @NNTiming.timeit(level=2, prefix="[API] ")
    def load(self, path=None, verbose=2):
        if path is None:
            path = os.path.join("Models", "Cache", "Model")
        else:
            path = os.path.join(path, "Model")
        self.initialize()
        try:
            with open(path + ".nn", "rb") as file:
                _dic = pickle.load(file)
                for key, value in _dic["structures"].items():
                    setattr(self, key, value)
                self.build()
                for key, value in _dic["params"].items():
                    setattr(self, key, value)
                self._init_optimizer()
                for i in range(len(self._metric_names) - 1, -1, -1):
                    name = self._metric_names[i]
                    if name not in self._available_metrics:
                        self._metric_names.pop(i)
                    else:
                        self._metrics.insert(0, self._available_metrics[name])
        except Exception as err:
            raise BuildNetworkError("Failed to load Network ({}), structure initialized".format(err))
        self._loaded = True

        saver = tf.train.Saver()
        saver.restore(self._sess, path)
        self._init_layers()
        self._init_structure(verbose)

        print()
        print("=" * 30)
        print("Model restored")
        print("=" * 30)

    @NNTiming.timeit(level=4, prefix="[API] ")
    def predict(self, x, get_raw_results=False, transfer_x=True):
        x = np.asarray(x, dtype=np.float32)
        if transfer_x:
            x = NNDist._transfer_x(x)
        y_pred = self._get_prediction(x)
        return y_pred if get_raw_results else np.argmax(y_pred, axis=1)

    @NNTiming.timeit()
    def evaluate(self, x, y, metrics=None, tar=0, prefix="Acc", **kwargs):
        logs, y_pred = [], self._get_prediction(NNDist._transfer_x(x))
        for i, metric_rs in enumerate(self._metric_rs):
            logs.append(self._sess.run(metric_rs, {
                self._tfy: y, self._y_pred: y_pred
            }))
        if isinstance(tar, int):
            print(prefix + ": {:12.8}".format(logs[tar]))
        return logs

    def draw_results(self):
        metrics_log, loss_log = {}, {}
        for key, value in sorted(self._logs.items()):
            metrics_log[key], loss_log[key] = value[:-1], value[-1]

        for i, name in enumerate(sorted(self._metric_names)):
            plt.figure()
            plt.title("Metric Type: {}".format(name))
            for key, log in sorted(metrics_log.items()):
                xs = np.arange(len(log[i])) + 1
                plt.plot(xs, log[i], label="Data Type: {}".format(key))
            plt.legend(loc=4)
            plt.show()
            plt.close()

        plt.figure()
        plt.title("Cost")
        for key, loss in sorted(loss_log.items()):
            xs = np.arange(len(loss)) + 1
            plt.plot(xs, loss, label="Data Type: {}".format(key))
        plt.legend()
        plt.show()

    def draw_conv_weights(self):
        with self._sess.as_default():
            for i, (name, weight) in enumerate(zip(self.layer_names, self._tf_weights)):
                weight = weight.eval()
                if len(weight.shape) != 4:
                    continue
                for j, _w in enumerate(weight.transpose(2, 3, 0, 1)):
                    VisUtil.show_batch_img(_w, "{} {} filter {}".format(name, i + 1, j + 1))

    def draw_conv_series(self, x, shape=None):
        x = np.asarray(x)
        for xx in x:
            VisUtil.show_img(VisUtil.trans_img(xx, shape), "Original")
            for i, (layer, ac) in enumerate(zip(
                    self._layers, self._get_acts(np.array([xx.transpose(1, 2, 0)], dtype=np.float32)))):
                if len(ac.shape) == 4:
                    VisUtil.show_batch_img(ac[0].transpose(2, 0, 1), "Layer {} ({})".format(i + 1, layer.name))
                else:
                    ac = ac[0]
                    length = sqrt(np.prod(ac.shape))
                    if length < 10:
                        continue
                    (height, width) = xx.shape[1:] if shape is None else shape[1:]
                    sqrt_shape = sqrt(height * width)
                    oh, ow = int(length * height / sqrt_shape), int(length * width / sqrt_shape)
                    VisUtil.show_img(ac[:oh * ow].reshape(oh, ow), "Layer {} ({})".format(i + 1, layer.name))

    @staticmethod
    def fuck_pycharm_warning():
        print(Axes3D.acorr)


class NNFrozen(NNBase):
    NNTiming = Timing()

    def __init__(self):
        super(NNFrozen, self).__init__()
        self._sess = tf.Session()
        self._entry = self._output = None

    @NNTiming.timeit(level=4, prefix="[API] ")
    def load(self, path=None, pb="Frozen.pb"):
        if path is None:
            path = os.path.join("Models", "Cache")
        try:
            with open(os.path.join(path, "Model.nn"), "rb") as file:
                _dic = pickle.load(file)
                for key, value in _dic["structures"].items():
                    setattr(self, key, value)
                for name, param in zip(self._layer_names, self._layer_params):
                    self.add(name, *param)
                for key, value in _dic["params"].items():
                    setattr(self, key, value)
        except Exception as err:
            raise BuildNetworkError("Failed to load Network ({}), structure initialized".format(err))

        with open(os.path.join(path, "IO.txt"), "r") as file:
            self._entry = file.readline().strip()[9:]
            self._output = file.readline().strip()[9:]
        Util.load_frozen_graph(os.path.join(path, pb), True, self._entry, self._output)

        print()
        print("=" * 30)
        print("Model restored")
        print("=" * 30)

    @NNTiming.timeit(level=2, prefix="[API] ")
    def predict(self, x, get_raw_results=False, **kwargs):
        x = NNDist._transfer_x(np.asarray(x))
        rs = []
        batch_size = floor(1e6 / np.prod(x.shape[1:]))
        epoch = int(ceil(len(x) / batch_size))
        output = self._sess.graph.get_tensor_by_name(self._output)
        bar = ProgressBar(max_value=epoch, name="Predict")
        for i in range(epoch):
            if i == epoch - 1:
                rs.append(self._sess.run(output, {
                    self._entry: x[i * batch_size:]
                }))
            else:
                rs.append(self._sess.run(output, {
                    self._entry: x[i * batch_size:(i + 1) * batch_size]
                }))
            bar.update()
        y_pred = np.vstack(rs).astype(np.float32)
        return y_pred if get_raw_results else np.argmax(y_pred, axis=1)

    @NNTiming.timeit(level=1, prefix="[API] ")
    def evaluate(self, x, y, metrics=None, tar=None, prefix="Acc", **kwargs):
        y_pred = self.predict(x)
        print("Acc: {:8.6} %".format(100 * np.sum(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)) / len(y)))


class NNPipe:
    NNTiming = Timing()

    def __init__(self, num):
        self._nn_lst = [NNBase() for _ in range(num)]
        for _nn in self._nn_lst:
            _nn.verbose = 0
        self._initialized = [False] * num
        self.is_sub_layer = False
        self.parent = None

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)
        return

    def __str__(self):
        return "NNPipe"

    __repr__ = __str__

    @property
    def name(self):
        return "NNPipe"

    @property
    def n_filters(self):
        return sum([_nn["layers"][-1].n_filters for _nn in self._nn_lst])

    @property
    def out_h(self):
        return self._nn_lst[0]["layers"][-1].out_h

    @property
    def out_w(self):
        return self._nn_lst[0]["layers"][-1].out_w

    @property
    def shape(self):
        # TODO: modify shape[0] to be the correct one
        return (self.n_filters, self.out_h, self.out_w), (self.n_filters, self.out_h, self.out_w)

    @property
    def info(self):
        return "Pipe ({:^3d})".format(len(self._nn_lst)) + " " * 65 + "- out: {}".format(
            self.shape[1])

    @property
    def initialized(self):
        return self._initialized

    @NNTiming.timeit(level=4, prefix="[API] ")
    def preview(self):
        print("=" * 90)
        print("Pipe Structure")
        for i, nn in enumerate(self._nn_lst):
            print("-" * 60 + "\n" + str(i) + "\n" + "-" * 60)
            nn.preview()

    @NNTiming.timeit(level=4, prefix="[API] ")
    def add(self, idx, layer, shape, *args, **kwargs):
        if shape is None:
            self._nn_lst[idx].add(layer, *args, **kwargs)
        else:
            self._nn_lst[idx].add(layer, shape, *args, **kwargs)
        self._initialized[idx] = True

    @NNTiming.timeit(level=1, prefix="[API] ")
    def get_rs(self, x, predict):
        return tf.concat([nn.get_rs(x, predict=predict, pipe=True) for nn in self._nn_lst], 3)
