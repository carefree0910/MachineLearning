import os
import cv2
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from NN.PyTorch.Basic.Layers import *
from NN.Basic.Networks import NNConfig, NNVerbose
from NN.PyTorch.Optimizers import OptFactory

from Util.Util import VisUtil
from Util.ProgressBar import ProgressBar
from Util.Bases import TorchBasicClassifierBase


# PyTorch Implementation without using auto-grad

class NNDist(TorchBasicClassifierBase):
    NNTiming = Timing()

    def __init__(self, **kwargs):
        super(NNDist, self).__init__(**kwargs)
        self._layers, self._weights, self._bias = [], [], []
        self._layer_names, self._layer_shapes, self._layer_params = [], [], []
        self._lr, self._epoch, self._regularization_param = 0, 0, 0
        self._w_optimizer, self._b_optimizer, self._optimizer_name = None, None, ""
        self.verbose = 1

        self._whether_apply_bias = False
        self._current_dimension = 0

        self._logs = {}
        self._metrics, self._metric_names = [], []

        self._x_min, self._x_max = 0, 0
        self._y_min, self._y_max = 0, 0

        self._layer_factory = LayerFactory()
        self._optimizer_factory = OptFactory()

        self._available_metrics = {
            "acc": NNDist.acc, "_acc": NNDist.acc,
            "f1": NNDist.f1_score, "_f1_score": NNDist.f1_score
        }

    @NNTiming.timeit(level=4, prefix="[Initialize] ")
    def initialize(self):
        self._layers, self._weights, self._bias = [], [], []
        self._layer_names, self._layer_shapes, self._layer_params = [], [], []
        self._lr, self._epoch, self._regularization_param = 0, 0, 0
        self._w_optimizer, self._b_optimizer, self._optimizer_name = None, None, ""
        self.verbose = 1

        self._whether_apply_bias = False
        self._current_dimension = 0

        self._logs = []
        self._metrics, self._metric_names = [], []

        self._x_min, self._x_max = 0, 0
        self._y_min, self._y_max = 0, 0

    # Property

    @property
    def name(self):
        return (
            "-".join([str(_layer.shape[1]) for _layer in self._layers]) +
            " at {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        )

    @property
    def layer_names(self):
        return [layer.name for layer in self._layers]

    @layer_names.setter
    def layer_names(self, value):
        self._layer_names = value

    @property
    def layer_shapes(self):
        return [layer.size() for layer in self._layers]

    @layer_shapes.setter
    def layer_shapes(self, value):
        self._layer_shapes = value

    @property
    def layer_params(self):
        return self._layer_params

    @layer_params.setter
    def layer_params(self, value):
        self.layer_params = value

    @property
    def layer_special_params(self):
        return [layer.special_params for layer in self._layers]

    @layer_special_params.setter
    def layer_special_params(self, value):
        for layer, sp_param in zip(self._layers, value):
            if sp_param is not None:
                layer.set_special_params(sp_param)

    @property
    def optimizer(self):
        return self._optimizer_name

    @optimizer.setter
    def optimizer(self, value):
        try:
            self._optimizer_name = value
        except KeyError:
            raise BuildNetworkError("Invalid Optimizer '{}' provided".format(value))

    # Utils

    @NNTiming.timeit(level=4)
    def _get_min_max(self, x, y):
        x, y = x.numpy(), y.numpy()
        self._x_min, self._x_max = np.min(x), np.max(x)
        self._y_min, self._y_max = np.min(y), np.max(y)

    @NNTiming.timeit(level=4, prefix="[API] ")
    def _split_data(self, x, y, x_test, y_test,
                    train_only, training_scale=NNConfig.TRAINING_SCALE):
        if train_only:
            if x_test is not None and y_test is not None:
                x, y = torch.cat((x, x_test)), torch.cat((y, y_test))
            x_train, y_train, x_test, y_test = x, y, x, y
        else:
            shuffle_suffix = torch.randperm(len(x))
            x, y = x[shuffle_suffix], y[shuffle_suffix]
            if x_test is None or y_test is None:
                train_len = int(len(x) * training_scale)
                x_train, y_train = x[:train_len], y[:train_len]
                x_test, y_test = x[train_len:], y[train_len:]
            elif x_test is None or y_test is None:
                raise BuildNetworkError("Please provide test sets if you want to split data on your own")
            else:
                x_train, y_train = x, y
        if NNConfig.BOOST_LESS_SAMPLES:
            if y_train.shape[1] != 2:
                raise BuildNetworkError("It is not permitted to boost less samples in multiple classification")
            y_train_arg = torch.max(y_train, dim=1)[1]
            y0 = y_train_arg == 0
            y1 = ~y0
            y_len, y0_len = len(y_train), torch.sum(y0)  # type: float
            if y0_len > int(0.5 * y_len):
                y0, y1 = y1, y0
                y0_len = y_len - y0_len
            boost_suffix = torch.IntTensor(y_len - y0_len).random_(y0_len)
            x_train = torch.cat((x_train[y1], x_train[y0][boost_suffix]))
            y_train = torch.cat((y_train[y1], y_train[y0][boost_suffix]))
            shuffle_suffix = torch.randperm(len(x_train))
            x_train, y_train = x_train[shuffle_suffix], y_train[shuffle_suffix]
        return (x_train, x_test), (y_train, y_test)

    @NNTiming.timeit(level=4)
    def _add_weight(self, shape):
        self._weights.append(torch.randn(*shape))
        self._bias.append(torch.zeros((1, shape[1])))

    @NNTiming.timeit(level=4)
    def _add_layer(self, layer, *args, **kwargs):
        if not self._layers and isinstance(layer, str):
            layer = self._layer_factory.get_root_layer_by_name(layer, *args, **kwargs)
            if layer:
                self.add(layer)
                return
        parent = self._layers[-1]
        if isinstance(parent, CostLayer):
            raise BuildLayerError("Adding layer after CostLayer is not permitted")
        if isinstance(layer, str):
            layer, shape = self._layer_factory.get_layer_by_name(
                layer, parent, self._current_dimension, *args, **kwargs
            )
            if shape is None:
                self.add(layer)
                return
            _current, _next = shape
        else:
            _current, _next = args
        if isinstance(layer, SubLayer):
            parent.child = layer
            layer.is_sub_layer = True
            layer.root = layer.root
            layer.root.last_sub_layer = layer
            self.parent = parent
            self._layers.append(layer)
            self._weights.append(torch.Tensor(0))
            self._bias.append(torch.Tensor([0.]))
            self._current_dimension = _next
        else:
            self._layers.append(layer)
            self._add_weight((_current, _next))
            self._current_dimension = _next
        self._update_layer_information(layer)

    @NNTiming.timeit(level=4)
    def _update_layer_information(self, layer):
        self._layer_params.append(layer.params)
        if len(self._layer_params) > 1 and not layer.is_sub_layer:
            self._layer_params[-1] = ((self._layer_params[-1][0][1],), *self._layer_params[-1][1:])

    @NNTiming.timeit(level=1)
    def _get_prediction(self, x, name=None, batch_size=1e6, verbose=None):
        if verbose is None:
            verbose = self.verbose
        fc_shape = np.prod(x.size()[1:])  # type: int
        single_batch = int(batch_size / fc_shape)
        if not single_batch:
            single_batch = 1
        if single_batch >= len(x):
            return self._get_activations(x, predict=True).pop()
        epoch = int(len(x) / single_batch)
        if not len(x) % single_batch:
            epoch += 1
        name = "Prediction" if name is None else "Prediction ({})".format(name)
        sub_bar = ProgressBar(max_value=epoch, name=name, start=False)
        if verbose >= NNVerbose.METRICS:
            sub_bar.start()
        rs, count = [self._get_activations(x[:single_batch], predict=True).pop()], single_batch
        if verbose >= NNVerbose.METRICS:
            sub_bar.update()
        while count < len(x):
            count += single_batch
            if count >= len(x):
                rs.append(self._get_activations(x[count-single_batch:], predict=True).pop())
            else:
                rs.append(self._get_activations(x[count-single_batch:count], predict=True).pop())
            if verbose >= NNVerbose.METRICS:
                sub_bar.update()
        return torch.cat(rs)

    @NNTiming.timeit(level=1)
    def _get_activations(self, x, predict=False):
        activations = [self._layers[0].activate(x, self._weights[0], self._bias[0], predict)]
        for i, layer in enumerate(self._layers[1:]):
            activations.append(layer.activate(
                activations[-1], self._weights[i + 1], self._bias[i + 1], predict))
        return activations

    @NNTiming.timeit(level=3)
    def _append_log(self, x, y, name, get_loss=True):
        y_pred = self._get_prediction(x, name)
        for i, metric in enumerate(self._metrics):
            self._logs[name][i].append(metric(
                torch.max(y, dim=1)[1].numpy(), torch.max(y_pred, dim=1)[1].numpy()
            ))
        if get_loss:
            self._logs[name][-1].append(
                (self._layers[-1].calculate(y, y_pred) / len(y))
            )

    @NNTiming.timeit(level=3)
    def _print_metric_logs(self, show_loss, data_type):
        print()
        print("=" * 47)
        for i, name in enumerate(self._metric_names):
            print("{:<16s} {:<16s}: {:12.8}".format(
                data_type, name, self._logs[data_type][i][-1]))
        if show_loss:
            print("{:<16s} {:<16s}: {:12.8}".format(
                data_type, "loss", self._logs[data_type][-1][-1]))
        print("=" * 47)

    # TODO
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
        xf = torch.linspace(self._x_min * plot_scale, self._x_max * plot_scale, plot_num)
        yf = torch.linspace(self._x_min * plot_scale, self._x_max * plot_scale, plot_num) * -1
        input_xs = torch.stack([
            xf.repeat(plot_num), yf.repeat(plot_num, 1).t().contiguous().view(-1)
        ], 1)

        activations = [
            activation.numpy().T.reshape(units[i + 1], plot_num, plot_num)
            for i, activation in enumerate(
                self._get_activations(input_xs, predict=True)
            )]
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
        for weight in self._weights:
            line_info = VisUtil.get_line_info(weight.numpy().copy())
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

    # Optimizing Process

    @NNTiming.timeit(level=4)
    def _init_optimizer(self):
        if not isinstance(self._w_optimizer, Optimizer):
            self._w_optimizer = self._optimizer_factory.get_optimizer_by_name(
                self._w_optimizer, self._weights, self._lr, self._epoch)
        if not isinstance(self._b_optimizer, Optimizer):
            self._b_optimizer = self._optimizer_factory.get_optimizer_by_name(
                self._b_optimizer, self._bias, self._lr, self._epoch)
        if self._w_optimizer.name != self._b_optimizer.name:
            self._optimizer_name = None
        else:
            self._optimizer_name = self._w_optimizer.name

    @NNTiming.timeit(level=1)
    def _opt(self, i, activation, delta):
        self._weights[i] *= self._regularization_param
        self._weights[i] += self._w_optimizer.run(
            i, activation.view(activation.size()[0], -1).t().mm(delta)
        )
        if self._whether_apply_bias:
            self._bias[i] += self._b_optimizer.run(
                i, torch.sum(delta, dim=0)
            )

    # API

    @NNTiming.timeit(level=4, prefix="[API] ")
    def add(self, layer, *args, **kwargs):
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
                self._add_weight(layer.shape)
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
                    elif _current != self._current_dimension:
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

    # TODO
    @NNTiming.timeit(level=4, prefix="[API] ")
    def build(self, units="build"):
        if isinstance(units, str):
            if units == "build":
                for name, param in zip(self._layer_names, self._layer_params):
                    self.add(name, *param)
            else:
                raise NotImplementedError("Invalid param '{}' provided to 'build' method".format(units))
        else:
            try:
                units = list(units)
            except ValueError as err:
                raise BuildLayerError(err)
            if len(units) < 2:
                raise BuildLayerError("At least 2 layers are needed")
            _input_shape = (units[0], units[1])
            self.initialize()
            self.add(ReLU(_input_shape))
            for unit_num in units[2:-1]:
                self.add(ReLU((unit_num,)))
            self.add("CrossEntropy", (units[-1],))

    @NNTiming.timeit(level=4, prefix="[API] ")
    def preview(self):
        if not self._layers:
            rs = "None"
        else:
            rs = (
                "Input  :  {:<10s} - {}\n".format("Dimension", self._layers[0].shape[0]) +
                "\n".join([
                    "Layer  :  {:<10s} - {} {}".format(
                        layer.name, layer.shape[1], layer.description
                    ) if isinstance(layer, SubLayer) else
                    "Layer  :  {:<10s} - {}".format(
                        layer.name, layer.shape[1]
                    ) for layer in self._layers[:-1]
                ]) + "\nCost   :  {:<16s}".format(str(self._layers[-1]))
            )
        print("=" * 30 + "\n" + "Structure\n" + "-" * 30 + "\n" + rs + "\n" + "-" * 30 + "\n")

    @NNTiming.timeit(level=1, prefix="[API] ")
    def fit(self,
            x, y, x_test=None, y_test=None,
            batch_size=128, record_period=1, train_only=False,
            optimizer=None, w_optimizer=None, b_optimizer=None,
            lr=0.001, lb=0.001, epoch=20, weight_scale=1, apply_bias=True,
            show_loss=True, metrics=None, do_log=True, verbose=None,
            visualize=False, visualize_setting=None,
            draw_weights=False, animation_params=None):
        self._lr, self._epoch = lr, epoch
        for weight in self._weights:
            weight *= weight_scale
        if not self._w_optimizer or not self._b_optimizer:
            if not self._optimizer_name:
                if optimizer is None:
                    optimizer = "Adam"
                self._w_optimizer = optimizer if w_optimizer is None else w_optimizer
                self._b_optimizer = optimizer if b_optimizer is None else b_optimizer
            else:
                if not self._w_optimizer:
                    self._w_optimizer = self._optimizer_name
                if not self._b_optimizer:
                    self._b_optimizer = self._optimizer_name
        self._init_optimizer()
        assert isinstance(self._w_optimizer, Optimizer) and isinstance(self._b_optimizer, Optimizer)
        print()
        print("=" * 30)
        print("Optimizers")
        print("-" * 30)
        print("w: {}\nb: {}".format(self._w_optimizer, self._b_optimizer))
        print("-" * 30)
        if not self._layers:
            raise BuildNetworkError("Please provide layers before fitting data")
        if y.shape[1] != self._current_dimension:
            raise BuildNetworkError("Output layer's shape should be {}, {} found".format(
                self._current_dimension, y.shape[1]))

        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        if x_test is not None and y_test is not None:
            x_test = torch.from_numpy(np.asarray(x_test, dtype=np.float32))
            y_test = torch.from_numpy(np.asarray(y_test, dtype=np.float32))
        (x_train, x_test), (y_train, y_test) = self._split_data(
            x, y, x_test, y_test, train_only)
        train_len = len(x_train)
        batch_size = min(batch_size, train_len)
        do_random_batch = train_len > batch_size
        train_repeat = 1 if not do_random_batch else int(train_len / batch_size) + 1
        self._regularization_param = 1 - lb * lr / batch_size
        self._get_min_max(x_train, y_train)

        self._metrics = ["acc"] if metrics is None else metrics
        for i, metric in enumerate(self._metrics):
            if isinstance(metric, str):
                if metric not in self._available_metrics:
                    raise BuildNetworkError("Metric '{}' is not implemented".format(metric))
                self._metrics[i] = self._available_metrics[metric]
        self._metric_names = [_m.__name__ for _m in self._metrics]

        self._logs = {
            name: [[] for _ in range(len(self._metrics) + 1)] for name in ("train", "cv", "test")
        }
        if verbose is not None:
            self.verbose = verbose

        layer_width = len(self._layers)
        self._whether_apply_bias = apply_bias

        bar = ProgressBar(max_value=max(1, epoch // record_period), name="Epoch", start=False)
        if self.verbose >= NNVerbose.EPOCH:
            bar.start()
        img, ims = None, []

        if draw_weights:
            weight_trace = [[[org] for org in weight] for weight in self._weights]
        else:
            weight_trace = []

        *animation_properties, animation_params = self._get_animation_params(animation_params)
        sub_bar = ProgressBar(max_value=train_repeat * record_period - 1, name="Iteration", start=False)
        for counter in range(epoch):
            self._w_optimizer.update()
            self._b_optimizer.update()
            if self.verbose >= NNVerbose.ITER and counter % record_period == 0:
                sub_bar.start()
            for _ in range(train_repeat):
                if do_random_batch:
                    batch = torch.randperm(train_len)[:batch_size]
                    x_batch, y_batch = x_train[batch], y_train[batch]
                else:
                    x_batch, y_batch = x_train, y_train
                activations = self._get_activations(x_batch)

                deltas = [self._layers[-1].bp_first(y_batch, activations[-1])]
                for i in range(-1, -len(activations), -1):
                    deltas.append(self._layers[i - 1].bp(activations[i - 1], self._weights[i], deltas[-1]))

                for i in range(layer_width - 1, 0, -1):
                    if not isinstance(self._layers[i], SubLayer):
                        self._opt(i, activations[i - 1], deltas[layer_width - i - 1])
                self._opt(0, x_batch, deltas[-1])

                if draw_weights:
                    for i, weight in enumerate(self._weights):
                        for j, new_weight in enumerate(weight.copy()):
                            weight_trace[i][j].append(new_weight)
                if self.verbose >= NNVerbose.DEBUG:
                    pass
                if self.verbose >= NNVerbose.ITER:
                    if sub_bar.update() and self.verbose >= NNVerbose.METRICS_DETAIL:
                        self._append_log(x, y, "train", get_loss=show_loss)
                        self._append_log(x_test, y_test, "cv", get_loss=show_loss)
                        self._print_metric_logs(show_loss, "train")
                        self._print_metric_logs(show_loss, "cv")
            if self.verbose >= NNVerbose.ITER:
                sub_bar.update()
            self._handle_animation(
                counter, x, y, ims, animation_params, *animation_properties,
                img=self._draw_2d_network(**animation_params), name="Neural Network"
            )
            if do_log:
                self._append_log(x, y, "train", get_loss=show_loss)
                self._append_log(x_test, y_test, "cv", get_loss=show_loss)
            if (counter + 1) % record_period == 0:
                if do_log and self.verbose >= NNVerbose.METRICS:
                    self._print_metric_logs(show_loss, "train")
                    self._print_metric_logs(show_loss, "cv")
                if visualize:
                    if visualize_setting is None:
                        self.visualize2d(x_test, y_test)
                    else:
                        self.visualize2d(x_test, y_test, *visualize_setting)
                if self.verbose >= NNVerbose.EPOCH:
                    bar.update(counter // record_period + 1)
                    if self.verbose >= NNVerbose.ITER:
                        sub_bar = ProgressBar(max_value=train_repeat * record_period - 1, name="Iteration", start=False)

        if do_log:
            self._append_log(x_test, y_test, "test", get_loss=show_loss)
        if img is not None:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if draw_weights:
            ts = np.arange(epoch * train_repeat + 1)
            for i, weight in enumerate(self._weights):
                plt.figure()
                for j in range(len(weight)):
                    plt.plot(ts, weight_trace[i][j])
                plt.title("Weights toward layer {} ({})".format(i + 1, self._layers[i].name))
                plt.show()
        self._handle_mp4(ims, animation_properties, "NN")
        return self._logs

    # TODO
    @NNTiming.timeit(level=2, prefix="[API] ")
    def save(self, path=None, name=None, overwrite=True):
        path = os.path.join("Models", "Cache") if path is None else os.path.join("Models", path)
        name = "Model.nn" if name is None else name
        if not os.path.exists(path):
            os.makedirs(path)
        _dir = os.path.join(path, name)
        if not overwrite and os.path.isfile(_dir):
            _count = 1
            _new_dir = _dir + "({})".format(_count)
            while os.path.isfile(_new_dir):
                _count += 1
                _new_dir = _dir + "({})".format(_count)
            _dir = _new_dir
        print()
        print("=" * 60)
        print("Saving Model to {}...".format(_dir))
        print("-" * 60)
        with open(_dir, "wb") as file:
            pickle.dump({
                "structures": {
                    "_layer_names": self.layer_names,
                    "_layer_params": self._layer_params,
                    "_cost_layer": self._layers[-1].name,
                    "_next_dimension": self._current_dimension
                },
                "params": {
                    "_logs": self._logs,
                    "_metric_names": self._metric_names,
                    "_weights": self._weights,
                    "_bias": self._bias,
                    "_optimizer_name": self._optimizer_name,
                    "_w_optimizer": self._w_optimizer,
                    "_b_optimizer": self._b_optimizer,
                    "layer_special_params": self.layer_special_params,
                }
            }, file)
        print("Done")
        print("=" * 60)

    # TODO
    @NNTiming.timeit(level=2, prefix="[API] ")
    def load(self, path=os.path.join("Models", "Cache", "Model.nn")):
        self.initialize()
        try:
            with open(path, "rb") as file:
                dic = pickle.load(file)
                for key, value in dic["structures"].items():
                    setattr(self, key, value)
                self.build()
                for key, value in dic["params"].items():
                    setattr(self, key, value)
                self._init_optimizer()
                for i in range(len(self._metric_names) - 1, -1, -1):
                    name = self._metric_names[i]
                    if name not in self._available_metrics:
                        self._metric_names.pop(i)
                    else:
                        self._metrics.insert(0, self._available_metrics[name])
                print()
                print("=" * 30)
                print("Model restored")
                print("=" * 30)
                return dic
        except Exception as err:
            raise BuildNetworkError("Failed to load Network ({}), structure initialized".format(err))

    @NNTiming.timeit(level=4, prefix="[API] ")
    def predict(self, x, get_raw_results=False, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(np.asarray(x, dtype=np.float32))
        if len(x.size()) == 1:
            x = x.view(1, -1)
        y_pred = self._get_prediction(x).numpy()
        return y_pred if get_raw_results else np.argmax(y_pred, axis=1)

    def draw_results(self):
        metrics_log, loss_log = {}, {}
        for key, value in sorted(self._logs.items()):
            metrics_log[key], loss_log[key] = value[:-1], value[-1]

        for i, name in enumerate(sorted(self._metric_names)):
            plt.figure()
            plt.title("Metric Type: {}".format(name))
            for key, log in sorted(metrics_log.items()):
                if key == "test":
                    continue
                xs = np.arange(len(log[i])) + 1
                plt.plot(xs, log[i], label="Data Type: {}".format(key))
            plt.legend(loc=4)
            plt.show()
            plt.close()

        plt.figure()
        plt.title("Cost")
        for key, loss in sorted(loss_log.items()):
            if key == "test":
                continue
            xs = np.arange(len(loss)) + 1
            plt.plot(xs, loss, label="Data Type: {}".format(key))
        plt.legend()
        plt.show()

    @staticmethod
    def fuck_pycharm_warning():
        print(Axes3D.acorr)
