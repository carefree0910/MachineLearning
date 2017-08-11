import numba

from NN.Errors import *
from NN.Basic.Optimizers import *

# TODO: Support 'SAME' padding


@numba.jit([
    "void(int64, int64, int64, int64, float32[:,:,:,:],"
    "int64, int64, int64, float32[:,:,:,:], float32[:,:,:,:])"
], nopython=True)
def conv_bp(n, n_filters, out_h, out_w, dx_padded,
            filter_height, filter_width, sd, inner_weight, delta):
    for i in range(n):
        for f in range(n_filters):
            for j in range(out_h):
                for k in range(out_w):
                    for h in range(dx_padded.shape[1]):
                        jsd, ksd = j * sd, k * sd
                        for p in range(filter_height):
                            for q in range(filter_width):
                                dx_padded[i, h, jsd+p, ksd+q] += (
                                    inner_weight[f][h][p][q] * delta[i, f, j, k]
                                )


@numba.jit([
    "void(int64, int64, int64, int64, float32[:,:,:,:], float32[:,:,:,:],"
    "int64, int64, int64, int32[:,:,:,:,:])"
], nopython=True)
def max_pool(n, n_channels, out_h, out_w, x, out,
             pool_height, pool_width, sd, pos_cache):
    for i in range(n):
        for j in range(n_channels):
            for k in range(out_h):
                for l in range(out_w):
                    ksd, lsd = k * sd, l * sd
                    _max = x[i, j, ksd, lsd]
                    pos = (0, 0)
                    for p in range(pool_height):
                        for q in range(pool_width):
                            if x[i, j, ksd+p, lsd+q] > _max:
                                _max = x[i, j, ksd+p, lsd+q]
                                pos = (p, q)
                    pos_cache[i, j, k, l] = pos
                    out[i, j, k, l] = _max


@numba.jit([
    "void(int64, int64, int64, int64, int64, float32[:,:,:,:], float32[:,:,:,:], int32[:,:,:,:,:])"
], nopython=True)
def max_pool_bp(n, n_channels, out_h, out_w, sd, dx, delta, pos_cache):
    for i in range(n):
        for j in range(n_channels):
            for k in range(out_h):
                for l in range(out_w):
                    ksd, lsd = k * sd, l * sd
                    pos = pos_cache[i, j, k, l]
                    dx[i, j, ksd+pos[0], lsd+pos[1]] = delta[i, j, k, l]


# Abstract Layers

class Layer:
    LayerTiming = Timing()

    def __init__(self, shape):
        """
        :param shape: shape[0] = units of previous layer
                      shape[1] = units of current layer (self)
        """
        self._shape = shape
        self.parent = None
        self.child = None
        self.is_fc = False
        self.is_fc_base = False
        self.is_sub_layer = False
        self._last_sub_layer = None

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return str(self)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def params(self):
        return self._shape,

    @property
    def special_params(self):
        return

    def set_special_params(self, dic):
        for key, value in dic.items():
            setattr(self, key, value)

    @property
    def root(self):
        return self

    @root.setter
    def root(self, value):
        raise BuildLayerError("Setting Layer's root is not permitted")

    @property
    def last_sub_layer(self):
        child = self.child
        if not child:
            return None
        while child.child:
            child = child.child
        return child

    @last_sub_layer.setter
    def last_sub_layer(self, value):
            self._last_sub_layer = value

    # Core

    def derivative(self, y, delta=None):
        return self._derivative(y, delta)

    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def activate(self, x, w, bias=None, predict=False):
        if self.is_fc:
            x = x.reshape(x.shape[0], -1)
        if self.is_sub_layer:
            if bias is None:
                return self._activate(x, predict)
            return self._activate(x + bias, predict)
        if bias is None:
            return self._activate(x.dot(w), predict)
        return self._activate(x.dot(w) + bias, predict)

    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def bp(self, y, w, prev_delta):
        if self.child is not None and isinstance(self.child, SubLayer):
            if not isinstance(self, SubLayer):
                return prev_delta
            return self._derivative(y, prev_delta)
        if isinstance(self, SubLayer):
            return self._derivative(y, prev_delta.dot(w.T) * self._root.derivative(y))
        return prev_delta.dot(w.T) * self._derivative(y)

    def _activate(self, x, predict):
        pass

    def _derivative(self, y, delta=None):
        pass

    # Util

    @staticmethod
    @LayerTiming.timeit(level=2, prefix="[Core Util] ")
    def safe_exp(y):
        return np.exp(y - np.max(y, axis=1, keepdims=True))


class SubLayer(Layer):
    def __init__(self, parent, shape):
        super(SubLayer, self).__init__(shape)
        self.parent = parent
        parent.child = self
        self._root = None
        self.description = ""

    @property
    def root(self):
        parent = self.parent
        while parent.parent:
            parent = parent.parent
        return parent

    @root.setter
    def root(self, value):
        self._root = value

    def get_params(self):
        pass

    @property
    def params(self):
        return self.get_params()

    def _activate(self, x, predict):
        raise NotImplementedError("Please implement activation function for " + self.name)

    def _derivative(self, y, delta=None):
        raise NotImplementedError("Please implement derivative function for " + self.name)


class ConvLayer(Layer):
    LayerTiming = Timing()

    def __init__(self, shape, stride=1, padding=0, parent=None):
        """
        :param shape:    shape[0] = shape of previous layer           c x h x w
                         shape[1] = shape of current layer's weight   f x c x h x w
        :param stride:   stride
        :param padding:  zero-padding
        """
        if parent is not None:
            parent = parent.root if parent.is_sub_layer else parent
            shape = parent.shape
        Layer.__init__(self, shape)
        self._stride, self._padding = stride, padding
        if len(shape) == 1:
            self.n_channels = self.n_filters = self.out_h = self.out_w = None
        else:
            self.feed_shape(shape)
        self.x_cache = self.x_col_cache = None
        self.inner_weight = None

    def feed_shape(self, shape):
        self._shape = shape
        self.n_channels, height, width = shape[0]
        self.n_filters, filter_height, filter_width = shape[1]
        full_height, full_width = width + 2 * self._padding, height + 2 * self._padding
        if (
            (full_height - filter_height) % self._stride != 0 or
            (full_width - filter_width) % self._stride != 0
        ):
            raise BuildLayerError(
                "Weight shape does not work, "
                "shape: {} - stride: {} - padding: {} not compatible with {}".format(
                    self._shape[1][1:], self._stride, self._padding, (height, width)
                ))
        self.out_h = int((height + 2 * self._padding - filter_height) / self._stride) + 1
        self.out_w = int((width + 2 * self._padding - filter_width) / self._stride) + 1

    @property
    def params(self):
        return self._shape, self._stride, self._padding

    @property
    def stride(self):
        return self._stride

    @property
    def padding(self):
        return self._padding

    def _activate(self, x, predict):
        raise NotImplementedError("Please implement activation function for " + self.name)

    def _derivative(self, y, delta=None):
        raise NotImplementedError("Please implement derivative function for " + self.name)


class ConvPoolLayer(ConvLayer):
    LayerTiming = Timing()

    def __init__(self, shape, stride=1, padding=0):
        """
        :param shape:    shape[0] = shape of previous layer           c x h x w
                         shape[1] = shape of pool window              c x ph x pw
        :param stride:   stride
        :param padding:  zero-padding
        """
        ConvLayer.__init__(self, shape, stride, padding)
        self._pool_cache = {}

    @property
    def params(self):
        return (self._shape[0], self._shape[1][1:]), self._stride, self._padding

    def feed_shape(self, shape):
        if len(shape[1]) == 2:
            shape = (shape[0], (shape[0][0], *shape[1]))
        ConvLayer.feed_shape(self, shape)

    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def activate(self, x, w, bias=None, predict=False):
        return self._activate(x, w, bias, predict)

    def _activate(self, x, *args):
        raise NotImplementedError("Please implement activation function for " + self.name)

    def _derivative(self, y, *args):
        raise NotImplementedError("Please implement derivative function for " + self.name)

    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def bp(self, y, w, prev_delta):
        return self._derivative(y, w, prev_delta)


# noinspection PyProtectedMember
class ConvMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        conv_layer, layer = bases

        def __init__(self, shape, stride=1, padding=0):
            conv_layer.__init__(self, shape, stride, padding)

        def _activate(self, x, w, bias, predict):
            self.x_cache, self.inner_weight = x, w
            n, n_channels, height, width = x.shape
            n_filters, _, filter_height, filter_width = w.shape

            p, sd = self._padding, self._stride
            x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

            height += 2 * p
            width += 2 * p

            shape = (n_channels, filter_height, filter_width, n, self.out_h, self.out_w)
            strides = (height * width, width, 1, n_channels * height * width, sd * width, sd)
            strides = x.itemsize * np.asarray(strides)
            x_cols = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides).reshape(
                n_channels * filter_height * filter_width, n * self.out_h * self.out_w)
            self.x_col_cache = x_cols

            if bias is None:
                res = w.reshape(n_filters, -1).dot(x_cols)
            else:
                res = w.reshape(n_filters, -1).dot(x_cols) + bias.reshape(-1, 1)
            res.shape = (n_filters, n, self.out_h, self.out_w)
            return layer._activate(self, res.transpose(1, 0, 2, 3), predict)

        def _derivative(self, y, w, prev_delta):
            n = len(y)
            n_channels, height, width = self._shape[0]
            n_filters, filter_height, filter_width = self._shape[1]

            p, sd = self._padding, self._stride
            if isinstance(prev_delta, tuple):
                prev_delta = prev_delta[0]

            __derivative = self.LayerTiming.timeit(level=1, func_name="bp", cls_name=name, prefix="[Core] ")(
                layer._derivative)
            if self.is_fc_base:
                delta = __derivative(self, y) * prev_delta.dot(w.T).reshape(y.shape)
            else:
                delta = __derivative(self, y) * prev_delta

            dw = delta.transpose(1, 0, 2, 3).reshape(n_filters, -1).dot(
                self.x_col_cache.T).reshape(self.inner_weight.shape)
            db = np.sum(delta, axis=(0, 2, 3))

            n_filters, _, filter_height, filter_width = self.inner_weight.shape
            *_, out_h, out_w = delta.shape

            dx_padded = np.zeros((n, n_channels, height + 2 * p, width + 2 * p), dtype=np.float32)
            conv_bp(
                n, n_filters, out_h, out_w, dx_padded,
                filter_height, filter_width, sd, self.inner_weight, delta
            )
            dx = dx_padded[..., p:-p, p:-p] if p > 0 else dx_padded
            return dx, dw, db

        def activate(self, x, w, bias=None, predict=False):
            return self.LayerTiming.timeit(level=1, func_name="activate", cls_name=name, prefix="[Core] ")(
                _activate)(self, x, w, bias, predict)

        def bp(self, y, w, prev_delta):
            return self.LayerTiming.timeit(level=1, func_name="bp", cls_name=name, prefix="[Core] ")(
                _derivative)(self, y, w, prev_delta)

        for key, value in locals().items():
            if str(value).find("function") >= 0 or str(value).find("property"):
                attr[key] = value

        return type(name, bases, attr)


# noinspection PyProtectedMember
class ConvSubMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        conv_layer, sub_layer = bases

        def __init__(self, parent, shape, *_args, **_kwargs):
            conv_layer.__init__(self, None, parent=parent, **_kwargs)
            self.out_h, self.out_w = parent.out_h, parent.out_w
            sub_layer.__init__(self, parent, shape, *_args, **_kwargs)
            self._shape = ((shape[0][0], self.out_h, self.out_w), shape[0])
            if name == "ConvNorm":
                self.gamma = np.ones(self.n_filters, dtype=np.float32)
                self.beta = np.ones(self.n_filters, dtype=np.float32)
                self.init_optimizers()

        def _activate(self, x, predict):
            n, n_channels, height, width = x.shape
            out = sub_layer._activate(self, x.transpose(0, 2, 3, 1).reshape(-1, n_channels), predict)
            return out.reshape(n, height, width, n_channels).transpose(0, 3, 1, 2)

        def _derivative(self, y, w, delta=None):
            if self.is_fc_base:
                delta = delta.dot(w.T).reshape(y.shape)
            n, n_channels, height, width = delta.shape
            dx = sub_layer._derivative(self, y, delta.transpose(0, 2, 3, 1).reshape(-1, n_channels))
            return dx.reshape(n, height, width, n_channels).transpose(0, 3, 1, 2)

        # noinspection PyUnusedLocal
        def activate(self, x, w, bias=None, predict=False):
            return self.LayerTiming.timeit(level=1, func_name="activate", cls_name=name, prefix="[Core] ")(
                _activate)(self, x, predict)

        def bp(self, y, w, prev_delta):
            if isinstance(prev_delta, tuple):
                prev_delta = prev_delta[0]
            return self.LayerTiming.timeit(level=1, func_name="bp", cls_name=name, prefix="[Core] ")(
                _derivative)(self, y, w, prev_delta)

        @property
        def params(self):
            return sub_layer.get_params(self)

        for key, value in locals().items():
            if str(value).find("function") >= 0 or str(value).find("property"):
                attr[key] = value

        return type(name, bases, attr)


# Activation Layers

class Tanh(Layer):
    def _activate(self, x, predict):
        return np.tanh(x)

    def _derivative(self, y, delta=None):
        return 1 - y ** 2


class Sigmoid(Layer):
    def _activate(self, x, predict):
        return 1 / (1 + np.exp(-x))

    def _derivative(self, y, delta=None):
        return y * (1 - y)


class ELU(Layer):
    def _activate(self, x, predict):
        rs, mask = x.copy(), x < 0
        rs[mask] = np.exp(rs[mask]) - 1
        return rs

    def _derivative(self, y, delta=None):
        _rs, _indices = np.ones(y.shape), y < 0
        _rs[_indices] = y[_indices] + 1
        return _rs


class ReLU(Layer):
    def _activate(self, x, predict):
        return np.maximum(0, x)

    def _derivative(self, y, delta=None):
        return y > 0


class Softplus(Layer):
    def _activate(self, x, predict):
        return np.log(1 + np.exp(x))

    def _derivative(self, y, delta=None):
        return 1 - 1 / np.exp(y)


class Identical(Layer):
    def _activate(self, x, predict):
        return x

    def _derivative(self, y, delta=None):
        return 1


# Convolution Layers

class ConvTanh(ConvLayer, Tanh, metaclass=ConvMeta):
    pass


class ConvSigmoid(ConvLayer, Sigmoid, metaclass=ConvMeta):
    pass


class ConvELU(ConvLayer, ELU, metaclass=ConvMeta):
    pass


class ConvReLU(ConvLayer, ReLU, metaclass=ConvMeta):
    pass


class ConvSoftplus(ConvLayer, Softplus, metaclass=ConvMeta):
    pass


class ConvIdentical(ConvLayer, Identical, metaclass=ConvMeta):
    pass


# Pooling Layers

class MaxPool(ConvPoolLayer):
    def _activate(self, x, *args):
        self.x_cache = x
        sd = self._stride
        n, n_channels, height, width = x.shape
        # noinspection PyTupleAssignmentBalance
        _, pool_height, pool_width = self._shape[1]
        same_size = pool_height == pool_width == sd
        tiles = height % pool_height == 0 and width % pool_width == 0
        if same_size and tiles:
            x_reshaped = x.reshape(n, n_channels, int(height / pool_height), pool_height,
                                   int(width / pool_width), pool_width)
            self._pool_cache["x_reshaped"] = x_reshaped
            out = x_reshaped.max(axis=3).max(axis=4)
            self._pool_cache["method"] = "reshape"
        else:
            out = np.zeros((n, n_channels, self.out_h, self.out_w), dtype=np.float32)
            pos_cache = np.zeros((n, n_channels, self.out_h, self.out_w, 2), dtype=np.int32)
            max_pool(
                n, n_channels, self.out_h, self.out_w, x, out,
                pool_height, pool_width, sd, pos_cache
            )
            self._pool_cache["method"] = "original"
            self._pool_cache["pos_cache"] = pos_cache
        return out

    def _derivative(self, y, *args):
        w, prev_delta = args
        if isinstance(prev_delta, tuple):
            prev_delta = prev_delta[0]
        if self.is_fc_base:
            delta = prev_delta.dot(w.T).reshape(y.shape)
        else:
            delta = prev_delta
        method = self._pool_cache["method"]
        if method == "reshape":
            x_reshaped_cache = self._pool_cache["x_reshaped"]
            dx_reshaped = np.zeros_like(x_reshaped_cache)
            out_newaxis = y[..., None, :, None]
            mask = (x_reshaped_cache == out_newaxis)  # type: np.ndarray
            dout_newaxis = delta[..., None, :, None]
            dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
            dx_reshaped[mask] = dout_broadcast[mask]
            dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
            dx = dx_reshaped.reshape(self.x_cache.shape)
        elif method == "original":
            sd = self._stride
            dx = np.zeros_like(self.x_cache)
            n, n_channels, *_ = self.x_cache.shape
            max_pool_bp(
                n, n_channels, self.out_h, self.out_w, sd, dx, delta, self._pool_cache["pos_cache"]
            )
        else:
            raise LayerError("Undefined pooling method '{}' found".format(method))
        return dx, None, None


# Special Layer

class Dropout(SubLayer):
    def __init__(self, parent, shape, keep_prob=0.5):
        if keep_prob < 0 or keep_prob >= 1:
            raise BuildLayerError("Probability of Dropout should be a positive float smaller than 1")
        SubLayer.__init__(self, parent, shape)
        self._mask = None
        self._prob = keep_prob
        self._prob_inv = 1 / keep_prob
        self.description = "(Keep prob: {})".format(keep_prob)

    def get_params(self):
        return self._prob,

    def _activate(self, x, predict):
        if not predict:
            # noinspection PyTypeChecker
            self._mask = np.random.binomial(
                [np.ones(x.shape)], self._prob
            )[0].astype(np.float32) * self._prob_inv
            return x * self._mask
        return x

    def _derivative(self, y, delta=None):
        return delta * self._mask


class Normalize(SubLayer):
    def __init__(self, parent, shape, lr=0.001, eps=1e-8, momentum=0.9, optimizers=None):
        SubLayer.__init__(self, parent, shape)
        self.sample_mean, self.sample_var = None, None
        self.running_mean, self.running_var = None, None
        self.x_cache, self.x_normalized_cache = None, None
        self._lr, self._eps = lr, eps
        if optimizers is None:
            self._g_optimizer, self._b_optimizer = Adam(self._lr), Adam(self._lr)
        else:
            self._g_optimizer, self._b_optimizer = optimizers
        self.gamma = np.ones(self.shape[1], dtype=np.float32)
        self.beta = np.ones(self.shape[1], dtype=np.float32)
        self._momentum = momentum
        self.init_optimizers()
        self.description = "(lr: {}, eps: {}, momentum: {}, optimizer: ({}, {}))".format(
            lr, eps, momentum, self._g_optimizer.name, self._b_optimizer.name
        )

    def get_params(self):
        return self._lr, self._eps, self._momentum, (self._g_optimizer.name, self._b_optimizer.name)

    @property
    def special_params(self):
        return {
            "gamma": self.gamma, "beta": self.beta,
            "running_mean": self.running_mean, "running_var": self.running_var,
            "_g_optimizer": self._g_optimizer, "_b_optimizer": self._b_optimizer
        }

    def init_optimizers(self):
        _opt_fac = OptFactory()
        if not isinstance(self._g_optimizer, Optimizer):
            self._g_optimizer = _opt_fac.get_optimizer_by_name(
                self._g_optimizer, None, self._lr, None
            )
        if not isinstance(self._b_optimizer, Optimizer):
            self._b_optimizer = _opt_fac.get_optimizer_by_name(
                self._b_optimizer, None, self._lr, None
            )
        self._g_optimizer.feed_variables([self.gamma])
        self._b_optimizer.feed_variables([self.beta])

    # noinspection PyTypeChecker
    def _activate(self, x, predict):
        if self.running_mean is None or self.running_var is None:
            self.running_mean = np.zeros(x.shape[1], dtype=np.float32)
            self.running_var = np.zeros(x.shape[1], dtype=np.float32)
        if not predict:
            self.sample_mean = np.mean(x, axis=0, keepdims=True)
            self.sample_var = np.var(x, axis=0, keepdims=True)
            x_normalized = (x - self.sample_mean) / np.sqrt(self.sample_var + self._eps)
            self.x_cache, self.x_normalized_cache = x, x_normalized
            out = self.gamma * x_normalized + self.beta
            self.running_mean = self._momentum * self.running_mean + (1 - self._momentum) * self.sample_mean
            self.running_var = self._momentum * self.running_var + (1 - self._momentum) * self.sample_var
        else:
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self._eps)
            out = self.gamma * x_normalized + self.beta
        return out

    def _derivative(self, y, delta=None):
        n, d = self.x_cache.shape
        dx_normalized = delta * self.gamma
        x_mu = self.x_cache - self.sample_mean
        sample_std_inv = 1.0 / np.sqrt(self.sample_var + self._eps)
        ds_var = -0.5 * np.sum(dx_normalized * x_mu, axis=0, keepdims=True) * sample_std_inv ** 3
        ds_mean = (-1.0 * np.sum(dx_normalized * sample_std_inv, axis=0, keepdims=True) - 2.0 *
                   ds_var * np.mean(x_mu, axis=0, keepdims=True))
        dx1 = dx_normalized * sample_std_inv
        dx2 = 2.0 / n * ds_var * x_mu
        dx = dx1 + dx2 + 1.0 / n * ds_mean
        dg = np.sum(delta * self.x_normalized_cache, axis=0)
        db = np.sum(delta, axis=0)
        self.gamma += self._g_optimizer.run(0, dg)
        self.beta += self._b_optimizer.run(0, db)
        self._g_optimizer.update()
        self._b_optimizer.update()
        return dx


class ConvDrop(ConvLayer, Dropout, metaclass=ConvSubMeta):
    pass


class ConvNorm(ConvLayer, Normalize, metaclass=ConvSubMeta):
    pass


# Cost Layer

class CostLayer(Layer):
    def __init__(self, shape, cost_function="MSE", transform=None):
        super(CostLayer, self).__init__(shape)
        self._available_cost_functions = {
            "MSE": CostLayer._mse,
            "SVM": CostLayer._svm,
            "CrossEntropy": CostLayer._cross_entropy
        }
        self._available_transform_functions = {
            "Softmax": CostLayer._softmax,
            "Sigmoid": CostLayer._sigmoid
        }
        if cost_function not in self._available_cost_functions:
            raise LayerError("Cost function '{}' not implemented".format(cost_function))
        self._cost_function_name = cost_function
        self._cost_function = self._available_cost_functions[cost_function]
        if transform is None and cost_function == "CrossEntropy":
            self._transform = "Softmax"
            self._transform_function = CostLayer._softmax
        else:
            self._transform = transform
            self._transform_function = self._available_transform_functions.get(transform, None)

    def __str__(self):
        return self._cost_function_name

    def _activate(self, x, predict):
        if self._transform_function is None:
            return x
        return self._transform_function(x)

    def _derivative(self, y, delta=None):
        raise LayerError("derivative function should not be called in CostLayer")

    def bp_first(self, y, y_pred):
        if self._cost_function_name == "CrossEntropy" and (
                self._transform == "Softmax" or self._transform == "Sigmoid"):
            return y - y_pred
        dy = -self._cost_function(y, y_pred)
        if self._transform_function is None:
            return dy
        return dy * self._transform_function(y_pred, diff=True)

    @property
    def calculate(self):
        return lambda y, y_pred: self._cost_function(y, y_pred, False)

    @property
    def cost_function(self):
        return self._cost_function_name

    @cost_function.setter
    def cost_function(self, value):
        if value not in self._available_cost_functions:
            raise LayerError("'{}' is not implemented".format(value))
        self._cost_function_name = value
        self._cost_function = self._available_cost_functions[value]

    def set_cost_function_derivative(self, func, name=None):
        name = "Custom Cost Function" if name is None else name
        self._cost_function_name = name
        self._cost_function = func

    # Transform Functions

    @staticmethod
    def _softmax(y, diff=False):
        if diff:
            return y * (1 - y)
        exp_y = CostLayer.safe_exp(y)
        return exp_y / np.sum(exp_y, axis=1, keepdims=True)

    @staticmethod
    def _sigmoid(y, diff=False):
        if diff:
            return y * (1 - y)
        return 1 / (1 + np.exp(-y))

    # Cost Functions

    @staticmethod
    def _mse(y, y_pred, diff=True):
        if diff:
            return -y + y_pred
        return 0.5 * np.average((y - y_pred) ** 2)

    @staticmethod
    def _svm(y, y_pred, diff=True):
        n, y = y_pred.shape[0], np.argmax(y, axis=1)
        correct_class_scores = y_pred[np.arange(n), y]
        margins = np.maximum(0, y_pred - correct_class_scores[..., None] + 1.0)
        margins[np.arange(n), y] = 0
        loss = np.sum(margins) / n
        num_pos = np.sum(margins > 0, axis=1)
        if not diff:
            return loss
        dx = np.zeros_like(y_pred)
        dx[margins > 0] = 1
        dx[np.arange(n), y] -= num_pos
        dx /= n
        return dx

    @staticmethod
    def _cross_entropy(y, y_pred, diff=True):
        if diff:
            return -y / y_pred + (1 - y) / (1 - y_pred)
        # noinspection PyTypeChecker
        return np.average(-y * np.log(np.maximum(y_pred, 1e-12)) - (1 - y) * np.log(np.maximum(1 - y_pred, 1e-12)))

    
# Factory

class LayerFactory:
    available_root_layers = {
        "Tanh": Tanh, "Sigmoid": Sigmoid,
        "ELU": ELU, "ReLU": ReLU, "Softplus": Softplus,
        "Identical": Identical,
        "ConvTanh": ConvTanh, "ConvSigmoid": ConvSigmoid,
        "ConvELU": ConvELU, "ConvReLU": ConvReLU, "ConvSoftplus": ConvSoftplus,
        "ConvIdentical": ConvIdentical,
        "MaxPool": MaxPool,
        "MSE": CostLayer, "SVM": CostLayer, "CrossEntropy": CostLayer
    }
    available_sub_layers = {
        "Dropout", "Normalize", "ConvNorm", "ConvDrop"
    }
    available_cost_functions = {
        "MSE", "SVM", "CrossEntropy"
    }
    available_special_layers = {
        "Dropout": Dropout,
        "Normalize": Normalize,
        "ConvDrop": ConvDrop,
        "ConvNorm": ConvNorm
    }
    special_layer_default_params = {
        "Dropout": (0.5, ),
        "Normalize": (0.001, 1e-8, 0.9),
        "ConvDrop": (0.5, ),
        "ConvNorm": (0.001, 1e-8, 0.9)
    }

    def get_root_layer_by_name(self, name, *args, **kwargs):
        if name not in self.available_sub_layers:
            if name in self.available_root_layers:
                if name in self.available_cost_functions:
                    kwargs["cost_function"] = name
                name = self.available_root_layers[name]
            else:
                raise BuildNetworkError("Undefined layer '{}' found".format(name))
            return name(*args, **kwargs)
        return None
    
    def get_layer_by_name(self, name, parent, current_dimension, *args, **kwargs):
        _layer = self.get_root_layer_by_name(name, *args, **kwargs)
        if _layer:
            return _layer, None
        _current, _next = parent.shape[1], current_dimension
        layer_param = self.special_layer_default_params[name]
        _layer = self.available_special_layers[name]
        if args or kwargs:
            _layer = _layer(parent, (_current, _next), *args, **kwargs)
        else:
            _layer = _layer(parent, (_current, _next), *layer_param)
        return _layer, (_current, _next)
