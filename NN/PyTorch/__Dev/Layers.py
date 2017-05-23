import torch.nn.functional as F

from NN.Errors import *
from NN.PyTorch.Auto.Optimizers import *

# TODO: Support 'SAME' padding


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
            x = x.view(x.size()[0], -1)
        if self.is_sub_layer:
            if bias is None:
                return self._activate(x, predict)
            return self._activate(x + bias, predict)
        mul = x.mm(w)
        if bias is None:
            return self._activate(mul, predict)
        return self._activate(x.mm(w) + bias.expand_as(mul), predict)

    @LayerTiming.timeit(level=1, prefix="[Core] ")
    def bp(self, y, w, prev_delta):
        if self.child is not None and isinstance(self.child, SubLayer):
            if not isinstance(self, SubLayer):
                return prev_delta
            return self._derivative(y, prev_delta)
        if isinstance(self, SubLayer):
            return self._derivative(y, prev_delta.mm(w.t()) * self._root.derivative(y))
        return prev_delta.mm(w.t()) * self._derivative(y)

    def _activate(self, x, predict):
        pass

    def _derivative(self, y, delta=None):
        pass

    # Util

    @staticmethod
    @LayerTiming.timeit(level=2, prefix="[Core Util] ")
    def safe_exp(y):
        return torch.exp(y - torch.max(y, dim=1)[0].expand_as(y))


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
            x_padded = F.pad(x, (p, p, p, p))

            height += 2 * p
            width += 2 * p


            return layer._activate(self, res.transpose(1, 0, 2, 3), predict)

        def activate(self, x, w, bias=None, predict=False):
            return self.LayerTiming.timeit(level=1, func_name="activate", cls_name=name, prefix="[Core] ")(
                _activate)(self, x, w, bias, predict)

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
                self.gamma, self.beta = np.ones(self.n_filters), np.zeros(self.n_filters)
                self.init_optimizers()

        def _activate(self, x, predict):
            n, n_channels, height, width = x.shape
            out = sub_layer._activate(self, x.transpose(0, 2, 3, 1).reshape(-1, n_channels), predict)
            return out.reshape(n, height, width, n_channels).transpose(0, 3, 1, 2)

        def _derivative(self, y, w, delta=None):
            if self.is_fc_base:
                delta = delta.dot(w.T).reshape(y.shape)
            n, n_channels, height, width = delta.shape
            # delta_new = delta.transpose(0, 2, 3, 1).reshape(-1, n_channels)
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
        return F.tanh(x)


class Sigmoid(Layer):
    def _activate(self, x, predict):
        return F.sigmoid(x)


class ELU(Layer):
    def _activate(self, x, predict):
        return F.elu(x)


class ReLU(Layer):
    def _activate(self, x, predict):
        return F.relu(x)


class Softplus(Layer):
    def _activate(self, x, predict):
        return F.softplus(x)


class Identical(Layer):
    def _activate(self, x, predict):
        return x


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
            out = np.zeros((n, n_channels, self.out_h, self.out_w))
            for i in range(n):
                for j in range(n_channels):
                    for k in range(self.out_h):
                        for l in range(self.out_w):
                            window = x[i, j, k * sd:pool_height + k * sd, l * sd:pool_width + l * sd]
                            out[i, j, k, l] = np.max(window)
            self._pool_cache["method"] = "original"
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
            mask = (x_reshaped_cache == out_newaxis)
            dout_newaxis = delta[..., None, :, None]
            dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
            dx_reshaped[mask] = dout_broadcast[mask]
            # noinspection PyTypeChecker
            dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
            dx = dx_reshaped.reshape(self.x_cache.shape)
        elif method == "original":
            sd = self._stride
            dx = np.zeros_like(self.x_cache)
            n, n_channels, *_ = self.x_cache.shape
            # noinspection PyTupleAssignmentBalance
            _, pool_height, pool_width = self._shape[1]
            for i in range(n):
                for j in range(n_channels):
                    for k in range(self.out_h):
                        for l in range(self.out_w):
                            window = self.x_cache[i, j, k*sd:pool_height+k*sd, l*sd:pool_width+l*sd]
                            # noinspection PyTypeChecker
                            dx[i, j, k*sd:pool_height+k*sd, l*sd:pool_width+l*sd] = (
                                window == np.max(window)) * delta[i, j, k, l]
        else:
            raise LayerError("Undefined pooling method '{}' found".format(method))
        return dx, None, None


# Special Layer

class Dropout(SubLayer):
    def __init__(self, parent, shape, prob=0.5):
        if prob < 0 or prob >= 1:
            raise BuildLayerError("Probability of Dropout should be a positive float smaller than 1")
        SubLayer.__init__(self, parent, shape)
        self._prob = prob
        self.description = "(Drop prob: {})".format(prob)

    def get_params(self):
        return self._prob,

    def _activate(self, x, predict):
        F.dropout(x, self._prob, not predict)


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
        self.gamma, self.beta = torch.ones(self.shape[1]), torch.zeros(self.shape[1])
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
            self.running_mean, self.running_var = torch.zeros(x.size()[1]), torch.zeros(x.size()[1])
        if not predict:
            self.sample_mean = torch.mean(x, dim=0)
            self.sample_var = torch.var(x, dim=0)
            x_normalized = (x - self.sample_mean) / torch.sqrt(self.sample_var + self._eps)
            self.x_cache, self.x_normalized_cache = x, x_normalized
            out = self.gamma * x_normalized + self.beta
            self.running_mean = self._momentum * self.running_mean + (1 - self._momentum) * self.sample_mean
            self.running_var = self._momentum * self.running_var + (1 - self._momentum) * self.sample_var
        else:
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self._eps)
            out = self.gamma * x_normalized + self.beta
        return out


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

    @property
    def calculate(self):
        return lambda y, y_pred: self._cost_function(y, y_pred)

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
    def _softmax(y):
        return F.log_softmax(y)

    @staticmethod
    def _sigmoid(y):
        return F.sigmoid(y)

    # Cost Functions

    @staticmethod
    def _mse(y, y_pred):
        return torch.nn.MSELoss()(y_pred, y)

    @staticmethod
    def _cross_entropy(y, y_pred):
        return F.cross_entropy(
            y_pred, torch.squeeze(torch.max(y, dim=1)[1], 1)
        )

    
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
        "MSE": CostLayer, "CrossEntropy": CostLayer
    }
    available_sub_layers = {
        "Dropout", "Normalize", "ConvNorm", "ConvDrop"
    }
    available_cost_functions = {
        "MSE", "CrossEntropy"
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
