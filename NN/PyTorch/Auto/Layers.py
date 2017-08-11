import torch.nn.functional as F
from torch.autograd import Variable

from NN.Errors import *
from NN.PyTorch.Optimizers import *


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
        if self.is_sub_layer:
            if bias is None:
                return self._activate(x, predict)
            return self._activate(x + bias.expand_as(x), predict)
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


# Special Layer

class Dropout(SubLayer):
    def __init__(self, parent, shape, prob=0.5):
        if prob < 0 or prob >= 1:
            raise BuildLayerError("Keep probability of Dropout should be a positive float smaller than 1")
        SubLayer.__init__(self, parent, shape)
        self._prob = prob
        self.description = "(Keep prob: {})".format(prob)

    def get_params(self):
        return self._prob,

    def _activate(self, x, predict):
        return F.dropout(x, 1 - self._prob, not predict)


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
        self.gamma = Variable(torch.ones(self.shape[1]), requires_grad=True)
        self.beta = Variable(torch.ones(self.shape[1]), requires_grad=True)
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
            self.running_mean = Variable(torch.zeros(x.size()[1]))
            self.running_var = Variable(torch.zeros(x.size()[1]))
        if not predict:
            self.sample_mean = torch.mean(x, dim=0)
            self.sample_var = torch.var(x, dim=0)
            x_normalized = (x - self.sample_mean.expand_as(x)) / torch.sqrt(self.sample_var + self._eps).expand_as(x)
            self.x_cache, self.x_normalized_cache = x, x_normalized
            out = self.gamma.expand_as(x_normalized) * x_normalized + self.beta.expand_as(x_normalized)
            self.running_mean = self._momentum * self.running_mean + (1 - self._momentum) * self.sample_mean
            self.running_var = self._momentum * self.running_var + (1 - self._momentum) * self.sample_var
            if self.gamma.grad is not None and self.beta.grad is not None:
                self._g_optimizer.update()
                self._b_optimizer.update()
                self.gamma.data -= self._g_optimizer.run(0, self.gamma.grad.data)
                self.beta.data -= self._b_optimizer.run(0, self.beta.grad.data)
                self.gamma.grad.data.zero_()
                self.beta.grad.data.zero_()
        else:
            x_normalized = (x - self.running_mean.expand_as(x)) / torch.sqrt(self.running_var + self._eps).expand_as(x)
            out = self.gamma.expand_as(x) * x_normalized + self.beta.expand_as(x)
        return out


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
        "MSE": CostLayer, "CrossEntropy": CostLayer
    }
    available_sub_layers = {
        "Dropout", "Normalize"
    }
    available_cost_functions = {
        "MSE", "CrossEntropy"
    }
    available_special_layers = {
        "Dropout": Dropout,
        "Normalize": Normalize
    }
    special_layer_default_params = {
        "Dropout": (0.5, ),
        "Normalize": (0.001, 1e-8, 0.9)
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
