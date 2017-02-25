import numpy as np
from abc import ABCMeta, abstractmethod

from Util.Timing import Timing


class Optimizers(metaclass=ABCMeta):

    OptTiming = Timing()

    def __init__(self, lr=0.01, cache=None):
        self.lr = lr
        self._cache = cache

    @property
    def name(self):
        return str(self)

    def feed_variables(self, variables):
        self._cache = [
            np.zeros(var.shape) for var in variables
        ]

    def feed_timing(self, timing):
        if isinstance(timing, Timing):
            self.OptTiming = timing

    @abstractmethod
    @OptTiming.timeit(level=1, prefix="[API] ")
    def run(self, i, dw):
        raise NotImplementedError("Please implement a 'run' method for your optimizer")

    @OptTiming.timeit(level=4, prefix="[API] ")
    def update(self):
        return self._update()

    @abstractmethod
    def _update(self):
        raise NotImplementedError("Please implement an 'update' method for your optimizer")

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)


class SGD(Optimizers):

    def run(self, i, dw):
        return self.lr * dw

    def _update(self):
        pass


class Momentum(Optimizers):

    def __init__(self, lr=0.01, cache=None, epoch=100, floor=0.5, ceiling=0.999):
        Optimizers.__init__(self, lr, cache)
        self._epoch, self._floor, self._ceiling = epoch, floor, ceiling
        self._step = (ceiling - floor) / epoch
        self._momentum = 0.5

    @property
    def epoch(self):
        return self._epoch

    @property
    def floor(self):
        return self._floor

    @property
    def ceiling(self):
        return self._ceiling

    def update_step(self):
        self._step = (self._ceiling - self._floor) / self._epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = value
        self.update_step()

    @floor.setter
    def floor(self, value):
        self._floor = value
        self.update_step()

    @ceiling.setter
    def ceiling(self, value):
        self._ceiling = value
        self.update_step()

    def run(self, i, dw):
        velocity = self._cache
        velocity[i] = velocity[i] * self._momentum + self.lr * dw
        return velocity[i]

    def _update(self):
        if self._momentum < self._ceiling:
            self._momentum += self._step


class NAG(Momentum):

    def run(self, i, dw):
        dw *= self.lr
        velocity = self._cache
        velocity[i] = self._momentum * velocity[i] + dw
        return self._momentum * velocity[i] + dw


class Adam(Optimizers):

    def __init__(self, lr=0.01, cache=None, beta1=0.9, beta2=0.999, eps=1e-8):
        Optimizers.__init__(self, lr, cache)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps

    def feed_variables(self, variables):
        self._cache = [
            [np.zeros(var.shape) for var in variables],
            [np.zeros(var.shape) for var in variables],
        ]

    def run(self, i, dw):
        self._cache[0][i] = self._cache[0][i] * self.beta1 + (1 - self.beta1) * dw
        self._cache[1][i] = self._cache[1][i] * self.beta2 + (1 - self.beta2) * (dw ** 2)
        return self.lr * self._cache[0][i] / (np.sqrt(self._cache[1][i] + self.eps))

    def _update(self):
        pass


class RMSProp(Optimizers):

    def __init__(self, lr=0.01, cache=None, decay_rate=0.9, eps=1e-8):
        Optimizers.__init__(self, lr, cache)
        self.decay_rate, self.eps = decay_rate, eps

    def run(self, i, dw):
        self._cache[i] = self._cache[i] * self.decay_rate + (1 - self.decay_rate) * dw ** 2
        return self.lr * dw / (np.sqrt(self._cache[i] + self.eps))

    def _update(self):
        pass


# Factory

class OptFactory:
    available_optimizers = {
        "MBGD": SGD, "Momentum": Momentum, "NAG": NAG, "Adam": Adam, "RMSProp": RMSProp
    }

    def get_optimizer_by_name(self, name, variables, timing, lr, epoch):
        try:
            _optimizer = self.available_optimizers[name](lr)
            if variables is not None:
                _optimizer.feed_variables(variables)
            _optimizer.feed_timing(timing)
            if epoch is not None and isinstance(_optimizer, Momentum):
                _optimizer.epoch = epoch
            return _optimizer
        except KeyError:
            raise NotImplementedError("Undefined Optimizer '{}' found".format(name))
