import tensorflow as tf

from Util.Timing import Timing


class Optimizers:

    OptTiming = Timing()

    def __init__(self, lr=1e-3):
        self._lr = lr
        self._opt = None

    @property
    def name(self):
        return str(self)

    def feed_timing(self, timing):
        if isinstance(timing, Timing):
            self.OptTiming = timing

    @OptTiming.timeit(level=1, prefix="[API] ")
    def minimize(self, x, *args, **kwargs):
        return self._opt.minimize(x, *args, **kwargs)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)


class SGD(Optimizers):

    def __init__(self, lr=1e-3):
        Optimizers.__init__(self, lr)
        self._opt = tf.train.GradientDescentOptimizer(self._lr)


class Momentum(Optimizers):

    def __init__(self, lr=1e-3, momentum=0.8):
        Optimizers.__init__(self, lr)
        self._opt = tf.train.MomentumOptimizer(self._lr, momentum)


class NAG(Optimizers):

    def __init__(self, lr=1e-3, momentum=0.8):
        Optimizers.__init__(self, lr)
        self._opt = tf.train.MomentumOptimizer(self._lr, momentum, use_nesterov=True)


class AdaDelta(Optimizers):

    def __init__(self, lr=1e-3, rho=0.95, eps=1e-8):
        Optimizers.__init__(self, lr)
        self._opt = tf.train.AdadeltaOptimizer(self._lr, rho, eps)


class AdaGrad(Optimizers):

    def __init__(self, lr=1e-3, init=0.1):
        Optimizers.__init__(self, lr)
        self._opt = tf.train.AdagradOptimizer(self._lr, init)


class Adam(Optimizers):

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        Optimizers.__init__(self, lr)
        self._opt = tf.train.AdamOptimizer(self._lr, beta1, beta2, eps)


class RMSProp(Optimizers):

    def __init__(self, lr=1e-3, decay=0.9, momentum=0.0, eps=1e-10):
        Optimizers.__init__(self, lr)
        self._opt = tf.train.RMSPropOptimizer(self._lr, decay, momentum, eps)


# Factory

class OptFactory:

    available_optimizers = {
        "MBGD": SGD, "Momentum": Momentum, "NAG": NAG,
        "AdaDelta": AdaDelta, "AdaGrad": AdaGrad,
        "Adam": Adam, "RMSProp": RMSProp
    }

    def get_optimizer_by_name(self, name, timing, lr, *args, **kwargs):
        try:
            _optimizer = self.available_optimizers[name](lr, *args, **kwargs)
            _optimizer.feed_timing(timing)
            return _optimizer
        except KeyError:
            raise NotImplementedError("Undefined Optimizer '{}' found".format(name))
