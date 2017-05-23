import tensorflow as tf

from Util.Timing import Timing

# TODO: Customize Optimizer


class Optimizer:
    OptTiming = Timing()

    def __init__(self, lr=1e-3):
        self._lr = lr
        self._opt = None

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return str(self)

    @OptTiming.timeit(level=1, prefix="[API] ")
    def minimize(self, x, *args, **kwargs):
        return self._opt.minimize(x, *args, **kwargs)


class MBGD(Optimizer):
    def __init__(self, lr=1e-3):
        Optimizer.__init__(self, lr)
        self._opt = tf.train.GradientDescentOptimizer(self._lr)


class Momentum(Optimizer):
    def __init__(self, lr=1e-3, momentum=0.8):
        Optimizer.__init__(self, lr)
        self._opt = tf.train.MomentumOptimizer(self._lr, momentum)


class NAG(Optimizer):
    def __init__(self, lr=1e-3, momentum=0.8):
        Optimizer.__init__(self, lr)
        self._opt = tf.train.MomentumOptimizer(self._lr, momentum, use_nesterov=True)


class AdaDelta(Optimizer):
    def __init__(self, lr=1e-3, rho=0.95, eps=1e-8):
        Optimizer.__init__(self, lr)
        self._opt = tf.train.AdadeltaOptimizer(self._lr, rho, eps)


class AdaGrad(Optimizer):
    def __init__(self, lr=1e-3, init=0.1):
        Optimizer.__init__(self, lr)
        self._opt = tf.train.AdagradOptimizer(self._lr, init)


class Adam(Optimizer):
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        Optimizer.__init__(self, lr)
        self._opt = tf.train.AdamOptimizer(self._lr, beta1, beta2, eps)


class RMSProp(Optimizer):
    def __init__(self, lr=1e-3, decay=0.9, momentum=0.0, eps=1e-10):
        Optimizer.__init__(self, lr)
        self._opt = tf.train.RMSPropOptimizer(self._lr, decay, momentum, eps)


# Factory

class OptFactory:
    available_optimizers = {
        "MBGD": MBGD, "Momentum": Momentum, "NAG": NAG,
        "AdaDelta": AdaDelta, "AdaGrad": AdaGrad,
        "Adam": Adam, "RMSProp": RMSProp
    }

    def get_optimizer_by_name(self, name, lr, *args, **kwargs):
        try:
            optimizer = self.available_optimizers[name](lr, *args, **kwargs)
            return optimizer
        except KeyError:
            raise NotImplementedError("Undefined Optimizer '{}' found".format(name))
