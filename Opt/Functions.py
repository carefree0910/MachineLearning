import numpy as np


class Function:
    """ The framework of target functions """
    def __init__(self, n, **kwargs):
        """
        :param n      : Dimension of input vector 
        :param kwargs : May contain 'grad_eps' which will be used to calculate numerical gradient
        """
        self.n = n
        self._params = {
            "grad_eps": kwargs.get("grad_eps", 1e-8)
        }

    def refresh_cache(self, x, dtype="all"):
        """
        Refresh cache if needed 
        :param x     : input vector
        :param dtype : control what kind of cache will be refreshed
        :return      : None
        """
        pass

    @property
    def x0(self):
        """ Should return the initial point """
        return np.random.random(self.n)

    @property
    def bounds(self):
        """ Should return the bounds. Format:
            [[min, max],
             [min, max],
             ...,
             [min, max]]
        """
        return

    def grad(self, x):
        """ Should return the gradient, support automatic differentiation """
        eps_base = np.random.random() * 1e-8
        grad_eps = []
        for i in range(len(x)):
            eps = np.zeros(len(x))
            eps[i] = eps_base
            x1 = x - 2 * eps; self.refresh_cache(x1, dtype="loss"); loss1 = self.loss(x1)  # type: float
            x2 = x - eps; self.refresh_cache(x2, dtype="loss"); loss2 = self.loss(x2)      # type: float
            x3 = x + eps; self.refresh_cache(x3, dtype="loss"); loss3 = self.loss(x3)      # type: float
            x4 = x + 2 * eps; self.refresh_cache(x4, dtype="loss"); loss4 = self.loss(x4)  # type: float
            grad_eps.append(1 / (12 * eps_base) * (loss1 - 8 * loss2 + 8 * loss3 - loss4))
        return np.array(grad_eps)

    def hessian(self, x):
        """ Should return the hessian matrix, support automatic differentiation """
        eps_base = np.random.random() * 1e-8
        grad_eps = []
        for i in range(len(x)):
            eps = np.zeros(len(x))
            eps[i] = eps_base
            x1 = x - 2 * eps; self.refresh_cache(x1); grad1 = self.grad(x1)  # type: np.ndarray
            x2 = x - eps; self.refresh_cache(x2); grad2 = self.grad(x2)      # type: np.ndarray
            x3 = x + eps; self.refresh_cache(x3); grad3 = self.grad(x3)      # type: np.ndarray
            x4 = x + 2 * eps; self.refresh_cache(x4); grad4 = self.grad(x4)  # type: np.ndarray
            grad_eps.append(1 / (12 * eps_base) * (grad1 - 8 * grad2 + 8 * grad3 - grad4))
        return np.array(grad_eps)

    def loss(self, x):
        """ Should return the loss """
        pass
