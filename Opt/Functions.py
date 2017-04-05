import numpy as np
import scipy.sparse as ss


class Function:
    """ The framework of target functions """
    def __init__(self, n, **kwargs):
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
        return

    @property
    def bounds(self):
        """ Should return the bounds """
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


class Torsion(Function):
    def __init__(self, n=100, c=25, **kwargs):
        super(Torsion, self).__init__(n, **kwargs)
        self.n, self._h2 = n ** 2, 1 / (n + 1) ** 2
        _a_tilde = ss.diags([-1, 4, -1], [-1, 0, 1], shape=(n, n), format="csr")
        _negative_In = -ss.eye(n)
        rows = [[None] * n for _ in range(n)]
        rows[0][0], rows[0][1] = _a_tilde, _negative_In
        for i in range(1, n - 1):
            rows[i][i] = _a_tilde
            rows[i][i-1] = rows[i][i+1] = _negative_In
        rows[n-1][n-2], rows[n-1][n-1] = _negative_In, _a_tilde
        self._a = ss.bmat(rows, format="csr")
        self._b = np.full(self.n, c * self._h2)
        self._bounds = [
            np.hstack([np.arange(i), np.full(n-i*2, i), np.arange(i)[::-1]]) for i in range(int(n*0.5))
        ]
        if n & 1:
            self._bounds = np.hstack(self._bounds + [
                np.hstack((np.arange(int(n * 0.5)), [int(n * 0.5)], np.arange(int(n * 0.5))[::-1]))
            ] + self._bounds[::-1])
        else:
            self._bounds = np.hstack(self._bounds + self._bounds[::-1]).astype(np.float)
        self._bounds /= (n + 1)

    @property
    def x0(self):
        x0 = np.random.random(self.n)
        indices = x0 > self._bounds
        x0[indices] = self._bounds[indices] * np.random.random(indices.sum())
        return x0

    @property
    def bounds(self):
        return np.c_[-self._bounds, self._bounds]

    def grad(self, x):
        return self._a.dot(x) - self._b

    def hessian(self, x):
        return self._a

    def loss(self, x):
        return 0.5 * self._a.dot(x).dot(x) - self._b.dot(x)
