import matplotlib.pyplot as plt

from i_Clustering.KMeans import KMeans
from Opt.Methods import *
from Opt.Functions import *
from Util.Util import DataUtil
from Util.Bases import ClassifierBase, RegressorBase


# Example1: Use GradientDescent & Wolfe to minimize l2_norm of (x^2 + 1)^2 using "Automatic Differentiation"
class Test(Function):
    def loss(self, x):
        return np.linalg.norm((x ** 2 + 1) ** 2)

func, method = Test(1024), GradientDescent
_x, f, n_iter, n_feva = method(func, Wolfe(func)).opt()
print(f, n_iter, n_feva)
print(np.linalg.norm(_x))


# Example2: Logistic Regression with LM & scipy's Strong Wolfe
# However this naive version will sometimes encounter overflow / underflow in exp
class LogisticRegression(Function):
    def __init__(self, n, x, y, **kwargs):
        super(LogisticRegression, self).__init__(n, **kwargs)
        self._x, self._y = np.atleast_2d(x), np.array(y)
        self._beta = self._pi = None
        self._dot_cache = self._exp_dot_cache = None

    @property
    def x0(self):
        """ Use some trick to initialize beta """
        x, y = self._x, self._y
        z = np.log((y + 0.5) / (1.5 - y))
        d = np.diag(6 / ((y + 1) * (2 - y)))
        self._beta = np.linalg.inv(x.T.dot(d).dot(x)).dot(x.T).dot(np.linalg.inv(d)).dot(z)
        return self._beta

    def refresh_cache(self, x, dtype="all"):
        self._beta = x
        self._dot_cache = self._x.dot(x)
        self._exp_dot_cache = np.exp(self._dot_cache)
        self._pi = self._exp_dot_cache / (1 + self._exp_dot_cache)

    def loss(self, x):
        return np.log(1 + self._exp_dot_cache).sum() - self._dot_cache.dot(self._y)

    def grad(self, x):
        return self._x.T.dot(self._pi - self._y)

    def hessian(self, x):
        grad = self._pi * (1 - self._pi)  # type: np.ndarray
        return self._x.T.dot(np.diag(grad)).dot(self._x)


class LR(ClassifierBase):
    def __init__(self, opt, line_search=None, **kwargs):
        super(LR, self).__init__(**kwargs)
        self._func = LogisticRegression
        self._beta = self._func_cache = None
        self._line_search = line_search
        self._opt_cache = None
        self._opt = opt

    def fit(self, x, y):
        self._func_cache = self._func(x.shape[1], x, y)
        line_search = None if self._line_search is None else self._line_search(self._func_cache)
        self._opt_cache = self._opt(self._func_cache, line_search)
        self._opt_cache.opt()
        self._beta = self._func_cache._beta

    def predict(self, x, get_raw_results=False, **kwargs):
        pi = 1 / (1 + np.exp(-np.atleast_2d(x).dot(self._beta)))
        if get_raw_results:
            return pi
        return (pi >= 0.5).astype(np.double)

data, labels = DataUtil.gen_two_clusters(one_hot=False)
lr = LR(LM)
lr.fit(data, labels)
lr.evaluate(data, labels)
lr["func_cache"].refresh_cache(lr["beta"])
print("Loss:", lr["func_cache"].loss(lr["beta"]))
lr.visualize2d(data, labels, dense=400)
# Draw Training Curve
plt.figure()
plt.plot(np.arange(len(lr["opt_cache"].log))+1, lr["opt_cache"].log)
plt.show()


# Example3: RBFN Regression with BFGS & Armijo using "Automatic Differentiation"
# "RBFN" represents "Radial Basis Function Network". Typically, we use Gaussian Function for this example
class RBFN(Function):
    def __init__(self, mat, y, mat_cv, y_cv, centers, x0=10, n=1, **kwargs):
        super(RBFN, self).__init__(n, **kwargs)
        self._mat_high_dim = np.atleast_2d(mat)[:, None, ...]
        self._mat_cv_high_dim = np.atleast_2d(mat_cv)[:, None, ...]
        self._y, self._y_cv = np.array(y), np.array(y_cv)
        self._centers = centers
        self._cache = None
        self._x0 = x0
        self.n = 1

    @property
    def x0(self):
        return np.array([self._x0], dtype=np.double)

    def refresh_cache(self, x, dtype="all"):
        k_mat = np.sum(np.exp(-(self._mat_high_dim - self._centers) ** 2 / x), axis=2)
        k_mat_cv = np.sum(np.exp(-(self._mat_cv_high_dim - self._centers) ** 2 / x), axis=2)
        self._cache = k_mat, k_mat_cv, np.linalg.lstsq(k_mat, self._y)[0]

    def loss(self, x):
        k_mat, k_mat_cv, alpha = self._cache
        return np.linalg.norm(k_mat_cv.dot(alpha) - self._y_cv)


class RBFNRegressor(RegressorBase):
    def __init__(self, opt, line_search=None, **kwargs):
        super(RBFNRegressor, self).__init__(**kwargs)
        self._func = RBFN
        self._sigma2 = self._centers = None
        self._line_search = line_search
        self._opt_cache = self._func_cache = None
        self._opt = opt

        self._params["n_centers"] = kwargs.get("n_centers", None)
        self._params["n_centers_rate"] = kwargs.get("n_centers_rate", 0.75)

    def rbf(self, x, s):
        return np.sum(np.exp(-(x[:, None, ...] - self._centers) ** 2 / s), axis=2)

    def fit(self, x, y, x_cv, y_cv, n_centers=None, n_centers_rate=None):
        if n_centers is None:
            if self._params["n_centers"] is not None:
                n_centers = int(self._params["n_centers"])
            else:
                if n_centers_rate is None:
                    n_centers_rate = self._params["n_centers_rate"]
                n_centers = int(n_centers_rate * len(x))
        k_means = KMeans(n_clusters=n_centers)
        k_means.fit(x)
        self._centers = k_means["centers"]
        self._func_cache = self._func(x, y, x_cv, y_cv, self._centers)
        line_search = None if self._line_search is None else self._line_search(self._func_cache)
        self._opt_cache = self._opt(self._func_cache, line_search)
        self._sigma2 = self._opt_cache.opt()[0]

    def predict(self, x, get_raw_results=False):
        alpha = self._func_cache._cache[-1]
        k_mat = self.rbf(x, self._sigma2)
        return k_mat.dot(alpha)

data = np.linspace(0, 20, 100)  # type: np.ndarray
labels = data ** 2 + np.random.rand(100) * 25
indices = np.random.permutation(len(data))
data, labels = data[..., None][indices], labels[indices]
rbfn = RBFNRegressor(BFGS, Armijo)
cv_idx, test_idx = 70, 85
rbfn.fit(data[:cv_idx], labels[:cv_idx], data[cv_idx:test_idx], labels[cv_idx:test_idx])
rbfn.visualize2d(data, labels, padding=1)


# Example4: Naive 3-Layer Neural Network with GradientDescent & Wolfe using "Automatic Differentiation"
# Structure: Input -> ReLU (12 units) -> Sigmoid + Cross Entropy (no bias)
class NNFunc(Function):
    def __init__(self, x, y, n_hidden=12, **kwargs):
        self._x, self._y = np.atleast_2d(x), np.array(y)
        if len(self._y.shape) == 1:
            self._y = np.array(self._y[..., None] == np.arange(np.max(self._y)), dtype=np.double)
        n_input, n_output = self._x.shape[1], self._y.shape[1]
        super(NNFunc, self).__init__(n_input * n_hidden + n_hidden * n_output, **kwargs)
        self._dims, self._cut_idx = (n_input, n_hidden, n_output), n_input * n_hidden
        self.mat1 = self.mat2 = None

    def loss(self, x):
        mat1 = x[:self._cut_idx].reshape(self._dims[:2])
        mat2 = x[self._cut_idx:].reshape(self._dims[1:])
        y_raw = np.maximum(0, self._x.dot(mat1)).dot(mat2)
        y_pred = 1 / (1 + np.exp(-y_raw))
        self.mat1, self.mat2 = mat1, mat2
        return np.average(
            -self._y * np.log(np.maximum(y_pred, 1e-12)) - (1 - self._y) * np.log(np.maximum(1 - y_pred, 1e-12)))


class NN(ClassifierBase):
    def __init__(self, opt, line_search=None, **kwargs):
        super(NN, self).__init__(**kwargs)
        self._func = NNFunc
        self._mat1 = self._mat2 = None
        self._line_search = line_search
        self._opt_cache = self._func_cache = None
        self._opt = opt

        self._params["n_hidden"] = kwargs.get("n_hidden", 24)
        self._params["use_scipy"] = kwargs.get("use_scipy", False)

    def fit(self, x, y, n_hidden=None, use_scipy=None):
        if n_hidden is None:
            n_hidden = self._params["n_hidden"]
        if use_scipy is None:
            use_scipy = self._params["use_scipy"]
        self._func_cache = self._func(x, y, n_hidden)
        if not use_scipy:
            line_search = None if self._line_search is None else self._line_search(self._func_cache)
            self._opt_cache = self._opt(self._func_cache, line_search)
        else:
            self._opt_cache = ScipyOpt(self._func_cache)
        self._opt_cache.opt()
        self._mat1, self._mat2 = self._func_cache.mat1, self._func_cache.mat2

    def predict(self, x, get_raw_results=False, **kwargs):
        y_raw = np.maximum(0, np.atleast_2d(x).dot(self._mat1)).dot(self._mat2)
        y_pred = 1 / (1 + np.exp(-y_raw))  # type: np.ndarray
        if get_raw_results:
            return y_pred
        return np.argmax(y_pred, axis=1)

data, labels = DataUtil.gen_xor()
nn = NN(GradientDescent, Wolfe)
nn.fit(data, labels)
nn.evaluate(data, labels)
nn.visualize2d(data, labels)
# Draw Training Curve
plt.figure()
plt.plot(np.arange(len(nn["opt_cache"].log))+1, nn["opt_cache"].log)
plt.show()
