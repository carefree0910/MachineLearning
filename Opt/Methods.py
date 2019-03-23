import random
import warnings
import numpy as np
try:
    from scipy import linalg, optimize
    from scipy.sparse import issparse, linalg as sparse_lin
    scipy_flag = True
except ImportError:
    linalg = optimize = issparse = sparse_lin = None
    scipy_flag = False

from Opt.Functions import Function
from Util.ProgressBar import ProgressBar

np.seterr(all="warn")


# Config

class LineSearchConfig:
    """ Default values for LineSearch parameters """
    method = "0.618"
    init = 0.5
    floor = 0.001
    epoch = 50
    rho = 1e-4
    sigma = 0.1
    gamma = 0.1
    eps = 1e-4
    t = 1.1


class OptConfig:
    """ Default values for Optimizer parameters """
    epoch = 1000
    eps = 1e-8
    nu = 0.1


# Line Search

class LineSearch:
    def __init__(self, func, **kwargs):
        """
        The framework for line search algorithms
        :param func   : Should be a SubClass of class 'Function' defined in Functions.py
        :param kwargs : May contain these parameters:
                init  : Average of random initial points   ; default: 0.5
                floor : Lower bound of α in Armijo         ; default: 0.001
                epoch : Maximum iteration for line search  ; default: 50
                rho   : ρ in all line search method        ; default: 1e-4
                sigma : σ in Wolfe, Strong Wolfe           ; default: 0.1
                gamma : Initial increment of parameter α   ; default: 0.1
                eps   : Lower bound of the interval length ; default: 1e-8
                t     : Boosting rate of γ                 ; default: 1.1
        """
        assert isinstance(func, Function)
        self._org_loss_cache = self._org_grad_cache = self._cache = None
        self._alpha = self._x = self._d = None
        self._func = func
        keys = [key for key in LineSearchConfig.__dict__ if "__" not in key]
        self._params = {key: kwargs.get(key, getattr(LineSearchConfig, key)) for key in keys}
        self.success_flag = [0, 0]
        self.counter = 0

    def min(self):
        """ Lower Bound Criterion """
        pass

    def max(self):
        """ Upper Bound Criterion """
        pass

    def func(self, alpha=None, diff=False):
        """
        Calculate the loss or the gradient
        :param alpha : Step size α
        :param diff  : If True, this function will return the gradient; If False, loss will be returned
        :return      : Loss or gradient
        """
        alpha = self._alpha if alpha is None else alpha
        (method, refresh, count) = (self._func.loss, "loss", 1) if not diff else (self._func.grad, "all", 0)
        self.counter += count
        if alpha == 0:
            return method(self._x)
        self._func.refresh_cache(self._x + alpha * self._d, dtype=refresh)
        return method(self._x + alpha * self._d)

    def step(self, x, d):
        """
        Perform a step of line search
        :param x      : Current x
        :param d      : Current direction
        :return       : α, feva, success, loss, gradient
                            where 'success' is a list with two elements:
                                success[0]: whether the initial search succeeded
                                success[1]: whether the line search criteria are satisfied
                            0: False; 1: True
        """
        self.counter = 0
        self.success_flag = [0, 0]
        self._alpha = 2 * self._params["init"] * random.random()
        self._x, self._d = x, d
        self._func.refresh_cache(self._x)
        self._org_loss_cache = self.func(0, False)
        self._org_grad_cache = self.func(0, True)
        a, b, fa, fb, self.success_flag[0] = self._get_init(self._params["gamma"], self._params["t"])
        fl = fr = None
        for _ in range(self._params["epoch"]):
            b_sub_a = b - a
            if b_sub_a < self._params["eps"]:
                return self._alpha, self.counter, 0, self.func(diff=False), self.func(diff=True)
            self._alpha = 0.5 * (a + b)
            if self._params["method"] == "poly":
                self._cache = self.func()
                c1 = (fb - fa) / b_sub_a
                c2 = ((self._cache - fa) / (self._alpha - 1) - c1) / (self._alpha - b)
                self._alpha = 0.5 * (a + b - c1 / c2)
            if self.min() and self.max():
                self.success_flag[1] = 1
                break
            self._func.refresh_cache(self._x)
            if self._params["method"] == "0.618":
                al = a + 0.382 * b_sub_a
                ar = a + 0.618 * b_sub_a
                if fl is None:
                    fl = self.func(al)
                if fr is None:
                    fr = self.func(ar)
                if fl < fr:
                    b, fl, fr = ar, None, fl
                else:
                    a, fl, fr = al, fr, None
            else:
                loss_cache = self.func()
                if loss_cache < self._cache:
                    a, fa = 0.5 * (a + b), self._cache
                else:
                    b, fb = self._alpha, loss_cache
        self._alpha = (a + b) / 2
        return self._alpha, self.counter, self.success_flag, self.func(diff=False), self.func(diff=True)

    def _init_core(self, gamma, current_f=None, next_f=None):
        """
        Core function used in _get_init method
        :param gamma     : Increment of α
        :param current_f : Current loss
        :param next_f    : 'Future' loss
        :return          : 'Future' α, current loss, 'Future' loss
        """
        next_alpha = self._alpha + gamma
        if current_f is None:
            current_f = self.func()
        if next_f is None:
            next_f = self.func(next_alpha)
        return next_alpha, current_f, next_f

    def _get_interval(self, next_alpha, current_f, next_f, success):
        """ Generate the target interval and corresponding loss and whether success """
        if self._alpha <= next_alpha:
            return self._alpha, next_alpha, current_f, next_f, int(success)
        return next_alpha, self._alpha, next_f, current_f, int(success)

    def _get_init(self, gamma, t):
        """
        Method to get the initial interval for line search
        :param gamma : Initial increment of α
        :param t     : Boosting rate of γ
        :return      : Target interval and corresponding loss and whether success
        """
        next_alpha, current_f, next_f = self._init_core(gamma)
        if next_f >= current_f:
            current_f, next_f = next_f, None
            self._alpha = next_alpha
            gamma *= -1
        for _ in range(self._params["epoch"]):
            next_alpha, current_f, next_f = self._init_core(gamma, current_f, next_f)
            if next_alpha <= 0:
                return self._get_interval(0, current_f, self._org_loss_cache, next_f >= current_f)
            if next_f >= current_f:
                return self._get_interval(next_alpha, current_f, next_f, 1)
            gamma *= t
            self._alpha = next_alpha
            current_f, next_f = next_f, None
        if next_f is None:
            next_f = self.func()
        return self._get_interval(next_alpha, current_f, next_f, 0)


class Armijo(LineSearch):
    def min(self):
        return self._alpha >= self._params["floor"]

    def max(self):
        return self.func() <= self._org_loss_cache + self._params["rho"] * (
            self._alpha * self._org_grad_cache.dot(self._d)
        )


class Goldstein(LineSearch):
    def min(self):
        self._cache = (self.func(), self._alpha * self._org_grad_cache.dot(self._d))
        return self._cache[0] <= self._org_loss_cache + self._cache[1] * (1 - self._params["rho"])

    def max(self):
        return self._cache[0] <= self._org_loss_cache + self._cache[1] * self._params["rho"]


class Wolfe(LineSearch):
    def min(self):
        self._cache = self._org_grad_cache.dot(self._d)
        return self.func(diff=True).dot(self._d) >= self._cache * self._params["sigma"]

    def max(self):
        return self.func() <= self._org_loss_cache + (
            self._cache * self._params["rho"] * self._alpha
        )


class StrongWolfe(LineSearch):
    def min(self):
        self._cache = self._org_grad_cache.dot(self._d)
        return np.abs(self.func(diff=True).dot(self._d)) <= -self._cache * self._params["sigma"]

    def max(self):
        return self.func() <= self._org_loss_cache + (
            self._cache * self._params["rho"] * self._alpha
        )


# Optimizer Frameworks

class Optimizer:
    def __init__(self, func, line_search=None, **kwargs):
        """
        The framework for general optimizers
        :param func        : Should be a SubClass of class 'Function' defined in Functions.py
        :param line_search : Should be a SubClass of class 'LineSearch' defined above
        :param kwargs      : May contain two parameters:
                     epoch : Maximum iteration for optimization  ; default: 1000
                     eps   : Tolerance                           ; default: 1e-8
        """
        assert isinstance(func, Function)
        assert line_search is None or isinstance(line_search, LineSearch)
        self._func = func
        self._search = line_search
        self._x, self._d, self._loss_cache, self._grad_cache = func.x0, None, None, None
        self.log = []
        self.feva = self.iter = 0
        self.success = np.zeros(2, dtype=np.int)
        self._params = {
            "epoch": kwargs.get("epoch", OptConfig.epoch),
            "eps": kwargs.get("eps", OptConfig.eps),
        }

    @staticmethod
    def solve(a, y, negative=True):
        """
        A wrapper for solving linear system 'ax = y' using:
            1) np.linalg.solve (if scipy is not available)
            2) cholesky decompose (if scipy is available and a is not sparse)
            3) spsolve (if scipy is available a is sparse)
        If scipy is available and matrix a is not sparse and is not positive definite, LinAlgError will be thrown
        :param a        : 'a'
        :param y        : 'y'
        :param negative : If True, we'll solve '-ax = y' instead
        :return         : 'x'
        """
        if scipy_flag:
            if issparse(a):
                if negative:
                    return sparse_lin.spsolve(-a, y)
                return sparse_lin.spsolve(a, y)
            l = linalg.cholesky(a, lower=True)
            if negative:
                z = linalg.solve_triangular(-l, y, lower=True)
            else:
                z = linalg.solve_triangular(l, y, lower=True)
            return linalg.solve_triangular(l.T, z)
        if negative:
            return np.linalg.solve(-a, y)
        return np.linalg.solve(a, y)

    def func(self, diff=0):
        """
        A wrapper of function calls
        :param diff : Should be 0, 1 or 2
                          0: return loss
                          1: return gradient
                          2: return hessian matrix
        :return     : Loss or gradient or hessian matrix
        """
        if diff == 0:
            self.feva += 1
            return self._func.loss(self._x)
        if diff == 1:
            return self._func.grad(self._x)
        return self._func.hessian(self._x)

    def get_d(self):
        """ Method that returns the opt direction. Should be overwritten by SubClasses"""
        return self._d

    def opt(self, epoch=None, eps=None):
        """
        Main procedure of opt
        :param epoch : Maximum iteration ; default: 1000
        :param eps   : Tolerance         ; default: 1e-8
        :return      : x*, f*, n_iter, feva
        """
        if epoch is None:
            epoch = self._params["epoch"]
        if eps is None:
            eps = self._params["eps"]
        self._func.refresh_cache(self._x)
        self._loss_cache, self._grad_cache = self.func(0), self.func(1)
        bar = ProgressBar(max_value=epoch, name="Opt")
        for _ in range(epoch):
            self.iter += 1
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:
                    if self._core(eps):
                        break
                    self.log.append(self._loss_cache)
                except RuntimeWarning as err:
                    print("\n", err, "\n")
                    break
                except np.linalg.linalg.LinAlgError as err:
                    print("\n", err, "\n")
                    break
            bar.update()
        bar.update()
        bar.terminate()
        return self._x, self._loss_cache, self.iter, self.feva

    def _core(self, eps):
        """ Core method for Optimizer, should return True if terminated """
        loss_cache = self._loss_cache
        self._x += self.get_d()
        self._func.refresh_cache(self._x)
        self._loss_cache = self.func(0)
        self._d = None
        if abs(loss_cache - self._loss_cache) < eps or np.linalg.norm(self._grad_cache) < eps:
            return True

    def _line_search_update(self):
        """ Proceed line search with x & d """
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                if self._search is not None:
                    alpha, feva, success_flag, self._loss_cache, self._grad_cache = self._search.step(
                        self._x, self.get_d())
                    self.success += success_flag
                elif not scipy_flag:
                    feva, alpha = 0, 1
                else:
                    def f(x):
                        self._func.refresh_cache(x, dtype="loss")
                        return self._func.loss(x)

                    def g(x):
                        self._func.refresh_cache(x)
                        return self._func.grad(x)

                    alpha, feva, _, self._loss_cache, old_f, self._grad_cache = optimize.line_search(
                        f, g, self._x, self.get_d()
                    )
            except RuntimeWarning:
                feva = 0
                if self._search is not None:
                    alpha = self._search._params["floor"]
                else:
                    alpha = 0.01
        self._x += alpha * self._d
        self.feva += feva
        self._d = None


class LineSearchOptimizer(Optimizer):
    """ Optimizer with line search """
    def _core(self, eps):
        loss_cache = self._loss_cache
        self._func.refresh_cache(self._x)
        self._line_search_update()
        if abs(loss_cache - self._loss_cache) < eps or np.linalg.norm(self._grad_cache) < eps:
            return True


# Gradient Descent Optimizers

class GradientDescent(LineSearchOptimizer):
    def get_d(self):
        if self._d is None:
            self._grad_cache = self.func(1)
            self._d = -self._grad_cache
        return self._d


# Newton Optimizers

class Newton(Optimizer):
    def get_d(self):
        if self._d is None:
            self._grad_cache = self.func(1)
            self._d = Optimizer.solve(self.func(2), self._grad_cache)
        return self._d


# Damped Newtons

class DampedNewton(Newton, LineSearchOptimizer):
    pass


class MergedNewton(DampedNewton):
    def __init__(self, func, line_search=None, **kwargs):
        super(MergedNewton, self).__init__(func, line_search)
        self._eps1 = kwargs.get("eps1", 0.75)
        self._eps2 = kwargs.get("eps2", 0.1)

    # noinspection PyTypeChecker
    def get_d(self):
        if self._d is not None:
            return self._d
        if self._grad_cache is None:
            self._grad_cache = self.func(1)
        try:
            self._d = Optimizer.solve(self.func(2), self.func(1))
            inner_prod, norm_prod = self._grad_cache.dot(self._d), np.linalg.norm(
                self._grad_cache
            ) * np.linalg.norm(self._d)
            if inner_prod > self._eps1 * norm_prod:
                self._d *= -1
            elif abs(inner_prod) <= self._eps2 * norm_prod:
                self._d = -self._grad_cache
        except np.linalg.linalg.LinAlgError:
            self._d = -self._grad_cache
        return self._d


class LM(DampedNewton):
    def __init__(self, func, line_search=None, **kwargs):
        super(LM, self).__init__(func, line_search)
        self._nu = kwargs.get("nu", OptConfig.nu)

    def get_d(self):
        if self._d is None:
            nu = .0
            while 1:
                try:
                    if nu == 0:
                        hessian = self.func(2)
                    else:
                        hessian = self.func(2) + np.diag(np.full(self._func.n, nu))
                    self._d = Optimizer.solve(hessian, self.func(1))
                    break
                except np.linalg.linalg.LinAlgError:
                    if nu == 0:
                        nu = self._nu
                    else:
                        nu *= 2
        return self._d


# Quasi Newtons

class QuasiNewton(DampedNewton):
    def __init__(self, func, line_search=None, method="H"):
        """ self._mat represents 'B' or 'H' depends on 'method' parameter """
        super(QuasiNewton, self).__init__(func, line_search)
        self._method, self._mat = method.upper(), None
        self._s = self._y = None

    def get_d(self):
        if self._d is None:
            if self._method == "B":
                self._d = Optimizer.solve(self._mat, self.func(1))
            else:
                self._d = -self._mat.dot(self.func(1))
        return self._d

    def _core(self, eps):
        if self._mat is None:
            self._mat = np.eye(self._func.n)
        if self._s is None and self._y is None:
            self._s, self._y = -self._x, -self._grad_cache
        loss_cache = self._loss_cache
        super(QuasiNewton, self)._core(eps)
        self._s += self._x
        self._y += self._grad_cache
        self._update_mat()
        self._s, self._y = -self._x, -self._grad_cache
        if abs(loss_cache - self._loss_cache) < eps or np.linalg.norm(self._grad_cache) < eps:
            return True

    def _update_mat(self):
        """ Core method for Quasi Newtons, matrix update algorithm should be defined here """
        pass


class SR1(QuasiNewton):
    def _update_mat(self):
        if self._method == "B":
            cache = (
                self._s - self._mat.dot(self._y),
                self._y
            )
        else:
            cache = (
                self._y - self._mat.dot(self._s),
                self._s
            )
        self._mat += cache[0][..., None].dot(cache[0][None, ...]) / cache[0].dot(cache[1])


class DFP(QuasiNewton):
    def _update_mat(self):
        if self._method == "B":
            cache1 = self._s.dot(self._mat)
            cache2 = self._s.dot(self._y)
            y1, y2 = self._y[..., None], self._y[None, ...]
            self._mat += (1 + cache1.dot(self._s) / cache2) * y1.dot(y2) / cache2 - (
                (y1.dot(cache1[None, ...]) + cache1[..., None].dot(y2)) / cache2
            )
        else:
            cache = self._y.dot(self._mat)
            self._mat += self._s[..., None].dot(self._s[None, ...]) / self._s.dot(self._y) - (
                cache[..., None].dot(cache[None, ...])
            ) / cache.dot(self._y)


class BFGS(DFP):
    def _update_mat(self):
        self._method = "H" if self._method == "B" else "B"
        self._s, self._y = self._y, self._s
        super(BFGS, self)._update_mat()
        self._s, self._y = self._y, self._s
        self._method = "H" if self._method == "B" else "B"


# Scipy Optimizer

class ScipyOpt:
    def __init__(self, func):
        self._func = func

    def opt(self):
        if not scipy_flag:
            raise ValueError("Scipy is not available")

        def f(_x):
            self._func.refresh_cache(_x)
            return self._func.loss(_x), self._func.grad(_x)
        result = optimize.minimize(f, self._func.x0, jac=True, method="L-BFGS-B", bounds=self._func.bounds)
        return result.x, result.fun, result.nit, result.nfev


__all__ = [
    "Armijo", "Goldstein", "Wolfe", "StrongWolfe",
    "GradientDescent", "Newton", "DampedNewton", "MergedNewton", "LM",
    "SR1", "DFP", "BFGS", "ScipyOpt"
]
