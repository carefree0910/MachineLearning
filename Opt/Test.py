from Opt.Methods import *
from Opt.Functions import *


# Example: Use GradientDescent & Wolfe to minimize l2_norm of (x^2 + 1)^2
class Test(Function):
    def loss(self, x):
        return np.linalg.norm((x ** 2 + 1) ** 2)

func, method = Test(1024), GradientDescent
f, n_iter, n_feva = method(func, Wolfe(func)).opt()[1:]
print(f, n_iter, n_feva)
