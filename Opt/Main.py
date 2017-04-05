from Opt.Methods import *
from Opt.Functions import *

func, method = Torsion(100), LM
f, n_iter, n_feva = method(func, Wolfe(func)).opt()[1:]
print(f, n_iter, n_feva)
f, n_iter, n_feva = method(func).opt()[1:]
print(f, n_iter, n_feva)


def f(_x):
    func.refresh_cache(_x)
    return func.loss(_x), func.grad(_x)

result = optimize.minimize(f, func.x0, jac=True, method="L-BFGS-B", bounds=func.bounds)
print(result.fun, result.nit, result.nfev)

_ = input("Done")
