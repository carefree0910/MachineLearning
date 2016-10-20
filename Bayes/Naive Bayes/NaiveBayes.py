from math import pi, exp

from config import *

sqrt_pi = pi ** 0.5


def gaussian(x, mu, sigma):
    return exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sqrt_pi * sigma)


def gaussian_maximum_likelihood(category, n_category, dim, n_dim, mu=None, sigma=None):

    if mu is None:
        mu = [0 for _ in range(n_category)]
        for c in range(n_category):
            mu[c] = sum([_x[dim] for _x in category[c]]) / len(category[c])

    if sigma is None:
        sigma = [0 for _ in range(n_category)]
        for c in range(n_category):
            sigma[c] = sum([(_x[dim] - mu[c]) ** 2 for _x in category[c]]) / (len(category[c]) - 1)

    def func(_c):
        def sub(_x):
            return gaussian(_x, mu[_c], sigma[_c])
        return sub

    return [func(_c=c) for c in range(n_dim)]


def estimate(x, xy_zip, category, lb=1, func=None, discrete_data=None):
    """
    :param x:               input matrix     :  x = (x1, ..., xn)                       (xi is a vector)
    :param xy_zip:          list             :  list(zip(x, y))
           y:               target vector    :  y = (ω1, ..., ωn)                       (ωi = 1, ..., m)(i=1, ..., n)
    :param category:        list             :  [ [x that belongs to ωi] ]              (len(category) = m)
    :param lb:              lambda           :  lambda = 1 -> Laplace Smoothing
    :param func:            function         :  None or func(x, tar_category) = p(x|tar_category)
    :param discrete_data:   list             :  If ωi is discrete, discrete_data[i] = n_possibilities
                                                else, discrete_data[i] = None or pre_configured_function
    :return:                function         :  func(x, tar_category) = p(x|tar_category)
    """

    n_category = len(category)
    n_dim = len(discrete_data)

    # Prior probability
    p_category = [0 for _ in range(n_category)]
    for w, x_lst in enumerate(category):
        p_category[w] = (len(x_lst) + lb) / (len(x) + lb * n_category)

    if func is None:
        data = [None for _ in range(n_dim)]
        for dim, n_possibilities in enumerate(discrete_data):
            if not isinstance(n_possibilities, int):
                if n_possibilities is None:
                    data[dim] = gaussian_maximum_likelihood(category, n_category, dim, n_dim, MU[dim], SIGMA[dim])
                else:
                    data[dim] = n_possibilities(category, n_category, dim, n_dim)
            else:
                new_category = [[0 for _ in range(n_possibilities)] for _ in range(n_category)]
                for xx, yy in xy_zip:
                    new_category[yy][xx[dim]] += 1
                data[dim] = [[
                     (new_category[c][p] + lb) / (len(category[c]) + lb * n_possibilities)
                     for p in range(n_possibilities)] for c in range(n_category)
                ]

        def func(input_x, tar_category):
            rs = 1
            for d, _x in enumerate(input_x):
                if discrete_data[d] is None:
                    rs *= data[d][tar_category](_x)
                else:
                    rs *= data[d][tar_category][_x]
            return rs * p_category[tar_category]

    return func


def predict(x, func, categories):
    m_arg, m_possibility = 0, 0
    for i, category in enumerate(categories):
        p = func(x, category)
        if p > m_possibility:
            m_arg, m_possibility = i, p

    return m_arg
