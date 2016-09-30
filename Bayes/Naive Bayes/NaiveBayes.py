import scipy.stats as stat

from config import *
from CFunc.CFunc import *


def gaussian_maximum_likelihood(x, category, p_category,
                                mu=None, sigma=None, multivariate_normal=True, use_scipy_norm=False):
    dim = len(x[0])

    if multivariate_normal:
        if mu is None:
            mu = {c: [] for c in category}
            for c, lst in mu.items():
                for i in range(dim):
                    lst.append(sum([xx[i] for xx in category[c]]) / len(category[c]))
        if sigma is None:
            sigma = {c: [] for c in category}
            for c, lst in sigma.items():
                for i in range(dim):
                    lst.append([0] * dim)
                    lst[i] = sum([xx[i] ** 2 for xx in category[c]]) / len(category[c]) - mu[c][i] ** 2

        def func(input_x, tar_category):
            return p_category[tar_category] * stat.multivariate_normal(doc=False)(
                mu[tar_category], sigma[tar_category]
            ).pdf(input_x)
    else:
        if mu is None:
            mu = {c: [] for c in category}
            for c, lst in mu.items():
                for i in range(dim):
                    lst.append(sum([xx[i] for xx in category[c]]) / len(category[c]))
        if sigma is None:
            sigma = {c: [] for c in category}
            for c, lst in sigma.items():
                for i in range(dim):
                    lst.append(sum([xx[i] ** 2 for xx in category[c]]) / len(category[c]) - mu[c][i] ** 2)

        def func(input_x, tar_category):
            rs = p_category[tar_category]
            for i in range(dim):
                if use_scipy_norm:
                    rs *= stat.norm(mu[tar_category][i], sigma[tar_category][i]).pdf(input_x[i])
                else:
                    rs *= gaussian(input_x[i], mu[tar_category][i], sigma[tar_category][i])
            return rs

    return func


def estimate(x, y, category, lb=1, func=None, discrete=None):
    """
    :param x:               input vector     :  x = (x1, ..., xn)
    :param y:               dictionary       :  { x: ωi }                                 (ωi = 1, ..., m)
    :param category:        dictionary       :  { ωi: [x] }                               (len(category) = m)
    :param lb:              lambda           :  lambda = 1 -> Laplace Smoothing
    :param func:            function         :  None or func(x, tar_category) = p(x|tar_category)
    :param discrete:  whether the distribution is continuous or not.
                            if its value is not None, it should be a list: [ n_possibility ]
    :return:                function         :  func(x, tar_category) = p(x|tar_category)
    """

    n_category = len(category)

    # Prior probability
    p_category = {}
    for key, value in category.items():
        p_category[key] = (len(value) + lb) / (len(x) + lb * n_category)

    if func is None:
        if discrete is None:
            func = gaussian_maximum_likelihood(x, category, p_category, MU, SIGMA, MULTIVARIATE_NORMAL, USE_SCIPY_NORM)
        else:
            data = []
            for dim in range(len(discrete)):
                new_category = {c: {p: 0 for p in discrete[dim]} for c in category}
                for key, value in y.items():
                    new_category[value][key[dim]] += 1
                data.append({
                    c: {p:
                            (new_category[c][p] + lb) / (len(category[c]) + lb * len(discrete[dim]))
                        for p in discrete[dim]
                        } for c in new_category
                })

            def func(input_x, tar_category):
                rs = 1
                for d, xx in enumerate(input_x):
                    rs *= data[d][tar_category][xx]
                return rs * p_category[tar_category]

    return func


def predict(x, func, categories):
    m_arg = 0
    possibilities = []
    for i, category in enumerate(categories):
        p = func(x, category)
        possibilities.append(p)
        if p > possibilities[m_arg]:
            m_arg = i

    return m_arg


def draw_border_core_process(ii, jj, func, categories):
    tmp_x = (ii, jj)
    for ik, k in enumerate(categories):
        for il in range(ik + 1, len(categories)):
            if abs(func(tmp_x, k) - func(tmp_x, categories[il])) < EPSILON:
                return True
    return False
