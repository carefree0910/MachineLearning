from scipy.stats import multivariate_normal

from config import *


def gaussian_maximum_likelihood(x, category, p_category, mu=None, sigma=None):
    dim = len(x[0])

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
        return p_category[tar_category] * multivariate_normal(doc=False)(
            mu[tar_category], sigma[tar_category]
        ).pdf(input_x)

    return func


def estimate(x, y, category, lb=1, func=None, not_continuous=None):
    """
    :param x:               input vector     :  x = (x1, ..., xn)
    :param y:               dictionary       :  { x: ωi }                                 (ωi = 1, ..., m)
    :param category:        dictionary       :  { ωi: [x] }                               (len(category) = m)
    :param lb:              lambda           :  lambda = 1 -> Laplace Smoothing
    :param func:            function         :  None or func(x, tar_category) = p(x|tar_category)
    :param not_continuous:  whether the distribution is continuous or not.
                            if its value is not None, it should be a list: [ n_possibility ]
    :return:                function         :  func(x, tar_category) = p(x|tar_category)
    """

    n_category = len(category)

    # Prior probability
    p_category = {}
    for key, value in category.items():
        p_category[key] = (len(value) + lb) / (len(x) + lb * n_category)

    if func is None:
        if not_continuous is None:
            func = gaussian_maximum_likelihood(x, category, p_category, MU, SIGMA)
        else:
            data = []
            for dim in range(len(not_continuous)):
                new_category = {c: {p: 0 for p in not_continuous[dim]} for c in category}
                for key, value in y.items():
                    new_category[value][key[dim]] += 1
                data.append({
                    c: {p:
                            (new_category[c][p] + lb) / (len(category[c]) + lb * len(not_continuous[dim]))
                        for p in not_continuous[dim]
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
