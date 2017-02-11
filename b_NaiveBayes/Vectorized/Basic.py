import numpy as np
from math import pi

from Util.Timing import Timing
from Util.Bases import ClassifierBase
from Util.Metas import ClassifierMeta

sqrt_pi = (2 * pi) ** 0.5


class NBFunctions:

    @staticmethod
    def gaussian(x, mu, sigma):
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sqrt_pi * sigma)

    @staticmethod
    def gaussian_maximum_likelihood(labelled_x, n_category, dim):

        mu = [np.sum(
            labelled_x[c][dim]) / len(labelled_x[c][dim]) for c in range(n_category)]
        sigma = [np.sum(
            (labelled_x[c][dim] - mu[c]) ** 2) / len(labelled_x[c][dim]) for c in range(n_category)]

        def func(_c):
            def sub(_x):
                return NBFunctions.gaussian(_x, mu[_c], sigma[_c])
            return sub

        return [func(_c=c) for c in range(n_category)]


class NaiveBayes(ClassifierBase, metaclass=ClassifierMeta):
    NaiveBayesTiming = Timing()

    def __init__(self):
        self._x = self._y = None
        self._data = self._func = None
        self._n_possibilities = None
        self._labelled_x = self._label_zip = None
        self._cat_counter = self._con_counter = None
        self.label_dic = self._feat_dics = None

    def feed_data(self, x, y, sample_weight=None):
        pass

    def _feed_sample_weight(self, sample_weight=None):
        pass

    @NaiveBayesTiming.timeit(level=2, prefix="[API] ")
    def get_prior_probability(self, lb=1):
        return [(_c_num + lb) / (len(self._y) + lb * len(self._cat_counter))
                for _c_num in self._cat_counter]

    @NaiveBayesTiming.timeit(level=2, prefix="[API] ")
    def fit(self, x=None, y=None, sample_weight=None, lb=1):
        if x is not None and y is not None:
            self.feed_data(x, y, sample_weight)
        self._func = self._fit(lb)

    def _fit(self, lb):
        pass

    @NaiveBayesTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_result=False):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        else:
            x = [xx[:] for xx in x]
        x = self._transfer_x(x)
        m_arg, m_probability = np.zeros(len(x), dtype=np.int8), np.zeros(len(x))
        for i in range(len(self._cat_counter)):
            p = self._func(x, i)
            _mask = p > m_probability
            m_arg[_mask], m_probability[_mask] = i, p[_mask]
        if not get_raw_result:
            return np.array([self.label_dic[arg] for arg in m_arg])
        return m_probability

    def _transfer_x(self, x):
        return x
