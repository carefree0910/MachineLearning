import numpy as np
from math import pi, exp

sqrt_pi = pi ** 0.5

SKIP_FIRST = False


class Util:

    @staticmethod
    def data_cleaning(line):
        # line = line.replace('"', "")
        return list(map(lambda c: c.strip(), line.split(",")))

    @staticmethod
    def get_raw_data():
        x = []
        skip_first_flag = None
        with open("Data/data2.txt", "r") as file:
            for line in file:
                if skip_first_flag is None and SKIP_FIRST:
                    skip_first_flag = True
                    continue
                x.append(Util.data_cleaning(line))
        return x


class NBFunctions:

    @staticmethod
    def gaussian(x, mu, sigma):
        return exp(-(x - mu) ** 2 / (2 * sigma)) / (sqrt_pi * sigma ** 0.5)

    # noinspection PyTypeChecker
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


class NaiveBayes:

    def __init__(self):
        self._func = None
        self._tar_idx = None
        self._n_possibilities = None
        self._data_len = self._data_dim = None
        self._labelled_x = self._label_zip = None
        self._category = self._categories = self._category_info = None
        self._initialized = False

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)
        return

    def feed_data(self, data, tar_idx=-1):
        pass

    def feed_sample_weight(self, sample_weight=None):
        pass

    def get_prior_probability(self, lb=1):
        return [(_c_num + lb) / (self._data_len + lb * len(self._category))
                for _c_num in self._category]

    def fit(self, x=None, y=None, sample_weight=None, lb=1):
        if not self._initialized:
            self.feed_data(np.hstack((x, y[:, None])))
        if sample_weight is not None:
            self.feed_sample_weight(sample_weight)
        self._func = self._fit(lb)

    def _fit(self, lb):
        pass

    def predict_one(self, x, get_raw_result=False):
        m_arg, m_possibility = 0, 0
        for i in range(len(self._category)):
            p = self._func(x, i)
            if p > m_possibility:
                m_arg, m_possibility = i, p
        if not get_raw_result:
            return m_arg
        return m_possibility

    def predict(self, x, get_raw_result=False):
        return np.array([self.predict_one(xx, get_raw_result) for xx in x])


class MultinomialNB(NaiveBayes):

    def feed_data(self, data, tar_idx=-1):
        tar_idx = int(tar_idx)
        attr_len = len(data[0])
        if tar_idx < 0:
            tar_idx = max(0, attr_len + tar_idx)
        features = np.array(data).T
        features = [set(feat) for feat in features]
        categories = [{_l: i for i, _l in enumerate(feats)} for feats in features]
        x = [[categories[i][_l] for i, _l in enumerate(line)] for line in data]
        y = [xx.pop(tar_idx) for xx in x]
        for lst in (features, categories):
            lst.append(lst.pop(tar_idx))
        category = np.bincount(y)
        n_possibilities = [len(feats) for feats in features]
        n_possibilities.pop()

        x, y = np.array(x), np.array(y)
        labels = [y == value for value in range(len(category))]
        labelled_x = [x[ci].T for ci in labels]

        self._tar_idx = tar_idx
        self._data_len, self._data_dim = len(y), x.shape[1]
        self._labelled_x, self._label_zip = labelled_x, list(zip(labels, labelled_x))
        self._category, self._categories, self._n_possibilities = category, categories, n_possibilities
        self.feed_sample_weight()
        self._initialized = True

    def feed_sample_weight(self, sample_weight=None):
        self._category_info = []
        for dim, _p in enumerate(self._n_possibilities):
            if sample_weight is None:
                self._category_info.append([
                    np.bincount(xx[dim], minlength=_p) for xx in self._labelled_x])
            else:
                local_weight = sample_weight * len(sample_weight)
                self._category_info.append([
                    np.bincount(xx[dim], weights=local_weight[label], minlength=_p)
                    for label, xx in self._label_zip])

    def _fit(self, lb):
        n_category = len(self._category)
        n_dim = len(self._n_possibilities)
        p_category = self.get_prior_probability(lb)

        data = [None for _ in range(n_dim)]
        for dim, n_possibilities in enumerate(self._n_possibilities):
            data[dim] = [[
                (self._category_info[dim][c][p] + lb) / (self._category[c] + lb * n_possibilities)
                for p in range(n_possibilities)
            ] for c in range(n_category)]
        self._data = [np.array(dim_info) for dim_info in data]

        def func(input_x, tar_category):
            rs = 1
            for d, _x in enumerate(input_x):
                rs *= data[d][tar_category][_x]
            return rs * p_category[tar_category]

        return func

    def get_xy_from_data(self, data):
        x = data
        y = [self._categories[-1][xx.pop(self._tar_idx)] for xx in x]
        for i, line in enumerate(x):
            for j, char in enumerate(line):
                x[i][j] = self._categories[j][char]
        return x, y

    def estimate(self, data):
        x, y = self.get_xy_from_data(data)
        y_pred = self.predict(x)
        print("Acc: {:12.6} %".format(100 * np.sum(y_pred == y) / len(y)))


class GaussianNB(NaiveBayes):

    def feed_data(self, data, tar_idx=-1):
        tar_idx = int(tar_idx)
        attr_len = len(data[0])
        if tar_idx < 0:
            tar_idx = max(0, attr_len + tar_idx)
        x = [list(map(lambda c: float(c), line)) for line in data]
        y = [int(xx.pop(tar_idx)) for xx in x]
        labels = sorted(list(set(y)))
        categories = {label: i for i, label in enumerate(labels)}

        x = np.array(x)
        y = np.array([categories[yy] for yy in y])

        category = np.bincount(y)
        labels = [y == label for label in labels]
        labelled_x = [x[ci].T for ci in labels]

        self._tar_idx = tar_idx
        self._data_dim = x.shape[1]
        self._data_len = len(y)
        self._labelled_x, self._label_zip = labelled_x, labels
        self._category, self._categories = category, categories
        self.feed_sample_weight()
        self._initialized = True

    def feed_sample_weight(self, sample_weight=None):
        if sample_weight is not None:
            local_weight = sample_weight * len(sample_weight)
            for i, label in enumerate(self._label_zip):
                self._labelled_x[i] *= local_weight[label]

    def _fit(self, lb):
        n_category = len(self._category)
        p_category = self.get_prior_probability(lb)
        data = [
            NBFunctions.gaussian_maximum_likelihood(
                self._labelled_x, n_category, dim) for dim in range(self._data_dim)]
        self._data = data

        def func(input_x, tar_category):
            rs = 1
            for d, _x in enumerate(input_x):
                rs *= data[d][tar_category](_x)
            return rs * p_category[tar_category]

        return func

    def estimate(self, data=None):
        x = [list(map(lambda c: float(c), line)) for line in data]
        y = [self._categories[xx.pop(self._tar_idx)] for xx in x]
        y_pred = self.predict(x)
        print("Acc: {:12.6} %".format(100 * np.sum(y_pred == y) / len(y)))


class MergedNB(NaiveBayes):

    def __init__(self, whether_discrete):
        NaiveBayes.__init__(self)
        self._whether_discrete = np.array(whether_discrete)
        self._whether_continuous = ~self._whether_discrete
        self._multinomial, self._gaussian = MultinomialNB(), GaussianNB()
        self._discrete_data, self._continuous_data = None, None

    @property
    def data(self):
        discrete_data = self._discrete_data.copy()
        _categories = self._multinomial["categories"]
        for i, line in enumerate(discrete_data):
            for j, _l in enumerate(line):
                discrete_data[i][j] = _categories[j][_l]
        return np.hstack((self._continuous_data.astype(np.double), discrete_data.astype(np.int)))

    def feed_data(self, data, tar_idx=-1):
        data = np.array(data)
        self._discrete_data, self._continuous_data = (
            data[:, self._whether_discrete], data[:, self._whether_continuous])
        _y = np.array([list(dd).pop(tar_idx) for dd in self._discrete_data])
        self._multinomial.feed_data(self._discrete_data, tar_idx)
        _y_category = self._multinomial["categories"][np.sum(self._whether_discrete)-1]
        _y = np.array([_y_category[yy] for yy in _y])
        self._category = np.bincount(_y)
        self._gaussian.feed_data(np.hstack((self._continuous_data, _y[:, None])), tar_idx)
        self._initialized = True

    def feed_sample_weight(self, sample_weight=None):
        self._multinomial.feed_sample_weight(sample_weight)
        self._gaussian.feed_sample_weight(sample_weight)

    def _fit(self, lb):
        self._multinomial.fit()
        self._gaussian.fit()
        discrete_func, continuous_func = self._multinomial["func"], self._gaussian["func"]

        def func(input_x, tar_category):
            input_x = np.array(input_x)
            return discrete_func(
                input_x[self._whether_discrete[:-1]].astype(np.int), tar_category) * continuous_func(
                input_x[self._whether_continuous[:-1]], tar_category)

        return func

    def estimate(self, data=None):
        data = np.array(data)
        _d, _c = data[:, self._whether_discrete], data[:, self._whether_continuous]
        _categories = self._multinomial["categories"]
        for i, line in enumerate(_d):
            for j, char in enumerate(line):
                _d[i][j] = _categories[j][char]
        data = np.zeros(data.shape)
        data[:, self._whether_discrete] = _d.astype(np.int)
        data[:, self._whether_continuous] = _c.astype(np.double)
        y, y_pred = self.predict(data[:, range(data.shape[1]-1)]), data[:, -1]
        print("Acc: {:8.6} %".format(100 * np.sum(y == y_pred) / len(y)))
