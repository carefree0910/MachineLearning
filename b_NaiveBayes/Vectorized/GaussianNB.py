import matplotlib.pyplot as plt

from b_NaiveBayes.Vectorized.Basic import *


class GaussianNB(NaiveBayes):

    def feed_data(self, x, y, sample_weights=None):
        x = np.array([list(map(lambda c: float(c), line)) for line in x])
        labels = list(set(y))
        label_dic = {label: i for i, label in enumerate(labels)}
        y = np.array([label_dic[yy] for yy in y])
        cat_counter = np.bincount(y)
        labels = [y == value for value in range(len(cat_counter))]
        labelled_x = [x[label].T for label in labels]

        self._x, self._y = x.T, y
        self._labelled_x, self._label_zip = labelled_x, labels
        self._cat_counter, self.label_dic = cat_counter, {i: _l for _l, i in label_dic.items()}
        self.feed_sample_weights(sample_weights)

    def feed_sample_weights(self, sample_weights=None):
        if sample_weights is not None:
            local_weights = sample_weights * len(sample_weights)
            for i, label in enumerate(self._label_zip):
                self._labelled_x[i] *= local_weights[label]

    def _fit(self, lb):
        lb = 0
        n_category = len(self._cat_counter)
        p_category = self.get_prior_probability(lb)
        data = [
            NBFunctions.gaussian_maximum_likelihood(
                self._labelled_x, n_category, dim) for dim in range(len(self._x))]
        self._data = data

        def func(input_x, tar_category):
            input_x = np.atleast_2d(input_x).T
            rs = np.ones(input_x.shape[1])
            for d, xx in enumerate(input_x):
                rs *= data[d][tar_category](xx)
            return rs * p_category[tar_category]

        return func

    def visualize(self, save=False):
        colors = plt.cm.Paired([i / len(self.label_dic) for i in range(len(self.label_dic))])
        colors = {_cat: _color for _cat, _color in zip(self.label_dic.values(), colors)}
        for j in range(len(self._x)):
            tmp_data = self._x[j]
            x_min, x_max = np.min(tmp_data), np.max(tmp_data)
            gap = x_max - x_min
            tmp_x = np.linspace(x_min-0.1*gap, x_max+0.1*gap, 200)
            title = "$j = {}$".format(j + 1)
            plt.figure()
            plt.title(title)
            for c in range(len(self.label_dic)):
                plt.plot(tmp_x, self._data[j][c](tmp_x),
                         c=colors[self.label_dic[c]], label="class: {}".format(self.label_dic[c]))
            plt.xlim(x_min-0.2*gap, x_max+0.2*gap)
            plt.legend()
            if not save:
                plt.show()
            else:
                plt.savefig("d{}".format(j + 1))
