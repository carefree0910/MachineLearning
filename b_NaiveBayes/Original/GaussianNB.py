from b_NaiveBayes.Original.Basic import *
from b_NaiveBayes.Original.MultinomialNB import MultinomialNB


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
        n_category = len(self._cat_counter)
        p_category = self.get_prior_probability(lb)
        data = [
            NBFunctions.gaussian_maximum_likelihood(
                self._labelled_x, n_category, dim) for dim in range(len(self._x))]
        self._data = data

        def func(input_x, tar_category):
            rs = 1
            for d, xx in enumerate(input_x):
                rs *= data[d][tar_category](xx)
            return rs * p_category[tar_category]

        return func

if __name__ == '__main__':
    import time

    _data = []
    with open("../Data/data.txt", "r") as file:
        for _line in file:
            _data.append(_line.strip().split(","))
    np.random.shuffle(_data)
    train_num = 6000
    xs = _data
    ys = [xx.pop(0) for xx in xs]

    nb = MultinomialNB()
    nb.feed_data(xs, ys)
    xs, ys = nb["x"].tolist(), nb["y"].tolist()

    train_x, test_x = xs[:train_num], xs[train_num:]
    train_y, test_y = ys[:train_num], ys[train_num:]

    train_num = 6000
    train_data = _data[:train_num]
    test_data = _data[train_num:]

    learning_time = time.time()
    nb = GaussianNB()
    nb.fit(train_x, train_y)
    learning_time = time.time() - learning_time

    estimation_time = time.time()
    nb.estimate(train_x, train_y)
    nb.estimate(test_x, test_y)
    estimation_time = time.time() - estimation_time

    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            learning_time, estimation_time,
            learning_time + estimation_time
        )
    )
