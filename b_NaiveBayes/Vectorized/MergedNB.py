from b_NaiveBayes.Vectorized.Basic import *
from b_NaiveBayes.Vectorized.MultinomialNB import MultinomialNB
from b_NaiveBayes.Vectorized.GaussianNB import GaussianNB
from Util import DataUtil


class MergedNB(NaiveBayes):

    def __init__(self, whether_continuous=None):
        NaiveBayes.__init__(self)
        self._multinomial, self._gaussian = MultinomialNB(), GaussianNB()
        if whether_continuous is None:
            self._whether_discrete = self._whether_continuous = None
        else:
            self._whether_continuous = np.array(whether_continuous)
            self._whether_discrete = ~self._whether_continuous

    def feed_data(self, x, y, sample_weights=None):
        if sample_weights is not None:
            sample_weights = np.array(sample_weights)
        x, y, wc, features, feat_dics, label_dic = DataUtil.quantize_data(
            x, y, wc=self._whether_continuous, separate=True)
        if self._whether_continuous is None:
            self._whether_continuous = wc
            self._whether_discrete = ~self._whether_continuous
        self.label_dic = label_dic

        discrete_x, continuous_x = x

        cat_counter = np.bincount(y)
        self._cat_counter = cat_counter

        labels = [y == value for value in range(len(cat_counter))]
        labelled_x = [discrete_x[ci].T for ci in labels]

        self._multinomial._x, self._multinomial._y = x, y
        self._multinomial._labelled_x, self._multinomial._label_zip = labelled_x, list(zip(labels, labelled_x))
        self._multinomial._cat_counter = cat_counter
        self._multinomial._feat_dics = [_dic for i, _dic in enumerate(feat_dics) if self._whether_discrete[i]]
        self._multinomial._n_possibilities = [len(feats) for i, feats in enumerate(features)
                                              if self._whether_discrete[i]]
        self._multinomial.label_dic = label_dic

        labelled_x = [continuous_x[label].T for label in labels]

        self._gaussian._x, self._gaussian._y = continuous_x.T, y
        self._gaussian._labelled_x, self._gaussian._label_zip = labelled_x, labels
        self._gaussian._cat_counter, self._gaussian.label_dic = cat_counter, label_dic

        self._feed_sample_weights(sample_weights)

    def _feed_sample_weights(self, sample_weights=None):
        self._multinomial._feed_sample_weights(sample_weights)
        self._gaussian._feed_sample_weights(sample_weights)

    def _fit(self, lb):
        self._multinomial.fit()
        self._gaussian.fit()
        p_category = self._multinomial.get_prior_probability(lb)
        discrete_func, continuous_func = self._multinomial["func"], self._gaussian["func"]

        def func(input_x, tar_category):
            input_x = np.atleast_2d(input_x)
            return discrete_func(
                input_x[:, self._whether_discrete].astype(np.int), tar_category) * continuous_func(
                input_x[:, self._whether_continuous], tar_category) / p_category[tar_category]

        return func

    def _transfer_x(self, x):
        _feat_dics = self._multinomial["feat_dics"]
        idx = 0
        for d, discrete in enumerate(self._whether_discrete):
            for i, sample in enumerate(x):
                if not discrete:
                    x[i][d] = float(x[i][d])
                else:
                    x[i][d] = _feat_dics[idx][sample[d]]
            if discrete:
                idx += 1
        return x

if __name__ == '__main__':
    import time

    _whether_continuous = [False] * 16
    _continuous_lst = [0, 5, 9, 11, 12, 13, 14]
    for _cl in _continuous_lst:
        _whether_continuous[_cl] = True

    train_num = 40000

    data_time = time.time()
    _data = DataUtil.get_dataset("bank1.0", "../../_Data/bank1.0.txt")
    np.random.shuffle(_data)
    train_x = _data[:train_num]
    test_x = _data[train_num:]
    train_y = [xx.pop() for xx in train_x]
    test_y = [xx.pop() for xx in test_x]
    data_time = time.time() - data_time

    learning_time = time.time()
    nb = MergedNB(_whether_continuous)
    nb.fit(train_x, train_y)
    learning_time = time.time() - learning_time

    estimation_time = time.time()
    nb.estimate(train_x, train_y)
    nb.estimate(test_x, test_y)
    estimation_time = time.time() - estimation_time

    print(
        "Data cleaning   : {:12.6} s\n"
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            data_time, learning_time, estimation_time,
            data_time + learning_time + estimation_time
        )
    )
    nb["multinomial"].visualize()
    nb["gaussian"].visualize()
