import os
import sys
root_path = os.path.abspath("../../")
if root_path not in sys.path:
    sys.path.append(root_path)

from b_NaiveBayes.Original.Basic import *
from b_NaiveBayes.Original.MultinomialNB import MultinomialNB
from b_NaiveBayes.Original.GaussianNB import GaussianNB

from Util.Util import DataUtil
from Util.Timing import Timing


class MergedNB(NaiveBayes):
    MergedNBTiming = Timing()

    def __init__(self, **kwargs):
        super(MergedNB, self).__init__(**kwargs)
        self._multinomial, self._gaussian = MultinomialNB(), GaussianNB()

        wc = kwargs.get("whether_continuous")
        if wc is None:
            self._whether_discrete = self._whether_continuous = None
        else:
            self._whether_continuous = np.asarray(wc)
            self._whether_discrete = ~self._whether_continuous

    @MergedNBTiming.timeit(level=1, prefix="[API] ")
    def feed_data(self, x, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        x, y, wc, features, feat_dicts, label_dict = DataUtil.quantize_data(
            x, y, wc=self._whether_continuous, separate=True)
        if self._whether_continuous is None:
            self._whether_continuous = wc
            self._whether_discrete = ~self._whether_continuous
        self.label_dict = label_dict
        discrete_x, continuous_x = x
        cat_counter = np.bincount(y)
        self._cat_counter = cat_counter
        labels = [y == value for value in range(len(cat_counter))]

        labelled_x = [discrete_x[ci].T for ci in labels]
        self._multinomial._x, self._multinomial._y = x, y
        self._multinomial._labelled_x, self._multinomial._label_zip = labelled_x, list(zip(labels, labelled_x))
        self._multinomial._cat_counter = cat_counter
        self._multinomial._feat_dicts = [dic for i, dic in enumerate(feat_dicts) if self._whether_discrete[i]]
        self._multinomial._n_possibilities = [len(feats) for i, feats in enumerate(features)
                                              if self._whether_discrete[i]]
        self._multinomial.label_dict = label_dict

        labelled_x = [continuous_x[label].T for label in labels]
        self._gaussian._x, self._gaussian._y = continuous_x.T, y
        self._gaussian._labelled_x, self._gaussian._label_zip = labelled_x, labels
        self._gaussian._cat_counter, self._gaussian.label_dict = cat_counter, label_dict

        self.feed_sample_weight(sample_weight)

    @MergedNBTiming.timeit(level=1, prefix="[Core] ")
    def feed_sample_weight(self, sample_weight=None):
        self._multinomial.feed_sample_weight(sample_weight)
        self._gaussian.feed_sample_weight(sample_weight)

    @MergedNBTiming.timeit(level=1, prefix="[Core] ")
    def _fit(self, lb):
        self._multinomial.fit()
        self._gaussian.fit()
        p_category = self._multinomial.get_prior_probability(lb)
        discrete_func, continuous_func = self._multinomial["func"], self._gaussian["func"]

        def func(input_x, tar_category):
            input_x = np.asarray(input_x)
            return discrete_func(
                input_x[self._whether_discrete].astype(np.int), tar_category) * continuous_func(
                input_x[self._whether_continuous], tar_category) / p_category[tar_category]

        return func

    @MergedNBTiming.timeit(level=1, prefix="[Core] ")
    def _transfer_x(self, x):
        feat_dicts = self._multinomial["feat_dicts"]
        idx = 0
        for d, discrete in enumerate(self._whether_discrete):
            if not discrete:
                x[d] = float(x[d])
            else:
                x[d] = feat_dicts[idx][x[d]]
            if discrete:
                idx += 1
        return x

if __name__ == '__main__':
    import time

    # whether_discrete = [True, False, True, True]
    # x = DataUtil.get_dataset("balloon2.0", "../../_Data/{}.txt".format("balloon2.0"))
    # y = [xx.pop() for xx in x]
    # learning_time = time.time()
    # nb = MergedNB(whether_discrete)
    # nb.fit(x, y)
    # learning_time = time.time() - learning_time
    # estimation_time = time.time()
    # nb.evaluate(x, y)
    # estimation_time = time.time() - estimation_time
    # print(
    #     "Model building  : {:12.6} s\n"
    #     "Estimation      : {:12.6} s\n"
    #     "Total           : {:12.6} s".format(
    #         learning_time, estimation_time,
    #         learning_time + estimation_time
    #     )
    # )

    whether_continuous = [False] * 16
    continuous_lst = [0, 5, 9, 11, 12, 13, 14]
    for cl in continuous_lst:
        whether_continuous[cl] = True

    train_num = 40000
    data_time = time.time()
    (x_train, y_train), (x_test, y_test) = DataUtil.get_dataset(
        "bank1.0", "../../_Data/bank1.0.txt", n_train=train_num)
    data_time = time.time() - data_time
    learning_time = time.time()
    nb = MergedNB(whether_continuous=whether_continuous)
    nb.fit(x_train, y_train)
    learning_time = time.time() - learning_time
    estimation_time = time.time()
    nb.evaluate(x_train, y_train)
    nb.evaluate(x_test, y_test)
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
