from b_NaiveBayes.Original.Basic import *
from b_NaiveBayes.Original.MultinomialNB import MultinomialNB
from b_NaiveBayes.Original.GaussianNB import GaussianNB

from Util import DataUtil


class MergedNB(NaiveBayes):

    def __init__(self, whether_discrete):
        NaiveBayes.__init__(self)
        self._whether_discrete = np.array(whether_discrete)
        self._whether_continuous = ~self._whether_discrete
        self._multinomial, self._gaussian = MultinomialNB(), GaussianNB()

    def feed_data(self, x, y, sample_weights=None):
        x = np.array(x)
        self._multinomial.feed_data(x[:, self._whether_discrete], y, sample_weights)
        y = self._multinomial["y"]
        self.label_dic = self._multinomial.label_dic
        self._cat_counter = self._multinomial["cat_counter"]
        self._gaussian.feed_data(x[:, self._whether_continuous], y, sample_weights)
        self._gaussian.label_dic = self._multinomial.label_dic

    def feed_sample_weights(self, sample_weights=None):
        self._multinomial.feed_sample_weights(sample_weights)
        self._gaussian.feed_sample_weights(sample_weights)

    def _fit(self, lb):
        self._multinomial.fit()
        self._gaussian.fit()
        discrete_func, continuous_func = self._multinomial["func"], self._gaussian["func"]

        def func(input_x, tar_category):
            input_x = np.array(input_x)
            return discrete_func(
                input_x[self._whether_discrete].astype(np.int), tar_category) * continuous_func(
                input_x[self._whether_continuous], tar_category)

        return func

    def _transfer_x(self, x):
        _feat_dics = self._multinomial["feat_dics"]
        idx = 0
        for d, discrete in enumerate(self._whether_discrete):
            if not discrete:
                x[d] = float(x[d])
            else:
                x[d] = _feat_dics[idx][x[d]]
            if discrete:
                idx += 1
        return x

if __name__ == '__main__':
    import time

    _whether_discrete = [True] * 16
    _continuous_lst = [0, 5, 9, 11, 12, 13, 14]
    for _cl in _continuous_lst:
        _whether_discrete[_cl] = False

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
    nb = MergedNB(_whether_discrete)
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
