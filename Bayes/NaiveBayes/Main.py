import time

from Bayes.NaiveBayes.NaiveBayes import *

np.random.seed(142857)


def main():

    whether_discrete = [True] * 20
    _continuous_lst = [i for i in range(9)]
    for _cl in _continuous_lst:
        whether_discrete[_cl] = False
    nb = MergedNB(whether_discrete)
    util = Util()

    train_num = 40000

    data_time = time.time()
    raw_data = util.get_raw_data()
    np.random.shuffle(raw_data)
    train_data = raw_data[:train_num]
    test_data = raw_data[train_num:]
    data_time = time.time() - data_time

    learning_time = time.time()
    nb.feed_data(train_data)
    nb.fit()
    learning_time = time.time() - learning_time

    estimation_time = time.time()
    nb.estimate(train_data)
    nb.estimate(test_data)
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

main()
