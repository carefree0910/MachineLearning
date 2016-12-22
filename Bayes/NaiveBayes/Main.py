import time

from Bayes.NaiveBayes.NaiveBayes import *

np.random.seed(142857)


def main():

    whether_discrete = [True] * 17
    _continuous_lst = [0, 5, 9, 11, 12, 13]
    for _cl in _continuous_lst:
        whether_discrete[_cl] = False
    nb = MergedNB(whether_discrete)
    # nb = MultinomialNB()
    util = Util()

    train_num = 40000

    data_time = time.time()
    raw_data = util.get_raw_data()
    np.random.shuffle(raw_data)
    train_data = raw_data[:train_num]
    test_data = raw_data[train_num:]
    nb.feed_data(train_data)
    data_time = time.time() - data_time

    learning_time = time.time()
    nb.fit()
    learning_time = time.time() - learning_time

    estimation_time = time.time()
    nb.estimate(test_data)
    # nb.estimate(train_data)
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
