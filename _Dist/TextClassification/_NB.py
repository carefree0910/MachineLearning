import os
import math
import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from sklearn import metrics

from _Dist.TextClassification.GenDataset import gen_dataset
from Util.ProgressBar import ProgressBar


def pick_best(sentence, prob_lst):
    rs = [prob["prior"] for prob in prob_lst]
    for j, _prob_dic in enumerate(prob_lst):
        for word in sentence:
            if word in _prob_dic:
                rs[j] *= _prob_dic[word]
            else:
                rs[j] /= _prob_dic["null"]
    return np.argmax(rs)


def train(power=6.46):
    dat_path = os.path.join("_Data", "dataset.dat")
    gen_dataset(dat_path)
    with open(dat_path, "rb") as _file:
        x, y = pickle.load(_file)
    _indices = np.random.permutation(len(x))
    x = [x[i] for i in _indices]
    y = [y[i] for i in _indices]
    data_len = len(x)
    batch_size = math.ceil(data_len*0.1)
    _test_sets, _prob_lists = [], []
    _total = sum([len(sentence) for sentence in x])
    for i in range(10):
        rs = [[] for _ in range(9)]
        _next = (i+1)*batch_size if i != 9 else data_len
        x_train = x[:i * batch_size] + x[(i + 1) * batch_size:]
        y_train = y[:i * batch_size] + y[(i + 1) * batch_size:]
        x_test, y_test = x[i*batch_size:_next], y[i*batch_size:_next]
        for xx, yy in zip(x_train, y_train):
            rs[yy] += xx
        _counters = [Counter(group) for group in rs]
        _test_sets.append((x_test, y_test))
        _prob_lst = []
        for counter in _counters:
            _sum = sum(counter.values())
            _prob_lst.append({
                key: value / _sum for key, value in counter.items()
            })
            _prob_lst[-1]["null"] = _sum * 2 ** power
            _prob_lst[-1]["prior"] = _sum / _total
        _prob_lists.append(_prob_lst)
    return _test_sets, _prob_lists


def test(test_sets, prob_lists):
    acc_lst = []
    for i in range(10):
        _prob_lst = prob_lists[i]
        x_test, y_test = test_sets[i]
        y_pred = np.array([pick_best(sentence, _prob_lst) for sentence in x_test])
        y_test = np.array(y_test)
        acc_lst.append(100 * np.sum(y_pred == y_test) / len(y_pred))
    return acc_lst

if __name__ == '__main__':
    _rs, epoch = [], 10
    bar = ProgressBar(max_value=epoch, name="_NB")
    for _ in range(epoch):
        _rs.append(test(*train()))
        bar.update()
    _rs = np.array(_rs).T
    # x_base = np.arange(len(_rs[0])) + 1
    # plt.figure()
    # for _acc_lst in _rs:
    #     plt.plot(x_base, _acc_lst)
    # plt.plot(x_base, np.average(_rs, axis=0), linewidth=4, label="Average")
    # plt.xlim(1, epoch)
    # plt.ylim(np.min(_rs), np.max(_rs)+2)
    # plt.legend(loc="lower right")
    # plt.show()
    plt.figure()
    plt.boxplot(_rs.T, vert=False, showmeans=True)
    plt.show()
    _rs = np.array(_rs).ravel()
    print("Acc Mean     : {:8.6}".format(np.average(_rs)))
    print("Acc Variance : {:8.6}".format(np.average((_rs - np.average(_rs)) ** 2)))

    sets, lists = train()
    acc_list = test(sets, lists)
    idx = np.argmax(acc_list)  # type: int
    lst_, (x_, y_) = lists[idx], sets[idx]
    print(metrics.classification_report(y_, [
        pick_best(sentence, lst_) for sentence in x_
    ], target_names=np.load(os.path.join("_Data", "LABEL_DIC.npy"))))

    print("Done")
