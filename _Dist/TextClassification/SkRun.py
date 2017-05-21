import os
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics

from _SKlearn.NaiveBayes import SKMultinomialNB
from _SKlearn.SVM import SKSVM, SKLinearSVM
from _Dist.TextClassification.GenDataset import gen_dataset
from Util.ProgressBar import ProgressBar


def main(clf):
    dat_path = os.path.join("_Data", "dataset.dat")
    gen_dataset(dat_path)
    with open(dat_path, "rb") as _file:
        x, y = pickle.load(_file)
    x = [" ".join(sentence) for sentence in x]
    _indices = np.random.permutation(len(x))
    x = list(np.array(x)[_indices])
    y = list(np.array(y)[_indices])
    data_len = len(x)
    batch_size = math.ceil(data_len * 0.1)
    acc_lst, y_results = [], []
    bar = ProgressBar(max_value=10, name=str(clf))
    for i in range(10):
        _next = (i + 1) * batch_size if i != 9 else data_len
        x_train = x[:i * batch_size] + x[(i + 1) * batch_size:]
        y_train = y[:i * batch_size] + y[(i + 1) * batch_size:]
        x_test, y_test = x[i * batch_size:_next], y[i * batch_size:_next]
        count_vec = CountVectorizer()
        counts_train = count_vec.fit_transform(x_train)
        x_test = count_vec.transform(x_test)
        tfidf_transformer = TfidfTransformer()
        x_train = tfidf_transformer.fit_transform(counts_train)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc_lst.append(clf.acc(y_test, y_pred))
        y_results.append([y_test, y_pred])
        del x_train, y_train, x_test, y_test, y_pred
        bar.update()
    return acc_lst, y_results


def run(clf):
    acc_records, y_records = [], []
    bar = ProgressBar(max_value=10, name="Main")
    for _ in range(10):
        if clf == "Naive Bayes":
            _clf = SKMultinomialNB(alpha=0.1)
        elif clf == "Non-linear SVM":
            _clf = SKSVM()
        else:
            _clf = SKLinearSVM()
        rs = main(_clf)
        acc_records.append(rs[0])
        y_records += rs[1]
        bar.update()
    acc_records = np.array(acc_records) * 100

    plt.figure()
    plt.boxplot(acc_records, vert=False, showmeans=True)
    plt.show()

    from Util.DataToolkit import DataToolkit
    idx = np.argmax(acc_records)  # type: int
    print(metrics.classification_report(y_records[idx][0], y_records[idx][1], target_names=np.load(os.path.join(
        "_Data", "LABEL_DIC.npy"
    ))))
    toolkit = DataToolkit(acc_records[np.argmax(np.average(acc_records, axis=1))])
    print("Acc Mean     : {:8.6}".format(toolkit.mean))
    print("Acc Variance : {:8.6}".format(toolkit.variance))
    print("Done")

if __name__ == '__main__':
    run("SVM")
