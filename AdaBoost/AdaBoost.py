import time
import numpy as np
from math import pi, exp, log
import matplotlib.pyplot as plt
from copy import deepcopy

from NaiveBayes import MergedNB

try:
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.tree import DecisionTreeClassifier
    import sklearn.naive_bayes as nb
except ImportError:
    DecisionTreeClassifier = Axes3D = nb = None

np.random.seed(142857)

sqrt_pi = pi ** 0.5
SKIP_FIRST = True


class Util:

    @staticmethod
    def data_cleaning(line):
        line = line.replace('"', "")
        return list(map(lambda c: c.strip(), line.split(";")))

    @staticmethod
    def get_raw_data():
        x = []
        skip_first_flag = None
        with open("Data/data.txt", "r") as file:
            for line in file:
                if skip_first_flag is None and SKIP_FIRST:
                    skip_first_flag = True
                    continue
                x.append(Util.data_cleaning(line))
        return x

    @staticmethod
    def gaussian(x, mu, sigma):
        return exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sqrt_pi * sigma)


class AdaBoost:

    _weak_clf = {
        "CvDTree": DecisionTreeClassifier,
        "NB": nb.GaussianNB,
        "SNB": MergedNB
    }

    """
        Naive implementation of AdaBoost
        Requirement: weak_clf should contain:
            1) 'fit'      method, which supports sample_weight
            2) 'predict'  method, which returns binary numpy array
                          it is recommended that it also provides raw results
    """

    def __init__(self):
        self._clf, self._clfs, self._clfs_weights = "", [], []
        self._sample_weight = None
        self._kwarg_cache = {}

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)
        return

    @property
    def title(self):
        rs = "Classifier: {}; Num: {}".format(self._clf, len(self._clfs))
        rs += " " + self.params
        return rs

    @property
    def params(self):
        rs = ""
        if self._kwarg_cache:
            tmp_rs = []
            for key, value in self._kwarg_cache.items():
                tmp_rs.append("{}: {}".format(key, value))
            rs += "( " + "; ".join(tmp_rs) + " )"
        return rs

    def reset(self):
        self._clf, self._clfs, self._clfs_weights = "", [], []
        self._sample_weight = None

    @staticmethod
    def get_xy(data):
        length = len(data)
        x, y = np.zeros((length, 3)), np.zeros(length, dtype=np.int8)
        for i, (_x, _y) in enumerate(data):
            x[i] = _x
            y[i] = _y
        return x, y

    # noinspection PyUnresolvedReferences
    def fit(self, x, y, clf=None, epoch=10, eps=1e-8, early_stop=False, *args, **kwargs):
        if clf is None or AdaBoost._weak_clf[clf] is None:
            clf = "CvDTree"
            kwargs = {"max_depth": 3}
        self._kwarg_cache = kwargs
        self._clf = clf
        ty = y.copy()
        ty[ty == 0] = -1
        if self._sample_weight is None:
            self._sample_weight = np.ones(len(x)) / len(x)
        tmp_clf = AdaBoost._weak_clf[clf](*args, **kwargs)
        for _ in range(epoch):
            tmp_clf.fit(x, y, self._sample_weight)
            y_pred = tmp_clf.predict(x)
            em = min(max((y_pred != y).dot(self._sample_weight[:, None])[0], eps), 1 - eps)
            am = 0.5 * log(1 / em - 1)
            y_pred[y_pred == 0] = -1
            tmp_vec = self._sample_weight * np.exp(-am * ty * y_pred)
            self._sample_weight = tmp_vec / np.sum(tmp_vec)
            self._clfs.append(deepcopy(tmp_clf.predict))
            self._clfs_weights.append(am)
            if early_stop:
                if em <= eps:
                    break
                if np.allclose(self.predict(x), y):
                    break

    def predict(self, x, get_raw_result=False):
        x = np.array(x)
        rs = np.zeros(len(x))
        for clf, am in zip(self._clfs, self._clfs_weights):
            y_pred = clf(x)
            y_pred[y_pred == 0] = -1
            rs += am * y_pred
        if get_raw_result:
            return rs
        return (rs >= 0).astype(np.int8)

    def estimate(self, x, y):
        y_pred = self.predict(x)
        print("Acc: {:8.6} %".format(100 * np.sum(y_pred == y) / len(y)))

    def visualize3d(self, x, y, dense=150):
        length = len(x)
        axis = np.array([[.0] * length, [.0] * length, [.0] * length])
        for i, xx in enumerate(x):
            axis[0][i] = xx[0]
            axis[1][i] = xx[1]
            axis[2][i] = xx[2]
        xs, ys = np.array(x), np.array(y)

        print("=" * 30 + "\n" + self.title)
        self.estimate(x, y)
        decision_function = lambda _x: self.predict(_x, get_raw_result=True)

        nx, ny, nz, margin = dense, dense, dense, 0.1
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        z_min, z_max = np.min(axis[2]), np.max(axis[2])
        x_margin = max(abs(x_min), abs(x_max)) * margin
        y_margin = max(abs(y_min), abs(y_max)) * margin
        z_margin = max(abs(z_min), abs(z_max)) * margin
        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin
        z_min -= z_margin
        z_max += z_margin

        def get_base(_nx, _ny, _nz):
            _xf = np.linspace(x_min, x_max, _nx)
            _yf = np.linspace(y_min, y_max, _ny)
            _zf = np.linspace(z_min, z_max, _nz)
            n_xf, n_yf, n_zf = np.meshgrid(_xf, _yf, _zf)
            return _xf, _yf, _zf, np.c_[n_xf.ravel(), n_yf.ravel(), n_zf.ravel()]

        xf, yf, zf, base_matrix = get_base(nx, ny, nz)

        t = time.time()
        z_xyz = decision_function(base_matrix).reshape((nx, ny, nz))
        p_classes = (decision_function(xs) > 0).astype(np.uint8)
        _, _, _, base_matrix = get_base(10, 10, 10)
        z_classes = (decision_function(base_matrix) > 0).astype(np.uint8)
        print("Decision Time: {:8.6} s".format(time.time() - t))

        print("Drawing figures...")
        z_xy = np.average(z_xyz, axis=2)
        z_yz = np.average(z_xyz, axis=1)
        z_xz = np.average(z_xyz, axis=0)

        xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
        yz_xf, yz_yf = np.meshgrid(yf, zf, sparse=True)
        xz_xf, xz_yf = np.meshgrid(xf, zf, sparse=True)

        fig = plt.figure(figsize=(16, 4), dpi=100)
        plt.title(self.title)
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        per = 1 / 2
        colors = plt.cm.rainbow([i * per for i in range(2)])

        ax1.scatter(axis[0], axis[1], axis[2], c=[colors[y] for y in ys])
        ax2.scatter(axis[0], axis[1], axis[2], c=["r" if y == 0 else "g" for y in p_classes], s=15)
        xyz_xf, xyz_yf, xyz_zf = base_matrix[:, 0], base_matrix[:, 1], base_matrix[:, 2]
        ax3.scatter(xyz_xf, xyz_yf, xyz_zf, c=["r" if y == 0 else "g" for y in z_classes], s=15)

        plt.show()
        plt.close()

        fig = plt.figure(figsize=(16, 4), dpi=100)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.set_title("xy figure")
        ax1.pcolormesh(xy_xf, xy_yf, z_xy > 0, cmap=plt.cm.Paired)
        ax1.contour(xf, yf, z_xy, c='k-', levels=[0])
        ax1.scatter(axis[0], axis[1], c=[colors[y] for y in ys])

        ax2.set_title("yz figure")
        ax2.pcolormesh(yz_xf, yz_yf, z_yz > 0, cmap=plt.cm.Paired)
        ax2.contour(yf, zf, z_yz, c='k-', levels=[0])
        ax2.scatter(axis[1], axis[2], c=[colors[y] for y in ys])

        ax3.set_title("xz figure")
        ax3.pcolormesh(xz_xf, xz_yf, z_xz > 0, cmap=plt.cm.Paired)
        ax3.contour(xf, zf, z_xz, c='k-', levels=[0])
        ax3.scatter(axis[0], axis[2], c=[colors[y] for y in ys])

        plt.show()

        print("Done.")

    def visualize2d(self, x, y, dense=200):
        length = len(x)
        axis = np.array([[.0] * length, [.0] * length])
        for i, xx in enumerate(x):
            axis[0][i] = xx[0]
            axis[1][i] = xx[1]
        xs, ys = np.array(x), np.array(y)

        print("=" * 30 + "\n" + self.title)
        self.estimate(x, y)
        decision_function = lambda _x: self.predict(_x, get_raw_result=True)

        nx, ny, margin = dense, dense, 0.1
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        x_margin = max(abs(x_min), abs(x_max)) * margin
        y_margin = max(abs(y_min), abs(y_max)) * margin
        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin

        def get_base(_nx, _ny):
            _xf = np.linspace(x_min, x_max, _nx)
            _yf = np.linspace(y_min, y_max, _ny)
            n_xf, n_yf = np.meshgrid(_xf, _yf)
            return _xf, _yf, np.c_[n_xf.ravel(), n_yf.ravel()]

        xf, yf, base_matrix = get_base(nx, ny)

        t = time.time()
        z = decision_function(base_matrix).reshape((nx, ny))
        print("Decision Time: {:8.6} s".format(time.time() - t))

        print("Drawing figures...")
        xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
        per = 1 / 2
        colors = plt.cm.rainbow([i * per for i in range(2)])

        plt.figure()
        plt.pcolormesh(xy_xf, xy_yf, z > 0, cmap=plt.cm.Paired)
        plt.contour(xf, yf, z, c='k-', levels=[0])
        plt.scatter(axis[0], axis[1], c=[colors[y] for y in ys])
        plt.show()

        print("Done.")

if __name__ == '__main__':

    whether_discrete = [True] * 17
    _continuous_lst = [0, 5, 9, 11, 12, 13]
    for _cl in _continuous_lst:
        whether_discrete[_cl] = False
    nb = MergedNB(whether_discrete)
    util = Util()

    train_num = 40000

    data_time = time.time()
    raw_data = util.get_raw_data()
    np.random.shuffle(raw_data)
    nb.feed_data(raw_data)
    clean_data = nb.data
    train_data = clean_data[:train_num]
    test_data = clean_data[train_num:]
    data_time = time.time() - data_time

    whether_discrete = [False] * 6 + [True] * 11
    # nb = MergedNB(whether_discrete)
    # nb.feed_data(train_data)
    # nb.fit()
    # nb.estimate(test_data)

    ada = AdaBoost()
    train_x, test_x = train_data[:, range(clean_data.shape[1] - 1)], test_data[:, range(clean_data.shape[1] - 1)]
    train_y, test_y = train_data[:, -1], test_data[:, -1]

    _t = time.time()
    ada.fit(train_x, train_y, epoch=100)
    ada.estimate(train_x, train_y)
    ada.estimate(test_x, test_y)
    print("Clf Num: {}".format(len(ada["clfs"])))
    print("Clf Params: {}".format(ada.params))
    print("Time Cost: {:8.6}".format(time.time() - _t))
    ada.reset()

    _t = time.time()
    ada.fit(train_x, train_y, "NB")
    ada.estimate(train_x, train_y)
    ada.estimate(test_x, test_y)
    print("Clf Num: {}".format(len(ada["clfs"])))
    print("Clf Params: {}".format(ada.params))
    print("Time Cost: {:8.6}".format(time.time() - _t))
    ada.reset()

    _t = time.time()
    ada.fit(train_x, train_y, "SNB", whether_discrete=whether_discrete)
    ada.estimate(train_x, train_y)
    ada.estimate(test_x, test_y)
    print("Clf Num: {}".format(len(ada["clfs"])))
    print("Clf Params: {}".format(ada.params))
    print("Time Cost: {:8.6}".format(time.time() - _t))
    ada.reset()
