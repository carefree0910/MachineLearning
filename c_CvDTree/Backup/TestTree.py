from b_NaiveBayes.Vectorized.MergedNB import MergedNB
from c_Tree.Tree import *

from sklearn.tree import DecisionTreeClassifier


class Util:

    @staticmethod
    def data_cleaning(line):
        line = line.replace('"', "")
        return list(map(lambda c: c.strip(), line.split(";")))

    @staticmethod
    def get_raw_data():
        x = []
        with open("Data/data2.txt", "r") as file:
            for line in file:
                x.append(Util.data_cleaning(line))
        return x


class SKTree(DecisionTreeClassifier):
    def visualize2d(self, x, y, dense=100):
        length = len(x)
        axis = np.array([[.0] * length, [.0] * length])
        for i, xx in enumerate(x):
            axis[0][i] = xx[0]
            axis[1][i] = xx[1]
        xs, ys = np.array(x), np.array(y)

        print("=" * 30 + "\n" + str(self))
        decision_function = lambda _xx: self.predict(_xx)

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


def main():
    # _data, _x, _y = [], [], []
    # with open("Data/data.txt", "r") as file:
    #     for line in file:
    #         _data.append(line.strip().split(","))
    # np.random.shuffle(_data)
    # for line in _data:
    #     _y.append(line.pop(0))
    #     _x.append(line)
    # _x, _y = np.array(_x), np.array(_y)
    # train_num = 5000
    # x_train = _x[:train_num]
    # y_train = _y[:train_num]
    # x_test = _x[train_num:]
    # y_test = _y[train_num:]
    # _fit_time = time.time()
    # _tree = CartTree()
    # _tree.fit(x_train, y_train)
    # _fit_time = time.time() - _fit_time
    # _tree.view()
    # _estimate_time = time.time()
    # _tree.estimate(x_test, y_test)
    # _estimate_time = time.time() - _estimate_time
    # print("Fit      Process : {:8.6} s\n"
    #       "Estimate Process : {:8.6} s".format(_fit_time, _estimate_time))
    # _tree.visualize()

    from Util import DataUtil
    _x, _y = DataUtil.gen_xor()
    _y = np.argmax(_y, axis=1)
    _fit_time = time.time()
    _tree = ID3Tree()
    _tree.fit(_x, _y)
    _fit_time = time.time() - _fit_time
    # _tree.view()
    _estimate_time = time.time()
    # _tree.estimate(_x, _y)
    _estimate_time = time.time() - _estimate_time
    print("Fit      Process : {:8.6} s\n"
          "Estimate Process : {:8.6} s".format(_fit_time, _estimate_time))
    _tree.visualize2d(_x, _y)

    # _whether_discrete = [True] * 16
    # _continuous_lst = [0, 5, 9, 11, 12, 13, 14]
    # for _cl in _continuous_lst:
    #     _whether_discrete[_cl] = False
    # util = Util()
    # _data = util.get_raw_data()
    # np.random.shuffle(_data)
    # _labels = [xx.pop() for xx in _data]
    # nb = MergedNB(_whether_discrete)
    # nb.fit(_data, _labels)
    # _dx, _cx = nb["multinomial"]["x"], nb["gaussian"]["x"]
    # _labels = nb["multinomial"]["y"]
    # _data = np.hstack((_dx, _cx.T))
    # train_num = 1000
    # x_train = _data[:train_num]
    # y_train = _labels[:train_num]
    # x_test = _data[train_num:]
    # y_test = _labels[train_num:]
    # _fit_time = time.time()
    # _tree = CartTree()
    # _tree.fit(x_train, y_train)
    # _fit_time = time.time() - _fit_time
    # _tree.view()
    # _estimate_time = time.time()
    # _tree.estimate(x_test, y_test)
    # _estimate_time = time.time() - _estimate_time
    # print("Fit      Process : {:8.6} s\n"
    #       "Estimate Process : {:8.6} s".format(_fit_time, _estimate_time))
    # _tree.visualize()

if __name__ == '__main__':
    main()
