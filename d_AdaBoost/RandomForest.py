from c_CvDTree.Tree import *
from Util import DataUtil


class RandomForest(ClassifierBase, metaclass=ClassifierMeta):
    RandomForestTiming = Timing()
    _cvd_trees = {
        "id3": ID3Tree,
        "c45": C45Tree,
        "cart": CartTree
    }

    def __init__(self):
        self._trees = []
        self._kwarg_cache = {}

    @property
    def params(self):
        rs = ""
        if self._kwarg_cache:
            tmp_rs = []
            for key, value in self._kwarg_cache.items():
                tmp_rs.append("{}: {}".format(key, value))
            rs += "( " + "; ".join(tmp_rs) + " )"
        return rs

    @staticmethod
    @RandomForestTiming.timeit(level=2, prefix="[StaticMethod] ")
    def most_appearance(arr):
        u, c = np.unique(arr, return_counts=True)
        return u[np.argmax(c)]

    @RandomForestTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, tree="cart", epoch=10, sample_weights=None, *args, **kwargs):
        self._kwarg_cache = kwargs
        n_sample = len(y)
        for _ in range(epoch):
            tmp_tree = RandomForest._cvd_trees[tree](*args, **kwargs)
            _indices = np.random.randint(n_sample, size=n_sample)
            tmp_tree.fit(x[_indices], y[_indices], sample_weight=sample_weights, feature_bound=1)
            self._trees.append(deepcopy(tmp_tree))

    @RandomForestTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x):
        _matrix = [_tree.predict(x) for _tree in self._trees]
        return np.array([RandomForest.most_appearance(rs) for rs in zip(*_matrix)])

    def visualize(self):
        try:
            for i, clf in enumerate(self._trees):
                clf.visualize()
                _ = input("Press any key to continue...")
        except AttributeError:
            return

if __name__ == '__main__':
    import time

    train_num = 6000
    (x_train, y_train), (x_test, y_test) = DataUtil.get_dataset(
        "mushroom", "../_Data/mushroom.txt", train_num=train_num, tar_idx=0)

    learning_time = time.time()
    forest = RandomForest()
    forest.fit(x_train, y_train)
    learning_time = time.time() - learning_time
    estimation_time = time.time()
    forest.estimate(x_train, y_train)
    forest.estimate(x_test, y_test)
    estimation_time = time.time() - estimation_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            learning_time, estimation_time,
            learning_time + estimation_time
        )
    )
    forest.show_timing_log()
    # forest.visualize()
