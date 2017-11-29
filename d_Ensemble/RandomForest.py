import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

from c_CvDTree.Tree import *

from Util.Util import DataUtil
from Util.ProgressBar import ProgressBar


def rf_task(args):
    x, trees, n_cores = args
    return [tree.predict(x, n_cores=n_cores) for tree in trees]


class RandomForest(ClassifierBase):
    RandomForestTiming = Timing()
    cvd_trees = {
        "ID3": ID3Tree,
        "C45": C45Tree,
        "Cart": CartTree
    }

    def __init__(self, **kwargs):
        super(RandomForest, self).__init__(**kwargs)
        self._tree, self._trees = "", []

        self._params["tree"] = kwargs.get("tree", "Cart")
        self._params["epoch"] = kwargs.get("epoch", 10)
        self._params["feature_bound"] = kwargs.get("feature_bound", "log")

    @property
    def title(self):
        return "Tree: {}; Num: {}".format(self._tree, len(self._trees))

    @staticmethod
    @RandomForestTiming.timeit(level=2, prefix="[Core] ")
    def most_appearance(arr):
        u, c = np.unique(arr, return_counts=True)
        return u[np.argmax(c)]

    @RandomForestTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, tree=None, epoch=None, feature_bound=None, **kwargs):
        if sample_weight is None:
            sample_weight = self._params["sample_weight"]
        if tree is None:
            tree = self._params["tree"]
        if epoch is None:
            epoch = self._params["epoch"]
        if feature_bound is None:
            feature_bound = self._params["feature_bound"]
        x, y = np.atleast_2d(x), np.asarray(y)
        n_sample = len(y)
        self._tree = tree
        bar = ProgressBar(max_value=epoch, name="RF")
        for _ in range(epoch):
            tmp_tree = RandomForest.cvd_trees[tree](**kwargs)
            indices = np.random.randint(n_sample, size=n_sample)
            if sample_weight is None:
                local_weight = None
            else:
                local_weight = sample_weight[indices]
                local_weight /= local_weight.sum()
            tmp_tree.fit(x[indices], y[indices], sample_weight=local_weight, feature_bound=feature_bound)
            self._trees.append(deepcopy(tmp_tree))
            bar.update()

    @RandomForestTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False, bound=None, **kwargs):
        trees = self._trees if bound is None else self._trees[:bound]
        matrix = self._multi_clf(x, trees, rf_task, kwargs, target=kwargs.get("target", "parallel"))
        return np.array([RandomForest.most_appearance(rs) for rs in matrix])

    @RandomForestTiming.timeit(level=1, prefix="[API] ")
    def evaluate(self, x, y, metrics=None, tar=0, prefix="Acc", **kwargs):
        kwargs["target"] = "single"
        super(RandomForest, self).evaluate(x, y, metrics, tar, prefix, **kwargs)

if __name__ == '__main__':
    import time

    train_num = 100
    (x_train, y_train), (x_test, y_test) = DataUtil.get_dataset(
        "mushroom", "../_Data/mushroom.txt", n_train=train_num, tar_idx=0)

    learning_time = time.time()
    forest = RandomForest()
    forest.fit(x_train, y_train)
    learning_time = time.time() - learning_time
    estimation_time = time.time()
    forest.evaluate(x_train, y_train)
    forest.evaluate(x_test, y_test)
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
