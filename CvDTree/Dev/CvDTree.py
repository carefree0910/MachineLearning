import time
import math
import numpy as np
from collections import Counter as Count

from sklearn.tree import DecisionTreeClassifier

# TODO: Debug - Pruning, CART Pruning
# TODO: Try batch prediction and visualization
# TODO: Support Continuous Data ; Feed sample-weight


# Util


# Util

class Counter:

    def __init__(self, arr, sample_weights=None):
        if sample_weights is None:
            self._counter = Count(arr)
        else:
            self._counter = {}
            sw_len = len(sample_weights)
            for elem, w in zip(arr, sample_weights):
                if elem not in self._counter:
                    self._counter[elem] = w * sw_len
                else:
                    self._counter[elem] += w * sw_len

    def keys(self):
        return self._counter.keys()

    def values(self):
        return self._counter.values()

    def __getitem__(self, item):
        return self._counter[item]


class Cluster:
    def __init__(self, data, labels, sample_weights=None, base=2):
        self._data = np.array(data).T
        self._labels = np.array(labels)
        self._counter = Counter(labels, sample_weights)
        self._sample_weights = sample_weights
        self._cache = None
        self._base = base

    def ent(self, ent=None, eps=1e-12):
        _len = len(self._labels)
        if ent is None:
            ent = [_val for _val in self._counter.values()]
        return max(eps, -sum([_c / _len * math.log(_c / _len, self._base) for _c in ent]))

    def gini(self, p=None):
        if p is None:
            p = [_val for _val in self._counter.values()]
        return 1 - sum([(_p / len(self._labels)) ** 2 for _p in p])

    def con_chaos(self, idx, criteria="ent"):
        if criteria == "ent":
            _method = lambda cluster: cluster.ent()
        elif criteria == "gini":
            _method = lambda cluster: cluster.gini()
        else:
            raise NotImplementedError("Conditional info criteria '{}' not defined".format(criteria))
        data = self._data[idx]
        features = list(sorted(set(data)))
        self._cache = tmp_labels = [data == feature for feature in features]
        label_lst = [self._labels[label] for label in tmp_labels]
        rs = 0
        for data_label, tar_label in zip(tmp_labels, label_lst):
            tmp_data = self._data.T[data_label]
            _ent = _method(Cluster(tmp_data, tar_label, self._sample_weights, self._base))
            rs += len(tmp_data) / len(data) * _ent
        return rs

    def info_gain(self, idx, criteria="ent", get_con_chaos=False):
        if criteria in ("ent", "ratio"):
            _con_chaos = self.con_chaos(idx)
            _gain = self.ent() - _con_chaos
            if criteria == "ratio":
                _gain = _gain / self.ent([np.sum(_cache) for _cache in self._cache])
        elif criteria == "gini":
            _con_chaos = self.con_chaos(idx, criteria="gini")
            _gain = self.gini() - _con_chaos
        else:
            raise NotImplementedError("Info_gain criteria '{}' not defined".format(criteria))
        return (_gain, _con_chaos) if get_con_chaos else _gain


# Node

class CvDNode:
    def __init__(self, tree=None, max_depth=None, base=2, ent=None,
                 depth=0, parent=None, is_root=True, prev_feat="Root"):
        self._data = self.labels = None
        self._max_depth = max_depth
        self._base = base
        self._ent = ent
        self._sw_cache = self._eps_cache = None
        self.criteria = None
        self.children = {}
        self.category = None

        self.tree = tree
        if tree is not None:
            tree.nodes.append(self)
        self.feature_dim = None
        self._depth = depth
        self.parent = parent
        self._is_root = is_root
        self._prev_feat = prev_feat
        self.weight = 0
        self.leafs = {}
        self.pruned = False

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)
        return

    @property
    def key(self):
        return self._depth, self._prev_feat, id(self)

    @property
    def height(self):
        if self.category is not None:
            return 1
        return 1 + max([_child.height for _child in self.children.values()])

    @property
    def prev_feat(self):
        return self._prev_feat

    def copy(self):
        _new_node = self.__class__(
            None, self._max_depth, self._base, self._ent,
            self._depth, self.parent, self._is_root, self._prev_feat)
        _new_node.tree = self.tree
        _new_node.feature_dim = self.feature_dim
        _new_node.category = self.category
        _new_node.labels = self.labels
        _new_node.pruned = self.pruned
        if self.children:
            for key, node in self.children.items():
                _new_node.children[key] = node.copy()
        else:
            _new_node.category = self.category
        if self.leafs:
            for key, leaf in self.leafs.items():
                _new_node.leafs[key] = leaf.copy()
        return _new_node

    def feed_tree(self, tree):
        self.tree = tree
        self.tree.nodes.append(self)

    def feed_data(self, data, labels):
        self._data = np.array(data).T
        self.labels = np.array(labels)

    def stop(self, eps):
        if (
            self._data.shape[1] == 1 or (self._ent is not None and self._ent <= eps)
            or (self._max_depth is not None and self._depth >= self._max_depth)
        ):
            self._handle_terminate()
            return True
        return False

    def early_stop(self, max_gain, eps):
        if max_gain <= eps:
            self._handle_terminate()
            return True
        return False

    def crop(self, x=None):
        x = self._data if x is None else x
        _mask = np.ones(len(x), dtype=np.bool)
        _mask[self.feature_dim] = False
        return x[_mask]

    def get_class(self):
        _counter = Counter(self.labels)
        return max(_counter.keys(), key=(lambda key: _counter[key]))

    def get_threshold(self):
        if self.category is None:
            rs = 0
            for leaf in self.leafs.values():
                _cluster = Cluster(None, leaf, base=self._base)
                rs += len(leaf) * _cluster.ent()
            return Cluster(None, self.labels, base=self._base).ent() - rs / (self.weight - 1)
        return 0

    def _gen_children(self, features, new_data, con_chaos):
        for feat in set(features):
            _feat_mask = features == feat
            _new_node = self.__class__(
                self.tree, self._max_depth, self._base, ent=con_chaos,
                depth=self._depth + 1, parent=self, is_root=False, prev_feat=feat)
            self.children[feat] = _new_node
            _new_node.fit(new_data[:, _feat_mask].T, self.labels[_feat_mask],
                          self._sw_cache, self._eps_cache)

    def _handle_terminate(self):
        self.tree.depth = max(self._depth, self.tree.depth)
        self.category = self.get_class()
        _parent = self
        while _parent is not None:
            _parent.leafs[self.key] = self.labels
            _parent.weight += 1
            _parent = _parent.parent

    def fit(self, data, labels, sample_weights=None, eps=1e-8):
        if data is not None and labels is not None:
            self.feed_data(data, labels)
        if self.stop(eps):
            return
        _cluster = Cluster(self._data.T, self.labels, sample_weights, self._base)
        _max_gain, _con_chaos = _cluster.info_gain(0, criteria=self.criteria, get_con_chaos=True)
        _max_feature = 0
        for i in range(1, len(self._data)):
            _tmp_gain, _tmp_con_chaos = _cluster.info_gain(i, criteria=self.criteria, get_con_chaos=True)
            if _tmp_gain > _max_gain:
                (_max_gain, _con_chaos), _max_feature = (_tmp_gain, _tmp_con_chaos), i
        if self.early_stop(_max_gain, eps):
            return
        self._sw_cache, self._eps_cache = sample_weights, eps
        self.feature_dim = _max_feature
        self._gen_children(self._data[_max_feature], self.crop(), _con_chaos)
        if self._is_root:
            self.tree.prune()

    def prune(self):
        self.category = self.get_class()
        dw = self.weight - 1
        self.weight = 1
        _pop_lst = [key for key in self.leafs]
        self.mark_pruned()
        _parent = self
        while _parent is not None:
            for _k in _pop_lst:
                _parent.leafs.pop(_k)
            _parent.leafs[self.key] = self.labels
            _parent.weight -= dw
            _parent = _parent.parent
        self.children = {}

    def mark_pruned(self):
        self.pruned = True
        if self.children is not None:
            for _child in self.children.values():
                _child.mark_pruned()

    def predict_one(self, x):
        if self.category is not None:
            return self.category
        try:
            return self.children[x[self.feature_dim]].predict_one(self.crop(x))
        except KeyError:
            return self.get_class()

    def predict(self, x):
        if self.category is not None:
            if self._is_root:
                return np.array([self.category] * len(x))
            return self.category
        x = np.atleast_2d(x)
        return np.array([self.predict_one(xx) for xx in x])

    def view(self, indent=2):
        print(" " * indent * self._depth, self)
        for _node in sorted(self.children.values()):
            _node.view()

    def __lt__(self, other):
        return self.prev_feat < other.prev_feat

    def __str__(self):
        if self.children:
            return "CvDNode ({}) ({} -> {})".format(
                self._depth, self._prev_feat, self.feature_dim)
        return "CvDNode ({}) ({} -> class: {})".format(
            self._depth, self._prev_feat, self.category)

    __repr__ = __str__


class ID3Node(CvDNode):
    def __init__(self, *args, **kwargs):
        CvDNode.__init__(self, *args, **kwargs)
        self.criteria = "ent"


class C45Node(CvDNode):
    def __init__(self, *args, **kwargs):
        CvDNode.__init__(self, *args, **kwargs)
        self.criteria = "ratio"


# Tree

class CvDBase:
    def __init__(self, max_depth=None, node=None):
        self.nodes = []
        self.trees = []
        self._threshold_cache = None
        self._max_depth = max_depth
        if node is None:
            self.root = CvDNode(self, max_depth)
        else:
            self.root = node
            self.root.feed_tree(self)
        self.depth = 1

    @staticmethod
    def acc(y, yp):
        return np.sum(np.array(y) == np.array(yp)) / len(y)

    def copy(self):
        _new_tree = self.__class__(self._max_depth, node=self.root.copy())
        _new_tree.nodes = [_node.copy() for _node in self.nodes]
        _new_tree.depth = self.depth
        return _new_tree

    def fit(self, data=None, labels=None, sample_weights=None, eps=1e-8):
        self.root.fit(data, labels, sample_weights, eps)
        _arg = np.argmax([CvDBase.acc(labels, tree.predict(data)) for tree in self.trees])
        _tar_tree = self.trees[_arg]
        self.nodes = _tar_tree.nodes
        self.depth = _tar_tree.depth
        self.root = _tar_tree.root

    def prune(self):
        self.trees.append(self.copy())
        if self.depth <= 2:
            return
        _nodes = [_node for _node in self.nodes if _node.category is None]
        if self._threshold_cache is None:
            _thresholds = [_node.get_threshold() for _node in _nodes]
        else:
            _thresholds = self._threshold_cache
        _arg = np.argmin(_thresholds)
        _nodes[_arg].prune()
        _thresholds[_arg] = _nodes[_arg].get_threshold()
        self.depth = self.root.height
        for i in range(len(self.nodes) - 1, -1, -1):
            if self.nodes[i].pruned:
                self.nodes.pop(i)
        for i in range(len(_thresholds) - 1, -1, -1):
            if _nodes[i].pruned:
                _thresholds.pop(i)
        self._threshold_cache = _thresholds
        if self.depth > 2:
            self.prune()
        else:
            self.trees.append(self.copy())
        pass

    def predict_one(self, x):
        return self.root.predict_one(x)

    def predict(self, x):
        return self.root.predict(x)

    def estimate(self, x, y):
        y = np.array(y)
        print("Acc: {:8.6} %".format(100 * np.sum(self.predict(x) == y) / len(y)))

    def view(self):
        self.root.view()

    def __str__(self):
        return "CvDTree ({})".format(self.depth)

    __repr__ = __str__


class ID3Tree(CvDBase):
    def __init__(self, *args, **kwargs):
        if "node" not in kwargs:
            CvDBase.__init__(self, node=ID3Node(), *args, **kwargs)
        else:
            CvDBase.__init__(self, *args, **kwargs)


class C45Tree(CvDBase):
    def __init__(self, *args, **kwargs):
        if "node" not in kwargs:
            CvDBase.__init__(self, node=C45Node(), *args, **kwargs)
        else:
            CvDBase.__init__(self, *args, **kwargs)

if __name__ == '__main__':
    _data, _x, _y = [], [], []
    with open("Data/data.txt", "r") as file:
        for line in file:
            _data.append(line.split(","))
    np.random.shuffle(_data)
    for line in _data:
        _y.append(line.pop(0))
        _x.append(line)
    _x, _y = np.array(_x).T, np.array(_y)
    for _i, line in enumerate(_x):
        _dic = {_c: i for i, _c in enumerate(set(line))}
        for _j, _elem in enumerate(line):
            _x[_i][_j] = _dic[_elem]
    _x = _x.T
    train_num = 5000
    x_train = _x[:train_num]
    y_train = _y[:train_num]
    x_test = _x[train_num:]
    y_test = _y[train_num:]

    _t = time.time()
    _tree = C45Tree()
    _tree.fit(x_train, y_train)
    _tree.view()
    _tree.estimate(x_test, y_test)
    print(time.time() - _t)

    _t = time.time()
    _sk_tree = DecisionTreeClassifier()
    _sk_tree.fit(x_train, y_train)
    y_pred = _sk_tree.predict(x_test)
    print("Acc: {:8.6} %".format(100 * np.sum(y_test == np.array(y_pred)) / len(y_pred)))
    print(time.time() - _t)
