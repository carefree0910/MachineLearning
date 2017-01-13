import time
import math
import numpy as np
from collections import Counter


# Util

class Cluster:
    def __init__(self, data, labels, base=2):
        self._data = np.array(data).T
        self._counters = Counter(labels)
        self._labels = np.array(labels)
        self._base = base

    def ent(self, ent=None, eps=1e-12):
        _len = len(self._labels)
        if ent is None:
            ent = [_val for _val in self._counters.values()]
        return max(eps, -sum([_c / _len * math.log(_c / _len, self._base) for _c in ent]))

    def con_ent(self, idx):
        data = self._data[idx]
        features = set(data)
        tmp_labels = [data == feature for feature in features]
        label_lst = [self._labels[label] for label in tmp_labels]
        rs = 0
        for data_label, tar_label in zip(tmp_labels, label_lst):
            tmp_data = self._data.T[data_label]
            _ent = Cluster(tmp_data, tar_label, base=self._base).ent()
            rs += len(tmp_data) / len(data) * _ent
        return rs

    def info_gain(self, idx):
        _con_ent = self.con_ent(idx)
        _gain = self.ent() - _con_ent
        return _gain, _con_ent


# Node

class CvDNode:
    def __init__(self, tree=None, max_depth=None, base=2, ent=None,
                 depth=0, parent=None, is_root=True, prev_feat="Root"):
        self._data = self.labels = None
        self._max_depth = max_depth
        self._base = base
        self.ent = ent
        self.children = {}
        self.category = None

        self.tree = tree
        if tree is not None:
            tree.nodes.append(self)
        self.feature_dim = None
        self.feats = []
        self._depth = depth
        self.parent = parent
        self.is_root = is_root
        self.prev_feat = prev_feat
        self.leafs = {}
        self.pruned = False

    @property
    def key(self):
        return self._depth, self.prev_feat, id(self)

    @property
    def height(self):
        if self.category is not None:
            return 1
        return 1 + max([_child.height for _child in self.children.values()])

    def feed_data(self, data, labels):
        self._data = data
        self.labels = labels

    def stop(self, eps):
        if (
            self._data.shape[1] == 1 or (self.ent is not None and self.ent <= eps)
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

    def get_class(self):
        _counter = Counter(self.labels)
        return max(_counter.keys(), key=(lambda key: _counter[key]))

    def _gen_children(self, feat, con_ent):
        features = self._data[:, feat]
        _new_feats = self.feats[:]
        _new_feats.remove(feat)
        for feat in set(features):
            _feat_mask = features == feat
            _new_node = self.__class__(
                self.tree, self._max_depth, self._base, ent=con_ent,
                depth=self._depth + 1, parent=self, is_root=False, prev_feat=feat)
            _new_node.feats = _new_feats
            self.children[feat] = _new_node
            _new_node.fit(self._data[_feat_mask, :], self.labels[_feat_mask])

    def _handle_terminate(self):
        self.category = self.get_class()
        _parent = self
        while _parent is not None:
            _parent.leafs[self.key] = self
            _parent = _parent.parent

    def fit(self, data, labels, eps=1e-8):
        if data is not None and labels is not None:
            self.feed_data(data, labels)
        if self.stop(eps):
            return
        _cluster = Cluster(self._data, self.labels, self._base)
        _max_gain, _con_ent = _cluster.info_gain(self.feats[0])
        _max_feature = self.feats[0]
        for feat in self.feats[1:]:
            _tmp_gain, _tmp_con_ent = _cluster.info_gain(feat)
            if _tmp_gain > _max_gain:
                (_max_gain, _con_ent), _max_feature = (_tmp_gain, _tmp_con_ent), feat
        if self.early_stop(_max_gain, eps):
            return
        self.feature_dim = _max_feature
        self._gen_children(_max_feature, _con_ent)
        if self.is_root:
            self.tree.prune()

    def mark_pruned(self):
        self.pruned = True
        if self.children:
            for child in self.children.values():
                child.mark_pruned()

    def prune(self):
        if self.category is None:
            self.category = self.get_class()
            self.feature_dim = None
        _pop_lst = [key for key in self.leafs]
        _parent = self
        while _parent is not None:
            for _k in _pop_lst:
                _parent.leafs.pop(_k)
            _parent.leafs[self.key] = self
            _parent = _parent.parent
        self.mark_pruned()
        self.children = {}

    def predict_one(self, x):
        if self.category is not None:
            return self.category
        try:
            return self.children[x[self.feature_dim]].predict_one(x)
        except KeyError:
            return self.get_class()

    def predict(self, x):
        if self.category is not None:
            if self.is_root:
                return [self.category] * len(x)
            return self.category
        x = np.atleast_2d(x)
        return [self.predict_one(xx) for xx in x]

    def view(self, indent=4):
        print(" " * indent * self._depth, self)
        for _node in sorted(self.children.values()):
            _node.view()

    def __lt__(self, other):
        return self.prev_feat < other.prev_feat

    def __str__(self):
        if self.category is None:
            return "CvDNode ({}) ({} -> {})".format(
                self._depth, self.prev_feat, self.feature_dim)
        return "CvDNode ({}) ({} -> class: {})".format(
            self._depth, self.prev_feat, self.category)

    __repr__ = __str__


# Tree

class CvDBase:
    def __init__(self, max_depth=None):
        self.nodes = []
        self._max_depth = max_depth
        self.root = CvDNode(self, max_depth)

    @property
    def depth(self):
        return self.root.height

    @staticmethod
    def acc(y, y_pred):
        return np.sum(np.array(y) == np.array(y_pred)) / len(y)

    def fit(self, data=None, labels=None, eps=1e-8):
        data, labels = np.array(data), np.array(labels)
        self.root.feats = [i for i in range(data.shape[1])]
        self.root.fit(data, labels, eps)

    def prune(self, alpha=1):
        if self.depth <= 2:
            return
        _tmp_nodes = [node for node in self.nodes if not node.is_root and not node.category]
        _old = np.array([sum(
            [leaf.ent * len(leaf.labels) for leaf in node.leafs.values()]
        ) + alpha * len(node.leafs) for node in _tmp_nodes])
        _new = np.array([node.ent * len(node.labels) + alpha for node in _tmp_nodes])
        _mask = (_old - _new) > 0
        arg = np.argmax(_mask)
        if _mask[arg]:
            _tmp_nodes[arg].prune()
            for i in range(len(self.nodes) - 1, -1, -1):
                if self.nodes[i].pruned:
                    self.nodes.pop(i)
            self.prune(alpha)

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

if __name__ == '__main__':
    _data, _x, _y = [], [], []
    with open("../data.txt", "r") as file:
        for line in file:
            _data.append(line.split(","))
    np.random.shuffle(_data)
    for line in _data:
        _y.append(line.pop(0))
        _x.append(line)
    _x, _y = np.array(_x), np.array(_y)
    train_num = 5000
    x_train = _x[:train_num]
    y_train = _y[:train_num]
    x_test = _x[train_num:]
    y_test = _y[train_num:]

    _t = time.time()
    _tree = CvDBase()
    _tree.fit(x_train, y_train)
    _tree.view()
    _tree.estimate(x_test, y_test)
    print("Time cost: {:8.6}".format(time.time() - _t))
