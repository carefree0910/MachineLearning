import time
import math
import numpy as np

# TODO: Try batch prediction and visualization
# TODO: Support Continuous Data
# TODO: CART


# Util

class Cluster:
    def __init__(self, data, labels, sample_weights=None, base=2):
        self._data = np.array(data).T
        if sample_weights is None:
            self._counters = np.bincount(labels)
        else:
            self._counters = np.bincount(labels, weights=sample_weights)
        self._sample_weights = sample_weights
        self._labels = np.array(labels)
        self._cache = None
        self._base = base

    def ent(self, ent=None, eps=1e-12):
        _len = len(self._labels)
        if ent is None:
            ent = [_val for _val in self._counters]
        return max(eps, -sum([_c / _len * math.log(_c / _len, self._base) if _c != 0 else 0 for _c in ent]))

    def gini(self, p=None):
        if p is None:
            p = [_val for _val in self._counters]
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
            if self._sample_weights is None:
                _ent = _method(Cluster(tmp_data, tar_label, base=self._base))
            else:
                _ent = _method(Cluster(tmp_data, tar_label, self._sample_weights[data_label], base=self._base))
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
        self.ent = ent
        self.criteria = None
        self.children = {}
        self.category = None
        self.sample_weights = self.prune_criteria = None

        self.tree = tree
        if tree is not None:
            tree.nodes.append(self)
        self.feature_dim = None
        self.feats = []
        self._depth = depth
        self.parent = parent
        self.is_root = is_root
        self._prev_feat = prev_feat
        self.leafs = {}
        self.pruned = False
        self.label_dic = {}

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)
        return

    @property
    def key(self):
        return self._depth, self.prev_feat, id(self)

    @property
    def height(self):
        if self.category is not None:
            return 1
        return 1 + max([_child.height for _child in self.children.values()])

    @property
    def prev_feat(self):
        return self._prev_feat

    @property
    def info_dic(self):
        return {
            "ent": self.ent,
            "labels": self.labels
        }

    def copy(self):
        _new_node = self.__class__(
            None, self._max_depth, self._base, self.ent,
            self._depth, self.parent, self.is_root, self._prev_feat)
        _new_node.tree = self.tree
        _new_node.feature_dim = self.feature_dim
        _new_node.category = self.category
        _new_node.labels = self.labels
        _new_node.pruned = self.pruned
        _new_node.label_dic = self.label_dic
        _new_node.sample_weights = self.sample_weights
        _new_node.prune_criteria = self.prune_criteria
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
        self._data = np.array(data)
        self.labels = np.array(labels)

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
        _counter = np.bincount(self.labels)
        return np.argmax(_counter)

    def get_threshold(self):
        if self.category is None:
            rs = 0
            for leaf in self.leafs.values():
                _cluster = Cluster(None, leaf["labels"], None, self._base)
                rs += len(leaf["labels"]) * _cluster.ent()
            return Cluster(None, self.labels, None, self._base).ent() - rs / (len(self.leafs) - 1)
        return 0

    def _gen_children(self, feat, con_chaos):
        features = self._data[:, feat]
        _new_feats = self.feats[:]
        _new_feats.remove(feat)
        for feat in set(features):
            _feat_mask = features == feat
            _new_node = self.__class__(
                self.tree, self._max_depth, self._base, ent=con_chaos,
                depth=self._depth + 1, parent=self, is_root=False, prev_feat=feat)
            _new_node.feats = _new_feats
            _new_node.label_dic = self.label_dic
            self.children[feat] = _new_node
            _local_weights = None if self.sample_weights is None else self.sample_weights[_feat_mask]
            _new_node.fit(self._data[_feat_mask, :], self.labels[_feat_mask],
                          _local_weights, self.prune_criteria)

    def _handle_terminate(self):
        self.tree.depth = max(self._depth, self.tree.depth)
        self.category = self.get_class()
        _parent = self
        while _parent is not None:
            _parent.leafs[self.key] = self.info_dic
            _parent = _parent.parent

    def fit(self, data, labels, sample_weights, prune_criteria, eps=1e-8):
        if data is not None and labels is not None:
            self.feed_data(data, labels)
        self.sample_weights, self.prune_criteria = sample_weights, prune_criteria
        if self.stop(eps):
            return
        _cluster = Cluster(self._data, self.labels, sample_weights, self._base)
        _max_gain, _con_chaos = _cluster.info_gain(self.feats[0], criteria=self.criteria, get_con_chaos=True)
        _max_feature = self.feats[0]
        for feat in self.feats:
            _tmp_gain, _tmp_con_chaos = _cluster.info_gain(feat, criteria=self.criteria, get_con_chaos=True)
            if _tmp_gain > _max_gain:
                (_max_gain, _con_chaos), _max_feature = (_tmp_gain, _tmp_con_chaos), feat
        if self.early_stop(_max_gain, eps):
            return
        self.feature_dim = _max_feature
        self._gen_children(_max_feature, _con_chaos)
        if self.is_root:
            self.tree.prune()

    def prune(self):
        if self.category is None:
            self.category = self.get_class()
        if self.prune_criteria not in ("normal", "cart"):
            return
        _pop_lst = [key for key in self.leafs]
        _parent = self
        while _parent is not None:
            for _k in _pop_lst:
                _parent.leafs.pop(_k)
            _parent.leafs[self.key] = self.info_dic
            _parent = _parent.parent
        self.mark_pruned()
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
            return self.children[x[self.feature_dim]].predict_one(x)
        except KeyError:
            return self.get_class()

    def view(self, indent=4):
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
            self._depth, self._prev_feat, self.label_dic[self.category])\

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
    def __init__(self, max_depth=None, node=None, prune_criteria="normal"):
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
        self.label_dic = {}
        self.prune_alpha = 1
        self.prune_criteria = prune_criteria

    @staticmethod
    def acc(y, y_pred):
        return np.sum(np.array(y) == np.array(y_pred)) / len(y)

    def copy(self):
        _new_tree = self.__class__(self._max_depth, node=self.root.copy())
        _new_tree.nodes = [_node.copy() for _node in self.nodes]
        _new_tree.label_dic = self.label_dic.copy()
        _new_tree.depth = self.depth
        return _new_tree

    def fit(self, data=None, labels=None, sample_weights=None, eps=1e-8, **kwargs):
        _dic = {c: i for i, c in enumerate(set(labels))}
        labels = np.array([_dic[yy] for yy in labels])
        self.label_dic = {value: key for key, value in _dic.items()}
        data = np.array(data)
        self.root.label_dic = self.label_dic
        self.root.feats = [i for i in range(data.shape[1])]
        self.root.fit(data, labels, sample_weights, self.prune_criteria, eps)
        if self.prune_criteria == "normal":
            self.prune_alpha = kwargs.get("alpha", self.prune_alpha)
        elif self.prune_criteria == "cart":
            _arg = np.argmax([CvDBase.acc(labels, tree.predict(data, False)) for tree in self.trees])
            _tar_tree = self.trees[_arg]
            self.nodes = _tar_tree.nodes
            self.depth = _tar_tree.depth
            self.root = _tar_tree.root

    def _reduce_nodes(self):
        for i in range(len(self.nodes) - 1, -1, -1):
            if self.nodes[i].pruned:
                self.nodes.pop(i)

    def prune(self):
        _continue = False
        if self.root.prune_criteria == "normal":
            if self.depth <= 2:
                return
            _tmp_nodes = [node for node in self.nodes if not node.is_root and not node.category]
            if not _tmp_nodes:
                return
            _old = np.array([sum(
                [leaf["ent"] * len(leaf["labels"]) for leaf in node.leafs.values()]
            ) + self.prune_alpha * len(node.leafs) for node in _tmp_nodes])
            _new = np.array([node.ent * len(node.labels) + self.prune_alpha for node in _tmp_nodes])
            _mask = (_old - _new) > 0
            arg = np.argmax(_mask)
            if _mask[arg]:
                _tmp_nodes[arg].prune()
                _continue = True
        elif self.root.prune_criteria == "cart":
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
                _continue = True
            else:
                self.trees.append(self.copy())
        else:
            return
        if _continue:
            self._reduce_nodes()
            self.prune()

    def predict_one(self, x, transform=True):
        if transform:
            return self.label_dic[self.root.predict_one(x)]
        return self.root.predict_one(x)

    def predict(self, x, transform=True):
        x = np.atleast_2d(x)
        return np.array([self.predict_one(xx, transform) for xx in x])

    def estimate(self, x, y):
        y = np.array(y)
        print("Acc: {:8.6} %".format(100 * np.sum(self.predict(x) == y) / len(y)))

    def view(self):
        self.root.view()

    def __str__(self):
        return "CvDTree ({})".format(self.depth)

    __repr__ = __str__


class CvDMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        base, node = bases

        def __init__(self, *_args, **_kwargs):
            if "node" not in _kwargs:
                CvDBase.__init__(self, node=node(max_depth=kwargs.get("max_depth")), *_args, **_kwargs)
            else:
                CvDBase.__init__(self, *_args, **_kwargs)

        attr["__init__"] = __init__
        return type(name, bases, attr)


class ID3Tree(CvDBase, ID3Node, metaclass=CvDMeta):
    pass


class C45Tree(CvDBase, C45Node, metaclass=CvDMeta):
    pass

if __name__ == '__main__':
    _data, _x, _y = [], [], []
    with open("Data/data.txt", "r") as file:
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
    _tree = C45Tree()
    _tree.fit(x_train, y_train)
    _tree.view()
    _tree.estimate(x_test, y_test)
    print("Time cost: {:8.6}".format(time.time() - _t))
