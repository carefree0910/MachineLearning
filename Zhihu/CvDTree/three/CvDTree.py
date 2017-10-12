import cv2
import time
import math
import numpy as np


# Util

class Cluster:
    def __init__(self, data, labels, sample_weights=None, base=2):
        self._data = np.array(data).T
        if sample_weights is None:
            self._counters = np.bincount(labels)
        else:
            # noinspection PyTypeChecker
            self._counters = np.bincount(labels, weights=sample_weights * len(sample_weights))
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

    def con_chaos(self, idx, criterion="ent"):
        if criterion == "ent":
            _method = lambda cluster: cluster.ent()
        elif criterion == "gini":
            _method = lambda cluster: cluster.gini()
        else:
            raise NotImplementedError("Conditional info criterion '{}' not defined".format(criterion))
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
                _new_weights = self._sample_weights[data_label]
                _ent = _method(Cluster(tmp_data, tar_label, _new_weights / np.sum(_new_weights), base=self._base))
            rs += len(tmp_data) / len(data) * _ent
        return rs

    def info_gain(self, idx, criterion="ent", get_con_chaos=False):
        if criterion in ("ent", "ratio"):
            _con_chaos = self.con_chaos(idx)
            _gain = self.ent() - _con_chaos
            if criterion == "ratio":
                _gain = _gain / self.ent([np.sum(_cache) for _cache in self._cache])
        elif criterion == "gini":
            _con_chaos = self.con_chaos(idx, criterion="gini")
            _gain = self.gini() - _con_chaos
        else:
            raise NotImplementedError("Info_gain criterion '{}' not defined".format(criterion))
        return (_gain, _con_chaos) if get_con_chaos else _gain


# Node

class CvDNode:
    def __init__(self, tree=None, base=2, ent=None,
                 depth=0, parent=None, is_root=True, prev_feat="Root"):
        self._data = self.labels = None
        self._base = base
        self.ent = ent
        self.criterion = None
        self.children = {}
        self.category = None
        self.sample_weights = None

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
        self.label_dict = {}

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)
        return

    def __lt__(self, other):
        return self.prev_feat < other.prev_feat

    def __str__(self):
        if self.children:
            return "CvDNode ({}) ({} -> {})".format(
                self._depth, self._prev_feat, self.feature_dim)
        return "CvDNode ({}) ({} -> class: {})".format(
            self._depth, self._prev_feat, self.label_dict[self.category])

    __repr__ = __str__

    @property
    def key(self):
        return self._depth, self.prev_feat, id(self)

    @property
    def height(self):
        if self.category is not None:
            return 1
        return 1 + max([_child.height for _child in self.children.values()])

    @property
    def max_depth(self):
        return self.tree.max_depth

    @property
    def prev_feat(self):
        return self._prev_feat

    @property
    def info_dict(self):
        return {
            "ent": self.ent,
            "labels": self.labels
        }

    def feed_tree(self, tree):
        self.tree = tree
        self.tree.nodes.append(self)

    def feed_data(self, data, labels):
        self._data = np.array(data)
        self.labels = np.array(labels)

    def stop(self, eps):
        if (
            self._data.shape[1] == 1 or (self.ent is not None and self.ent <= eps)
            or (self.max_depth is not None and self._depth >= self.max_depth)
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

    def _gen_children(self, feat, con_chaos):
        features = self._data[:, feat]
        new_feats = self.feats[:]
        new_feats.remove(feat)
        for feat in set(features):
            feat_mask = features == feat
            new_node = self.__class__(
                self.tree, self._base, ent=con_chaos,
                depth=self._depth + 1, parent=self, is_root=False, prev_feat=feat)
            new_node.feats = new_feats
            new_node.label_dict = self.label_dict
            self.children[feat] = new_node
            _local_weights = None if self.sample_weights is None else self.sample_weights[feat_mask]
            new_node.fit(self._data[feat_mask, :], self.labels[feat_mask], _local_weights)

    def _handle_terminate(self):
        self.category = self.get_class()
        parent = self
        while parent is not None:
            parent.leafs[self.key] = self.info_dict
            parent = parent.parent

    def fit(self, data, labels, sample_weights, eps=1e-8):
        if data is not None and labels is not None:
            self.feed_data(data, labels)
        self.sample_weights = sample_weights
        if self.stop(eps):
            return
        _cluster = Cluster(self._data, self.labels, sample_weights, self._base)
        _max_gain, _con_chaos = _cluster.info_gain(self.feats[0], criterion=self.criterion, get_con_chaos=True)
        _max_feature = self.feats[0]
        for feat in self.feats[1:]:
            _tmp_gain, _tmp_con_chaos = _cluster.info_gain(feat, criterion=self.criterion, get_con_chaos=True)
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
            self.feature_dim = None
        _pop_lst = [key for key in self.leafs]
        _parent = self
        while _parent is not None:
            for _k in _pop_lst:
                _parent.leafs.pop(_k)
            _parent.leafs[self.key] = self.info_dict
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

    def update_layers(self):
        self.tree.layers[self._depth].append(self)
        for _node in sorted(self.children.values()):
            _node.update_layers()


class ID3Node(CvDNode):
    def __init__(self, *args, **kwargs):
        CvDNode.__init__(self, *args, **kwargs)
        self.criterion = "ent"


class C45Node(CvDNode):
    def __init__(self, *args, **kwargs):
        CvDNode.__init__(self, *args, **kwargs)
        self.criterion = "ratio"


# Tree

class CvDBase:
    def __init__(self, max_depth=None, node=None):
        self.nodes = []
        self.trees = []
        self.layers = []
        self._threshold_cache = None
        self._max_depth = max_depth
        self.root = node
        self.root.feed_tree(self)
        self.label_dict = {}
        self.prune_alpha = 1

    def __str__(self):
        return "CvDTree ({})".format(self.depth)

    __repr__ = __str__

    @property
    def depth(self):
        return self.root.height

    @staticmethod
    def acc(y, y_pred):
        return np.sum(np.array(y) == np.array(y_pred)) / len(y)

    def fit(self, data=None, labels=None, sample_weights=None, eps=1e-8, **kwargs):
        dic = {c: i for i, c in enumerate(set(labels))}
        labels = np.array([dic[yy] for yy in labels])
        self.label_dict = {value: key for key, value in dic.items()}
        data = np.array(data)
        self.root.label_dict = self.label_dict
        self.root.feats = [i for i in range(data.shape[1])]
        self.root.fit(data, labels, sample_weights, eps)
        self.prune_alpha = kwargs.get("alpha", self.prune_alpha)

    def _reduce_nodes(self):
        for i in range(len(self.nodes) - 1, -1, -1):
            if self.nodes[i].pruned:
                self.nodes.pop(i)

    def prune(self):
        _continue = False
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
        arg = np.argmax(_mask)  # type: int
        if _mask[arg]:
            _tmp_nodes[arg].prune()
            _continue = True
        if _continue:
            self._reduce_nodes()
            self.prune()

    def predict_one(self, x, transform=True):
        if transform:
            return self.label_dict[self.root.predict_one(x)]
        return self.root.predict_one(x)

    def predict(self, x, transform=True):
        x = np.atleast_2d(x)
        return np.array([self.predict_one(xx, transform) for xx in x])

    def estimate(self, x, y):
        y = np.array(y)
        print("Acc: {:8.6} %".format(100 * np.sum(self.predict(x) == y) / len(y)))

    def view(self):
        self.root.view()

    def draw(self, radius=24, width=1200, height=800, padding=0.2, plot_num=30, title="CvDTree"):
        self.layers = [[] for _ in range(self.depth)]
        self.root.update_layers()
        units = [len(layer) for layer in self.layers]

        img = np.ones((height, width, 3), np.uint8) * 255
        axis0_padding = int(height / (len(self.layers) - 1 + 2 * padding)) * padding + plot_num
        axis0 = np.linspace(
            axis0_padding, height - axis0_padding, len(self.layers), dtype=np.int)
        axis1_padding = plot_num
        axis1 = [np.linspace(axis1_padding, width - axis1_padding, unit + 2, dtype=np.int)
                 for unit in units]
        axis1 = [axis[1:-1] for axis in axis1]

        for i, (y, xs) in enumerate(zip(axis0, axis1)):
            for j, x in enumerate(xs):
                if i == 0:
                    cv2.circle(img, (x, y), radius, (225, 100, 125), 1)
                else:
                    cv2.circle(img, (x, y), radius, (125, 100, 225), 1)
                node = self.layers[i][j]
                if node.feature_dim is not None:
                    text = str(node.feature_dim)
                    color = (0, 0, 255)
                else:
                    text = str(self.label_dict[node.category])
                    color = (0, 255, 0)
                cv2.putText(img, text, (x-7*len(text)+2, y+3), cv2.LINE_AA, 0.6, color, 1)

        for i, y in enumerate(axis0):
            if i == len(axis0) - 1:
                break
            for j, x in enumerate(axis1[i]):
                new_y = axis0[i + 1]
                dy = new_y - y - 2 * radius
                for k, new_x in enumerate(axis1[i + 1]):
                    dx = new_x - x
                    length = np.sqrt(dx**2+dy**2)
                    ratio = 0.5 - min(0.4, 1.2 * 24/length)
                    if self.layers[i + 1][k] in self.layers[i][j].children.values():
                        cv2.line(img, (x, y+radius), (x+int(dx*ratio), y+radius+int(dy*ratio)),
                                 (125, 125, 125), 1)
                        cv2.putText(img, self.layers[i+1][k].prev_feat,
                                    (x+int(dx*0.5)-6, y+radius+int(dy*0.5)),
                                    cv2.LINE_AA, 0.6, (0, 0, 0), 1)
                        cv2.line(img, (new_x-int(dx*ratio), new_y-radius-int(dy*ratio)), (new_x, new_y-radius),
                                 (125, 125, 125), 1)

        cv2.imshow(title, img)
        cv2.waitKey(0)
        return img


class CvDMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        _, node = bases

        def __init__(self, **_kwargs):
            _max_depth = None if "max_depth" not in _kwargs else _kwargs.pop("max_depth")
            CvDBase.__init__(self, _max_depth, node(**_kwargs))

        @property
        def max_depth(self):
            return self._max_depth

        attr["__init__"] = __init__
        attr["max_depth"] = max_depth
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
    _tree.draw()
