import numpy as np

from c_Tree.Cluster import Cluster

np.random.seed(142857)


class CvDNode:
    def __init__(self, tree=None, base=2, chaos=None,
                 depth=0, parent=None, is_root=True, prev_feat="Root"):
        self._x = self._y = None
        self.base, self.chaos = base, chaos
        self.criterion = self.category = None
        self.left_child = self.right_child = None
        self._children, self.leafs = {}, {}
        self.sample_weights = None
        self.wc = self.floor = self.ceiling = None

        self.tree = tree
        if tree is not None:
            self.wc = tree.whether_continuous
            tree.nodes.append(self)
        self.feature_dim, self.tar, self.feats = None, None, []
        self.parent, self.is_root = parent, is_root
        self._depth, self.prev_feat = depth, prev_feat
        self.is_cart = self.is_continuous = self.pruned = False

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    def __lt__(self, other):
        return self.prev_feat < other.prev_feat

    def __str__(self):
        if self.category is None:
            return "CvDNode ({}) ({} -> {})".format(
                self._depth, self.prev_feat, self.feature_dim)
        return "CvDNode ({}) ({} -> class: {})".format(
            self._depth, self.prev_feat, self.tree.label_dic[self.category])

    __repr__ = __str__

    @property
    def children(self):
        return {
            "left": self.left_child, "right": self.right_child
        } if (self.is_cart or self.is_continuous) else self._children

    @property
    def height(self):
        if self.category is not None:
            return 1
        return 1 + max([_child.height if _child is not None else 0 for _child in self.children.values()])

    @property
    def info_dic(self):
        return {
            "chaos": self.chaos,
            "y": self._y
        }

    # Grow

    def feed_data(self, x, y):
        self._x, self._y = x, y
        if self.floor is None:
            self.floor = [None] * self._x.shape[1]
        if self.ceiling is None:
            self.ceiling = [None] * self._x.shape[1]

    def stop1(self, eps):
        if (
            self._x.shape[1] == 0 or (self.chaos is not None and self.chaos <= eps)
            or (self.tree.max_depth is not None and self._depth >= self.tree.max_depth)
        ):
            self._handle_terminate()
            return True
        return False

    def stop2(self, max_gain, eps):
        if max_gain <= eps:
            self._handle_terminate()
            return True
        return False

    def get_category(self):
        return np.argmax(np.bincount(self._y))

    def _handle_terminate(self):
        self.category = self.get_category()
        _parent = self.parent
        while _parent is not None:
            _parent.leafs[id(self)] = self.info_dic
            _parent = _parent.parent

    def fit(self, x, y, sample_weights, eps=1e-8):
        self.feed_data(x, y)
        self.sample_weights = sample_weights
        if self.stop1(eps):
            return
        _cluster = Cluster(self._x, self._y, sample_weights, self.base)
        _max_gain = _con_chaos = 0
        _max_feature = _max_tar = None
        for feat in self.feats:
            if self.wc[feat]:
                _samples = np.sort(self._x.T[feat])
                _set = (_samples[:-1] + _samples[1:]) * 0.5
            else:
                if self.is_cart:
                    _set = self.tree.feature_sets[feat]
                else:
                    _set = None
            if self.is_cart or self.wc[feat]:
                for tar in _set:
                    _tmp_gain, _tmp_con_chaos = _cluster.bin_info_gain(
                        feat, tar, criterion=self.criterion, get_chaos_lst=True, continuous=self.wc[feat])
                    if _tmp_gain > _max_gain:
                        (_max_gain, _con_chaos), _max_feature, _max_tar = (_tmp_gain, _tmp_con_chaos), feat, tar
            else:
                _tmp_gain, _tmp_con_chaos = _cluster.info_gain(
                    feat, self.criterion, True, self.tree.feature_sets[feat])
                if _tmp_gain > _max_gain:
                    (_max_gain, _con_chaos), _max_feature = (_tmp_gain, _tmp_con_chaos), feat
        if self.stop2(_max_gain, eps):
            return
        self.feature_dim = _max_feature
        if self.is_cart or self.wc[_max_feature]:
            self.tar = _max_tar
            if not self.wc[_max_feature]:
                self.tree.feature_sets[_max_feature].discard(_max_tar)
            self._gen_children(_con_chaos)
            if (self.left_child.category is None and self.left_child.feature_dim is None) or (
                self.right_child.category is None and self.right_child.feature_dim is None) or (
                self.left_child.category is not None and self.left_child.category == self.right_child.category
            ):
                self.prune()
                self.tree.reduce_nodes()
        else:
            self._gen_children(_con_chaos)

    def _gen_children(self, con_chaos):
        feat, tar = self.feature_dim, self.tar
        self.is_continuous = continuous = self.wc[feat]
        features = self._x[:, feat]
        _new_feats = self.feats[:]
        if continuous:
            _mask = features < tar
            if self.floor[feat] is not None:
                _masks = [_mask & (self.floor[feat] <= features), tar < features]
            elif self.ceiling[feat] is not None:
                _masks = [_mask, (tar <= features) & (features <= self.ceiling[feat])]
            else:
                _masks = [~_mask, _mask]
        else:
            if self.is_cart:
                _mask = features != tar
                _masks = [~_mask, _mask]
            else:
                _masks = None
        if self.is_cart or continuous:
            _feats = [tar, "+"] if not continuous else ["{:6.4}-".format(tar), "{:6.4}+".format(tar)]
            for _feat, side, attr in zip(_feats, ["left_child", "right_child"], [
                ("ceiling", "floor"), ("floor", "ceiling")
            ]):
                _new_node = self.__class__(
                    self.tree, self.base, chaos=con_chaos,
                    depth=self._depth + 1, parent=self, is_root=False, prev_feat=_feat)
                _new_node.criterion = self.criterion
                setattr(self, side, _new_node)
                setattr(_new_node, attr[1], getattr(self, attr[1]).copy())
                _bound = getattr(self, attr[0]).copy()
                _bound[feat] = tar
                setattr(_new_node, attr[0], _bound)
            for _node, _feat_mask in zip([self.left_child, self.right_child], _masks):
                _local_weights = None if self.sample_weights is None else self.sample_weights[_feat_mask]
                tmp_data, tmp_labels = self._x[_feat_mask, :], self._y[_feat_mask]
                if len(tmp_labels) == 0:
                    continue
                _node.feats = _new_feats
                _node.fit(tmp_data, tmp_labels, _local_weights)
        else:
            _new_feats.remove(self.feature_dim)
            for feat in self.tree.feature_sets[self.feature_dim]:
                _feat_mask = features == feat
                tmp_x = self._x[_feat_mask, :]
                if len(tmp_x) == 0:
                    continue
                _new_node = self.__class__(
                    tree=self.tree, base=self.base, chaos=con_chaos,
                    depth=self._depth + 1, parent=self, is_root=False, prev_feat=feat)
                _new_node.feats = _new_feats
                for attr in ("ceiling", "floor"):
                    _bound = getattr(self, attr).copy()
                    setattr(_new_node, attr, _bound)
                self.children[feat] = _new_node
                if self.sample_weights is None:
                    _local_weights = None
                else:
                    _local_weights = self.sample_weights[_feat_mask]
                    _local_weights /= np.sum(_local_weights)
                _new_node.fit(tmp_x, self._y[_feat_mask], _local_weights)

    # Prune

    def get_threshold(self):
        if self.category is None:
            rs = 0
            for leaf in self.leafs.values():
                _cluster = Cluster(None, leaf["y"], None, self.base)
                rs += len(leaf["y"]) * _cluster.ent()
            return Cluster(self._x, self._y, None, self.base).ent() - rs / (len(self.leafs) - 1)
        return 0

    def prune(self):
        if self.category is None:
            self.category = self.get_category()
            self.feature_dim = None
        _pop_lst = [key for key in self.leafs]
        _parent = self.parent
        while _parent is not None:
            for _k in _pop_lst:
                _parent.leafs.pop(_k)
            _parent.leafs[id(self)] = self.info_dic
            _parent = _parent.parent
        self.mark_pruned()
        self.left_child = self.right_child = None
        self._children = {}
        self.leafs = {}

    def mark_pruned(self):
        self.pruned = True
        for _child in self.children.values():
            if _child is not None:
                _child.mark_pruned()

    # Util

    def cut_tree(self):
        self.tree = None
        for child in self.children.values():
            if child is not None:
                child.cut_tree()

    def feed_tree(self, tree):
        self.tree = tree
        self.tree.nodes.append(self)
        self.wc = tree.whether_continuous
        for child in self.children.values():
            if child is not None:
                child.feed_tree(tree)

    def predict_one(self, x):
        if self.category is not None:
            return self.category
        if self.is_continuous:
            if x[self.feature_dim] < self.tar:
                return self.left_child.predict_one(x)
            return self.right_child.predict_one(x)
        if self.is_cart:
            if x[self.feature_dim] == self.tar:
                return self.left_child.predict_one(x)
            return self.right_child.predict_one(x)
        else:
            try:
                return self.children[x[self.feature_dim]].predict_one(x)
            except KeyError:
                return self.get_category()

    def predict(self, x):
        return np.array([self.predict_one(xx) for xx in x])

    def view(self, indent=4):
        print(" " * indent * self._depth, self)
        for _node in sorted(self.children):
            _node = self.children[_node]
            if _node is not None:
                _node.view()

    def update_layers(self):
        self.tree.layers[self._depth].append(self)
        for _node in sorted(self.children):
            _node = self.children[_node]
            if _node is not None:
                _node.update_layers()


class ID3Node(CvDNode):
    def __init__(self, *args, **kwargs):
        CvDNode.__init__(self, *args, **kwargs)
        self.criterion = "ent"


class C45Node(CvDNode):
    def __init__(self, *args, **kwargs):
        CvDNode.__init__(self, *args, **kwargs)
        self.criterion = "ratio"


class CartNode(CvDNode):
    def __init__(self, *args, **kwargs):
        CvDNode.__init__(self, *args, **kwargs)
        self.criterion = "gini"
        self.is_cart = True
