import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
from math import log2

from c_CvDTree.Cluster import Cluster

from Util.Metas import TimingMeta


class CvDNode(metaclass=TimingMeta):
    def __init__(self, tree=None, base=2, chaos=None,
                 depth=0, parent=None, is_root=True, prev_feat="Root", **kwargs):
        self._x = self._y = None
        self.base, self.chaos = base, chaos
        self.criterion = self.category = None
        self.left_child = self.right_child = None
        self._children, self.leafs = {}, {}
        self.sample_weight = None
        self.wc = None

        self.tree = tree
        if tree is not None:
            self.wc = tree.whether_continuous
            tree.nodes.append(self)
        self.feature_dim, self.tar, self.feats = None, None, []
        self.parent, self.is_root = parent, is_root
        self._depth, self.prev_feat = depth, prev_feat
        self.is_cart = self.is_continuous = self.affected = self.pruned = False

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    def __lt__(self, other):
        return self.prev_feat < other.prev_feat

    def __str__(self):
        return self.__class__.__name__

    __repr__ = __str__

    @property
    def info(self):
        if self.category is None:
            return "CvDNode ({}) ({} -> {})".format(
                self._depth, self.prev_feat, self.feature_dim)
        return "CvDNode ({}) ({} -> class: {})".format(
            self._depth, self.prev_feat, self.tree.y_transformer[self.category])

    @property
    def children(self):
        return {
            "left": self.left_child, "right": self.right_child
        } if (self.is_cart or self.is_continuous) else self._children

    @property
    def height(self):
        if self.category is not None:
            return 1
        return 1 + max([child.height if child is not None else 0 for child in self.children.values()])

    @property
    def info_dict(self):
        return {
            "chaos": self.chaos,
            "y": self._y
        }

    # Grow

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
        parent = self.parent
        while parent is not None:
            parent.leafs[id(self)] = self.info_dict
            parent = parent.parent

    def prune(self):
        self.category = self.get_category()
        pop_lst = [key for key in self.leafs]
        parent = self.parent
        while parent is not None:
            parent.affected = True
            pop = parent.leafs.pop
            for k in pop_lst:
                pop(k)
            parent.leafs[id(self)] = self.info_dict
            parent = parent.parent
        self.mark_pruned()
        self.feature_dim = None
        self.left_child = self.right_child = None
        self._children = {}
        self.leafs = {}

    def mark_pruned(self):
        self.pruned = True
        for child in self.children.values():
            if child is not None:
                child.mark_pruned()

    def fit(self, x, y, sample_weight, feature_bound=None, eps=1e-8):
        self._x, self._y = np.atleast_2d(x), np.asarray(y)
        self.sample_weight = sample_weight
        if self.stop1(eps):
            return
        cluster = Cluster(self._x, self._y, sample_weight, self.base)
        if self.is_root:
            if self.criterion == "gini":
                self.chaos = cluster.gini()
            else:
                self.chaos = cluster.ent()
        max_gain, chaos_lst = 0, []
        max_feature = max_tar = None
        feat_len = len(self.feats)
        if feature_bound is None:
            indices = range(0, feat_len)
        elif feature_bound == "log":
            indices = np.random.permutation(feat_len)[:max(1, int(log2(feat_len)))]
        else:
            indices = np.random.permutation(feat_len)[:feature_bound]
        tmp_feats = [self.feats[i] for i in indices]
        xt, feat_sets = self._x.T, self.tree.feature_sets
        bin_ig, ig = cluster.bin_info_gain, cluster.info_gain
        for feat in tmp_feats:
            if self.wc[feat]:
                samples = np.sort(xt[feat])
                feat_set = (samples[:-1] + samples[1:]) * 0.5
            else:
                if self.is_cart:
                    feat_set = feat_sets[feat]
                else:
                    feat_set = None
            if self.is_cart or self.wc[feat]:
                for tar in feat_set:
                    tmp_gain, tmp_chaos_lst = bin_ig(
                        feat, tar, criterion=self.criterion, get_chaos_lst=True, continuous=self.wc[feat])
                    if tmp_gain > max_gain:
                        (max_gain, chaos_lst), max_feature, max_tar = (tmp_gain, tmp_chaos_lst), feat, tar
            else:
                tmp_gain, tmp_chaos_lst = ig(
                    feat, self.criterion, True, self.tree.feature_sets[feat])
                if tmp_gain > max_gain:
                    (max_gain, chaos_lst), max_feature = (tmp_gain, tmp_chaos_lst), feat
        if self.stop2(max_gain, eps):
            return
        self.feature_dim = max_feature
        if self.is_cart or self.wc[max_feature]:
            self.tar = max_tar
            self._gen_children(chaos_lst, feature_bound)
            if (self.left_child.category is not None and
                    self.left_child.category == self.right_child.category):
                self.prune()
                self.tree.reduce_nodes()
        else:
            self._gen_children(chaos_lst, feature_bound)

    def _gen_children(self, chaos_lst, feature_bound):
        feat, tar = self.feature_dim, self.tar
        self.is_continuous = continuous = self.wc[feat]
        features = self._x[..., feat]
        new_feats = self.feats.copy()
        if continuous:
            mask = features < tar
            masks = [mask, ~mask]
        else:
            if self.is_cart:
                mask = features == tar
                masks = [mask, ~mask]
                self.tree.feature_sets[feat].discard(tar)
            else:
                masks = None
        if self.is_cart or continuous:
            feats = [tar, "+"] if not continuous else ["{:6.4}-".format(tar), "{:6.4}+".format(tar)]
            for feat, side, chaos in zip(feats, ["left_child", "right_child"], chaos_lst):
                new_node = self.__class__(
                    self.tree, self.base, chaos=chaos,
                    depth=self._depth + 1, parent=self, is_root=False, prev_feat=feat)
                new_node.criterion = self.criterion
                setattr(self, side, new_node)
            for node, feat_mask in zip([self.left_child, self.right_child], masks):
                if self.sample_weight is None:
                    local_weights = None
                else:
                    local_weights = self.sample_weight[feat_mask]
                    local_weights /= np.sum(local_weights)
                tmp_data, tmp_labels = self._x[feat_mask, ...], self._y[feat_mask]
                if len(tmp_labels) == 0:
                    continue
                node.feats = new_feats
                node.fit(tmp_data, tmp_labels, local_weights, feature_bound)
        else:
            new_feats.remove(self.feature_dim)
            for feat, chaos in zip(self.tree.feature_sets[self.feature_dim], chaos_lst):
                feat_mask = features == feat
                tmp_x = self._x[feat_mask, ...]
                if len(tmp_x) == 0:
                    continue
                new_node = self.__class__(
                    tree=self.tree, base=self.base, chaos=chaos,
                    depth=self._depth + 1, parent=self, is_root=False, prev_feat=feat)
                new_node.feats = new_feats
                self.children[feat] = new_node
                if self.sample_weight is None:
                    local_weights = None
                else:
                    local_weights = self.sample_weight[feat_mask]
                    local_weights /= np.sum(local_weights)
                new_node.fit(tmp_x, self._y[feat_mask], local_weights, feature_bound)

    # Util

    def update_layers(self):
        self.tree.layers[self._depth].append(self)
        for node in sorted(self.children):
            node = self.children[node]
            if node is not None:
                node.update_layers()

    def cost(self, pruned=False):
        if not pruned:
            return sum([leaf["chaos"] * len(leaf["y"]) for leaf in self.leafs.values()])
        return self.chaos * len(self._y)

    def get_threshold(self):
        return (self.cost(pruned=True) - self.cost()) / (len(self.leafs) - 1)

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
        print(" " * indent * self._depth, self.info)
        for node in sorted(self.children):
            node = self.children[node]
            if node is not None:
                node.view()


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
