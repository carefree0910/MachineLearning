import cv2
from copy import deepcopy

from c_CvDTree.Node import *

from Util.Timing import Timing
from Util.Bases import ClassifierBase
from Util.Metas import ClassifierMeta


class CvDBase(ClassifierBase, metaclass=ClassifierMeta):
    CvDBaseTiming = Timing()

    def __init__(self, whether_continuous=None, max_depth=None, node=None):
        self.nodes, self.layers, self.roots = [], [], []
        self._max_depth = max_depth
        self.root = node
        self.feature_sets = []
        self.label_dic = {}
        self.prune_alpha = 1
        self.whether_continuous = whether_continuous

    def feed_data(self, x, continuous_rate=0.2):
        xt = x.T
        self.feature_sets = [set(dimension) for dimension in xt]
        data_len, data_dim = x.shape
        if self.whether_continuous is None:
            self.whether_continuous = np.array(
                [len(feat) >= continuous_rate * data_len for feat in self.feature_sets])
        else:
            self.whether_continuous = np.array(self.whether_continuous)
        self.root.feats = [i for i in range(x.shape[1])]
        self.root.feed_tree(self)

    # Grow

    @CvDBaseTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, alpha=None, sample_weight=None, eps=1e-8,
            cv_rate=0.2, train_only=False, feature_bound=None):
        _dic = {c: i for i, c in enumerate(set(y))}
        y = np.array([_dic[yy] for yy in y])
        self.label_dic = {value: key for key, value in _dic.items()}
        x = np.array(x)
        self.prune_alpha = alpha if alpha is not None else x.shape[1] / 2
        if not train_only and self.root.is_cart:
            _train_num = int(len(x) * (1-cv_rate))
            _indices = np.random.permutation(np.arange(len(x)))
            _train_indices = _indices[:_train_num]
            _test_indices = _indices[_train_num:]
            if sample_weight is not None:
                _train_weights = sample_weight[_train_indices]
                _test_weights = sample_weight[_test_indices]
                _train_weights /= np.sum(_train_weights)
                _test_weights /= np.sum(_test_weights)
            else:
                _train_weights = _test_weights = None
            x_train, y_train = x[_train_indices], y[_train_indices]
            x_cv, y_cv = x[_test_indices], y[_test_indices]
        else:
            x_train, y_train, _train_weights = x, y, sample_weight
            x_cv = y_cv = _test_weights = None
        self.feed_data(x_train)
        self.root.fit(x_train, y_train, _train_weights, feature_bound, eps)
        self.prune(x_cv, y_cv, _test_weights)

    @CvDBaseTiming.timeit(level=3, prefix="[Util] ")
    def reduce_nodes(self):
        for i in range(len(self.nodes)-1, -1, -1):
            if self.nodes[i].pruned:
                self.nodes.pop(i)

    # Prune

    @CvDBaseTiming.timeit(level=4)
    def _update_layers(self):
        self.layers = [[] for _ in range(self.root.height)]
        self.root.update_layers()

    @CvDBaseTiming.timeit(level=1)
    def _prune(self):
        self._update_layers()
        _tmp_nodes = []
        for _node_lst in self.layers[::-1]:
            for _node in _node_lst[::-1]:
                if _node.category is None:
                    _tmp_nodes.append(_node)
        _old = np.array([node.cost() + self.prune_alpha * len(node.leafs) for node in _tmp_nodes])
        _new = np.array([node.cost(pruned=True) + self.prune_alpha for node in _tmp_nodes])
        _mask = _old >= _new
        while True:
            if self.root.height == 1:
                break
            p = np.argmax(_mask)
            if _mask[p]:
                _tmp_nodes[p].prune()
                for i, node in enumerate(_tmp_nodes):
                    if node.affected:
                        _old[i] = node.cost() + self.prune_alpha * len(node.leafs)
                        _mask[i] = _old[i] >= _new[i]
                        node.affected = False
                for i in range(len(_tmp_nodes) - 1, -1, -1):
                    if _tmp_nodes[i].pruned:
                        _tmp_nodes.pop(i)
                        _old = np.delete(_old, i)
                        _new = np.delete(_new, i)
                        _mask = np.delete(_mask, i)
            else:
                break
        self.reduce_nodes()

    @CvDBaseTiming.timeit(level=1)
    def _cart_prune(self):
        self.root.cut_tree()
        _tmp_nodes = [node for node in self.nodes if node.category is None]
        _thresholds = np.array([node.get_threshold() for node in _tmp_nodes])
        while True:
            root_copy = deepcopy(self.root)
            self.roots.append(root_copy)
            if self.root.height == 1:
                break
            p = np.argmin(_thresholds)
            _tmp_nodes[p].prune()
            for i, node in enumerate(_tmp_nodes):
                if node.affected:
                    _thresholds[i] = node.get_threshold()
                    node.affected = False
            for i in range(len(_tmp_nodes) - 1, -1, -1):
                if _tmp_nodes[i].pruned:
                    _tmp_nodes.pop(i)
                    _thresholds = np.delete(_thresholds, i)
        self.reduce_nodes()

    @staticmethod
    @CvDBaseTiming.timeit(level=1, prefix="[Util] ")
    def acc(y, y_pred, weights):
        if weights is not None:
            return np.sum((np.array(y) == np.array(y_pred)) * weights) / len(y)
        return np.sum(np.array(y) == np.array(y_pred)) / len(y)

    @CvDBaseTiming.timeit(level=3, prefix="[Util] ")
    def prune(self, x_cv, y_cv, weights):
        if self.root.is_cart:
            if x_cv is not None and y_cv is not None:
                self._cart_prune()
                _arg = np.argmax([CvDBase.acc(y_cv, tree.predict(x_cv), weights) for tree in self.roots])
                _tar_root = self.roots[_arg]
                self.nodes = []
                _tar_root.feed_tree(self)
                self.root = _tar_root
        else:
            self._prune()

    # Util

    @CvDBaseTiming.timeit(level=1, prefix="[API] ")
    def predict_one(self, x):
        return self.label_dic[self.root.predict_one(x)]

    @CvDBaseTiming.timeit(level=3, prefix="[API] ")
    def predict(self, x):
        return np.array([self.predict_one(xx) for xx in x])

    @CvDBaseTiming.timeit(level=3, prefix="[API] ")
    def view(self):
        self.root.view()

    @CvDBaseTiming.timeit(level=2, prefix="[API] ")
    def visualize(self, radius=24, width=1200, height=800, padding=0.2, plot_num=30, title="CvDTree"):
        self._update_layers()
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
                    text = str(node.feature_dim + 1)
                    color = (0, 0, 255)
                else:
                    text = str(self.label_dic[node.category])
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
                        cv2.putText(img, str(self.layers[i+1][k].prev_feat),
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

        def __init__(self, whether_continuous=None, _max_depth=None, _node=None, **_kwargs):
            tmp_node = _node if isinstance(_node, CvDNode) else node
            CvDBase.__init__(self, whether_continuous, _max_depth, tmp_node(**_kwargs))
            self.name = name

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


class CartTree(CvDBase, CartNode, metaclass=CvDMeta):
    pass
