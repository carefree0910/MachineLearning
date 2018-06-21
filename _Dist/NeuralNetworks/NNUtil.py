import os
import math
import datetime
import unicodedata
import numpy as np
import tensorflow as tf
import scipy.stats as ss

from scipy import interp
from sklearn import metrics


def init_w(shape, name):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())


def init_b(shape, name):
    return tf.get_variable(name, shape, initializer=tf.zeros_initializer())


def fully_connected_linear(net, shape, appendix, pruner=None):
    with tf.name_scope("Linear{}".format(appendix)):
        w_name = "W{}".format(appendix)
        w = init_w(shape, w_name)
        if pruner is not None:
            w = pruner.prune_w(*pruner.get_w_info(w))
        b = init_b(shape[1], "b{}".format(appendix))
        return tf.add(tf.matmul(net, w), b, name="Linear{}".format(appendix))


def prepare_tensorboard_verbose(sess):
    tb_log_folder = os.path.join(
        os.path.sep, "tmp", "tbLogs",
        str(datetime.datetime.now())[:19].replace(":", "-")
    )
    train_dir = os.path.join(tb_log_folder, "train")
    test_dir = os.path.join(tb_log_folder, "test")
    for tmp_dir in (train_dir, test_dir):
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)
    tf.summary.merge_all()
    tf.summary.FileWriter(train_dir, sess.graph)


class Metrics:
    sign_dict = {
        "f1_score": 1,
        "r2_score": 1,
        "auc": 1, "multi_auc": 1, "acc": 1, "binary_acc": 1,
        "mse": -1, "ber": -1, "log_loss": -1,
        "correlation": 1, "top_10_return": 1
    }
    require_prob = {key: False for key in sign_dict}
    require_prob["auc"] = True
    require_prob["multi_auc"] = True

    @staticmethod
    def check_shape(y, binary=False):
        y = np.asarray(y, np.float32)
        if len(y.shape) == 2:
            if binary:
                if y.shape[1] == 2:
                    return y[..., 1]
                assert y.shape[1] == 1
                return y.ravel()
            return np.argmax(y, axis=1)
        return y

    @staticmethod
    def f1_score(y, pred):
        return metrics.f1_score(Metrics.check_shape(y), Metrics.check_shape(pred))

    @staticmethod
    def r2_score(y, pred):
        return metrics.r2_score(y, pred)

    @staticmethod
    def auc(y, pred):
        return metrics.roc_auc_score(
            Metrics.check_shape(y, True),
            Metrics.check_shape(pred, True)
        )

    @staticmethod
    def multi_auc(y, pred):
        n_classes = pred.shape[1]
        if len(y.shape) == 1:
            y = Toolbox.get_one_hot(y, n_classes)
        fpr, tpr = [None] * n_classes, [None] * n_classes
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y[:, i], pred[:, i])
        new_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        new_tpr = np.zeros_like(new_fpr)
        for i in range(n_classes):
            new_tpr += interp(new_fpr, fpr[i], tpr[i])
        new_tpr /= n_classes
        return metrics.auc(new_fpr, new_tpr)

    @staticmethod
    def acc(y, pred):
        return np.mean(Metrics.check_shape(y) == Metrics.check_shape(pred))

    @staticmethod
    def binary_acc(y, pred):
        return np.mean((y > 0) == (pred > 0))

    @staticmethod
    def mse(y, pred):
        return np.mean(np.square(y.ravel() - pred.ravel()))

    @staticmethod
    def ber(y, pred):
        mat = metrics.confusion_matrix(Metrics.check_shape(y), Metrics.check_shape(pred))
        tp = np.diag(mat)
        fp = mat.sum(axis=0) - tp
        fn = mat.sum(axis=1) - tp
        tn = mat.sum() - (tp + fp + fn)
        return 0.5 * np.mean((fn / (tp + fn) + fp / (tn + fp)))

    @staticmethod
    def log_loss(y, pred):
        return metrics.log_loss(y, pred)

    @staticmethod
    def correlation(y, pred):
        return float(ss.pearsonr(y, pred)[0])

    @staticmethod
    def top_10_return(y, pred):
        return np.mean(y[pred >= np.percentile(pred, 90)])

    @staticmethod
    def from_fpr_tpr(pos, fpr, tpr, metric):
        if metric == "ber":
            return 0.5 * (1 - tpr + fpr)
        return tpr * pos + (1 - fpr) * (1 - pos)


class Losses:
    @staticmethod
    def mse(y, pred, _, weights=None):
        if weights is None:
            return tf.losses.mean_squared_error(y, pred)
        return tf.losses.mean_squared_error(y, pred, tf.reshape(weights, [-1, 1]))

    @staticmethod
    def cross_entropy(y, pred, already_prob, weights=None):
        if already_prob:
            eps = 1e-12
            pred = tf.log(tf.clip_by_value(pred, eps, 1 - eps))
        if weights is None:
            return tf.losses.softmax_cross_entropy(y, pred)
        return tf.losses.softmax_cross_entropy(y, pred, weights)

    @staticmethod
    def correlation(y, pred, _, weights=None):
        y_mean, y_var = tf.nn.moments(y, 0)
        pred_mean, pred_var = tf.nn.moments(pred, 0)
        if weights is None:
            e = tf.reduce_mean((y - y_mean) * (pred - pred_mean))
        else:
            e = tf.reduce_mean((y - y_mean) * (pred - pred_mean) * weights)
        return -e / tf.sqrt(y_var * pred_var)

    @staticmethod
    def perceptron(y, pred, _, weights=None):
        if weights is None:
            return -tf.reduce_mean(y * pred)
        return -tf.reduce_mean(y * pred * weights)

    @staticmethod
    def clipped_perceptron(y, pred, _, weights=None):
        if weights is None:
            return -tf.reduce_mean(tf.maximum(0., y * pred))
        return -tf.reduce_mean(tf.maximum(0., y * pred * weights))

    @staticmethod
    def regression(y, pred, *_):
        return Losses.correlation(y, pred, *_)


class Activations:
    @staticmethod
    def elu(x, name):
        return tf.nn.elu(x, name)

    @staticmethod
    def relu(x, name):
        return tf.nn.relu(x, name)

    @staticmethod
    def selu(x, name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return tf.multiply(scale, tf.where(x >= 0., x, alpha * tf.nn.elu(x)), name)

    @staticmethod
    def sigmoid(x, name):
        return tf.nn.sigmoid(x, name)

    @staticmethod
    def tanh(x, name):
        return tf.nn.tanh(x, name)

    @staticmethod
    def softplus(x, name):
        return tf.nn.softplus(x, name)

    @staticmethod
    def softmax(x, name):
        return tf.nn.softmax(x, name=name)

    @staticmethod
    def sign(x, name):
        return tf.sign(x, name)

    @staticmethod
    def one_hot(x, name):
        return tf.multiply(
            x,
            tf.cast(tf.equal(x, tf.expand_dims(tf.reduce_max(x, 1), 1)), tf.float32),
            name=name
        )


class Toolbox:
    @staticmethod
    def is_number(s):
        try:
            s = float(s)
            if math.isnan(s):
                return False
            return True
        except ValueError:
            try:
                unicodedata.numeric(s)
                return True
            except (TypeError, ValueError):
                return False

    @staticmethod
    def all_same(target):
        x = target[0]
        for new in target[1:]:
            if new != x:
                return False
        return True

    @staticmethod
    def all_unique(target):
        seen = set()
        return not any(x in seen or seen.add(x) for x in target)

    @staticmethod
    def warn_all_same(i, logger=None):
        warn_msg = "All values in column {} are the same, it'll be treated as redundant".format(i)
        print(warn_msg) if logger is None else logger.debug(warn_msg)

    @staticmethod
    def warn_all_unique(i, logger=None):
        warn_msg = "All values in column {} are unique, it'll be treated as redundant".format(i)
        print(warn_msg) if logger is None else logger.debug(warn_msg)

    @staticmethod
    def pop_nan(feat):
        no_nan_feat = []
        for f in feat:
            try:
                f = float(f)
                if math.isnan(f):
                    continue
                no_nan_feat.append(f)
            except ValueError:
                no_nan_feat.append(f)
        return no_nan_feat

    @staticmethod
    def shrink_nan(feat):
        new = np.asarray(feat, np.float32)
        new = new[~np.isnan(new)].tolist()
        if len(new) < len(feat):
            new.append(float("nan"))
        return new

    @staticmethod
    def get_data(file, sep=" ", include_header=False, logger=None):
        msg = "Fetching data"
        print(msg) if logger is None else logger.debug(msg)
        data = [[elem if elem else "nan" for elem in line.strip().split(sep)] for line in file]
        if include_header:
            return data[1:]
        return data

    @staticmethod
    def get_one_hot(y, n_class):
        if y is None:
            return
        one_hot = np.zeros([len(y), n_class])
        one_hot[range(len(one_hot)), np.asarray(y, np.int)] = 1
        return one_hot

    @staticmethod
    def get_feature_info(data, numerical_idx, is_regression, logger=None):
        generate_numerical_idx = False
        if numerical_idx is None:
            generate_numerical_idx = True
            numerical_idx = [False] * len(data[0])
        else:
            numerical_idx = list(numerical_idx)
        data_t = data.T if isinstance(data, np.ndarray) else list(zip(*data))
        if type(data[0][0]) is not str:
            shrink_features = [Toolbox.shrink_nan(feat) for feat in data_t]
        else:
            shrink_features = data_t
        feature_sets = [
            set() if idx is None or idx else set(shrink_feature)
            for idx, shrink_feature in zip(numerical_idx, shrink_features)
        ]
        n_features = [len(feature_set) for feature_set in feature_sets]
        all_num_idx = [
            True if not feature_set else all(Toolbox.is_number(str(feat)) for feat in feature_set)
            for feature_set in feature_sets
        ]
        if generate_numerical_idx:
            np_shrink_features = [
                shrink_feature if not all_num else np.asarray(shrink_feature, np.float32)
                for all_num, shrink_feature in zip(all_num_idx, shrink_features)
            ]
            all_unique_idx = [
                len(feature_set) == len(np_shrink_feature)
                and (not all_num or np.allclose(np_shrink_feature, np_shrink_feature.astype(np.int32)))
                for all_num, feature_set, np_shrink_feature in zip(all_num_idx, feature_sets, np_shrink_features)
            ]
            numerical_idx = Toolbox.get_numerical_idx(feature_sets, all_num_idx, all_unique_idx, logger)
            for i, numerical in enumerate(numerical_idx):
                if numerical is None:
                    all_num_idx[i] = None
        else:
            for i, (feature_set, shrink_feature) in enumerate(zip(feature_sets, shrink_features)):
                if i == len(numerical_idx) - 1 or numerical_idx[i] is None:
                    continue
                if feature_set:
                    if len(feature_set) == 1:
                        Toolbox.warn_all_same(i, logger)
                        all_num_idx[i] = numerical_idx[i] = None
                    continue
                if Toolbox.all_same(shrink_feature):
                    Toolbox.warn_all_same(i, logger)
                    all_num_idx[i] = numerical_idx[i] = None
                elif numerical_idx[i]:
                    shrink_feature = np.asarray(shrink_feature, np.float32)
                    if np.max(shrink_feature[~np.isnan(shrink_feature)]) < 2 ** 30:
                        if np.allclose(shrink_feature, np.array(shrink_feature, np.int32)):
                            if Toolbox.all_unique(shrink_feature):
                                Toolbox.warn_all_unique(i, logger)
                                all_num_idx[i] = numerical_idx[i] = None
        if is_regression:
            all_num_idx[-1] = numerical_idx[-1] = True
            feature_sets[-1] = set()
            n_features.pop()
        return feature_sets, n_features, all_num_idx, numerical_idx

    @staticmethod
    def get_numerical_idx(feature_sets, all_num_idx, all_unique_idx, logger=None):
        rs = []
        print("Generating numerical_idx")
        for i, (feat_set, all_num, all_unique) in enumerate(
            zip(feature_sets, all_num_idx, all_unique_idx)
        ):
            if len(feat_set) == 1:
                Toolbox.warn_all_same(i, logger)
                rs.append(None)
                continue
            no_nan_feat = Toolbox.pop_nan(feat_set)
            if not all_num:
                if len(feat_set) == len(no_nan_feat):
                    rs.append(False)
                    continue
                if not all(Toolbox.is_number(str(feat)) for feat in no_nan_feat):
                    rs.append(False)
                    continue
            no_nan_feat = np.array(list(no_nan_feat), np.float32)
            int_no_nan_feat = no_nan_feat.astype(np.int32)
            n_feat, feat_min, feat_max = len(no_nan_feat), no_nan_feat.min(), no_nan_feat.max()
            if not np.allclose(no_nan_feat, int_no_nan_feat):
                rs.append(True)
                continue
            if all_unique:
                Toolbox.warn_all_unique(i, logger)
                rs.append(None)
                continue
            feat_min, feat_max = int(feat_min), int(feat_max)
            if np.allclose(np.sort(no_nan_feat), np.linspace(feat_min, feat_max, n_feat)):
                rs.append(False)
                continue
            if feat_min >= 20 and n_feat >= 20:
                rs.append(True)
            elif 1.5 * n_feat >= feat_max - feat_min:
                rs.append(False)
            else:
                rs.append(True)
        return np.array(rs)


class TrainMonitor:
    def __init__(self, sign, snapshot_ratio, history_ratio=3, tolerance_ratio=2,
                 extension=5, std_floor=0.001, std_ceiling=0.01):
        self.sign = sign
        self.snapshot_ratio = snapshot_ratio
        self.n_history = int(snapshot_ratio * history_ratio)
        self.n_tolerance = int(snapshot_ratio * tolerance_ratio)
        self.extension = extension
        self.std_floor, self.std_ceiling = std_floor, std_ceiling
        self._scores = []
        self.flat_flag = False
        self._is_best = self._running_best = None
        self._running_sum = self._running_square_sum = None
        self._descend_increment = self.n_history * extension / 30

        self._over_fit_performance = math.inf
        self._best_checkpoint_performance = -math.inf
        self._descend_counter = self._flat_counter = self.over_fitting_flag = 0
        self.info = {"terminate": False, "save_checkpoint": False, "save_best": False, "info": None}

    def punish_extension(self):
        self._descend_counter += self._descend_increment

    def _update_running_info(self, last_score, n_history):
        if n_history < self.n_history or n_history == len(self._scores):
            if self._running_sum is None or self._running_square_sum is None:
                self._running_sum = self._scores[0] + self._scores[1]
                self._running_square_sum = self._scores[0] ** 2 + self._scores[1] ** 2
            else:
                self._running_sum += last_score
                self._running_square_sum += last_score ** 2
        else:
            previous = self._scores[-n_history - 1]
            self._running_sum += last_score - previous
            self._running_square_sum += last_score ** 2 - previous ** 2
        if self._running_best is None:
            if self._scores[0] > self._scores[1]:
                improvement = 0
                self._running_best, self._is_best = self._scores[0], False
            else:
                improvement = self._scores[1] - self._scores[0]
                self._running_best, self._is_best = self._scores[1], True
        elif self._running_best > last_score:
            improvement = 0
            self._is_best = False
        else:
            improvement = last_score - self._running_best
            self._running_best = last_score
            self._is_best = True
        return improvement

    def _handle_overfitting(self, last_score, res, std):
        if self._descend_counter == 0:
            self.info["save_best"] = True
            self._over_fit_performance = last_score
        self._descend_counter += min(self.n_tolerance / 3, -res / std)
        self.over_fitting_flag = 1

    def _handle_recovering(self, improvement, last_score, res, std):
        if res > 3 * std and self._is_best and improvement > std:
            self.info["save_best"] = True
        new_counter = self._descend_counter - res / std
        if self._descend_counter > 0 >= new_counter:
            self._over_fit_performance = math.inf
            if last_score > self._best_checkpoint_performance:
                self._best_checkpoint_performance = last_score
                if last_score > self._running_best - std:
                    self.info["save_checkpoint"] = True
                    self.info["info"] = (
                        "Current snapshot ({}) seems to be working well, "
                        "saving checkpoint in case we need to restore".format(len(self._scores))
                    )
            self.over_fitting_flag = 0
        self._descend_counter = max(new_counter, 0)

    def _handle_is_best(self):
        if self._is_best:
            self.info["terminate"] = False
            if self.info["save_best"]:
                self.info["save_checkpoint"] = True
                self.info["save_best"] = False
                self.info["info"] = (
                    "Current snapshot ({}) leads to best result we've ever had, "
                    "saving checkpoint since ".format(len(self._scores))
                )
                if self.over_fitting_flag:
                    self.info["info"] += "we've suffered from over-fitting"
                else:
                    self.info["info"] += "performance has improved significantly"

    def _handle_period(self, last_score):
        if len(self._scores) % self.snapshot_ratio == 0 and last_score > self._best_checkpoint_performance:
            self._best_checkpoint_performance = last_score
            self.info["terminate"] = False
            self.info["save_checkpoint"] = True
            self.info["info"] = (
                "Current snapshot ({}) leads to best checkpoint we've ever had, "
                "saving checkpoint in case we need to restore".format(len(self._scores))
            )

    def check(self, new_metric):
        last_score = new_metric * self.sign
        self._scores.append(last_score)
        n_history = min(self.n_history, len(self._scores))
        if n_history == 1:
            return self.info
        improvement = self._update_running_info(last_score, n_history)
        self.info["save_checkpoint"] = False
        mean = self._running_sum / n_history
        std = math.sqrt(max(self._running_square_sum / n_history - mean ** 2, 1e-12))
        std = min(std, self.std_ceiling)
        if std < self.std_floor:
            if self.flat_flag:
                self._flat_counter += 1
        else:
            self._flat_counter = max(self._flat_counter - 1, 0)
            res = last_score - mean
            if res < -std and last_score < self._over_fit_performance - std:
                self._handle_overfitting(last_score, res, std)
            elif res > std:
                self._handle_recovering(improvement, last_score, res, std)
        if self._flat_counter >= self.n_tolerance * self.n_history:
            self.info["info"] = "Performance not improving"
            self.info["terminate"] = True
            return self.info
        if self._descend_counter >= self.n_tolerance:
            self.info["info"] = "Over-fitting"
            self.info["terminate"] = True
            return self.info
        self._handle_is_best()
        self._handle_period(last_score)
        return self.info


class DNDF:
    def __init__(self, n_class, n_tree=10, tree_depth=4):
        self.n_class = n_class
        self.n_tree, self.tree_depth = n_tree, tree_depth
        self.n_leaf = 2 ** (tree_depth + 1)
        self.n_internals = self.n_leaf - 1

    def __call__(self, net, n_batch_placeholder, dtype="output", pruner=None):
        name = "DNDF_{}".format(dtype)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            flat_probabilities = self.build_tree_projection(dtype, net, pruner)
            routes = self.build_routes(flat_probabilities, n_batch_placeholder)
            features = tf.concat(routes, 1, name="Feature_Concat")
            if dtype == "feature":
                return features
            leafs = self.build_leafs()
            leafs_matrix = tf.concat(leafs, 0, name="Prob_Concat")
            return tf.divide(
                tf.matmul(features, leafs_matrix),
                float(self.n_tree), name=name
            )

    def build_tree_projection(self, dtype, net, pruner):
        with tf.name_scope("Tree_Projection"):
            flat_probabilities = []
            fc_shape = net.shape[1].value
            for i in range(self.n_tree):
                with tf.name_scope("Decisions"):
                    p_left = tf.nn.sigmoid(fully_connected_linear(
                        net=net,
                        shape=[fc_shape, self.n_internals],
                        appendix="_tree_mapping{}_{}".format(i, dtype), pruner=pruner
                    ))
                    p_right = 1 - p_left
                    p_all = tf.concat([p_left, p_right], 1)
                    flat_probabilities.append(tf.reshape(p_all, [-1]))
        return flat_probabilities

    def build_routes(self, flat_probabilities, n_batch_placeholder):
        with tf.name_scope("Routes"):
            n_flat_prob = 2 * self.n_internals
            batch_indices = tf.reshape(
                tf.range(0, n_flat_prob * n_batch_placeholder, n_flat_prob),
                [-1, 1]
            )
            n_repeat, n_local_internals = int(self.n_leaf * 0.5), 1
            increment_mask = np.repeat([0, self.n_internals], n_repeat)
            routes = [
                tf.gather(p_flat, batch_indices + increment_mask)
                for p_flat in flat_probabilities
            ]
            for depth in range(1, self.tree_depth + 1):
                n_repeat = int(n_repeat * 0.5)
                n_local_internals *= 2
                increment_mask = np.repeat(np.arange(
                    n_local_internals - 1, 2 * n_local_internals - 1
                ), 2)
                increment_mask += np.tile([0, self.n_internals], n_local_internals)
                increment_mask = np.repeat(increment_mask, n_repeat)
                for i, p_flat in enumerate(flat_probabilities):
                    routes[i] *= tf.gather(p_flat, batch_indices + increment_mask)
        return routes

    def build_leafs(self):
        with tf.name_scope("Leafs"):
            if self.n_class == 1:
                local_leafs = [
                    init_w([self.n_leaf, 1], "RegLeaf{}".format(i))
                    for i in range(self.n_tree)
                ]
            else:
                local_leafs = [
                    tf.nn.softmax(w, name="ClfLeafs{}".format(i))
                    for i, w in enumerate([
                        init_w([self.n_leaf, self.n_class], "RawClfLeafs")
                        for _ in range(self.n_tree)
                    ])
                ]
        return local_leafs


class Pruner:
    def __init__(self, alpha=None, beta=None, gamma=None, r=1., eps=1e-12, prune_method="soft_prune"):
        self.alpha, self.beta, self.gamma, self.r, self.eps = alpha, beta, gamma, r, eps
        self.org_ws, self.masks, self.cursor = [], [], -1
        self.method = prune_method
        if prune_method == "soft_prune" or prune_method == "hard_prune":
            if alpha is None:
                self.alpha = 1e-2
            if beta is None:
                self.beta = 1
            if gamma is None:
                self.gamma = 1
            if prune_method == "hard_prune":
                self.alpha *= 0.01
            self.cond_placeholder = None
        elif prune_method == "surgery":
            if alpha is None:
                self.alpha = 1
            if beta is None:
                self.beta = 1
            if gamma is None:
                self.gamma = 0.0001
            self.r = None
            self.cond_placeholder = tf.placeholder(tf.bool, (), name="Prune_flag")
        else:
            raise NotImplementedError("prune_method '{}' is not defined".format(prune_method))

    @property
    def params(self):
        return {
            "eps": self.eps, "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
            "max_ratio": self.r, "method": self.method
        }

    @staticmethod
    def get_w_info(w):
        with tf.name_scope("get_w_info"):
            w_abs = tf.abs(w)
            w_abs_mean, w_abs_var = tf.nn.moments(w_abs, None)
            return w, w_abs, w_abs_mean, tf.sqrt(w_abs_var)

    def prune_w(self, w, w_abs, w_abs_mean, w_abs_std):
        self.cursor += 1
        self.org_ws.append(w)
        with tf.name_scope("Prune"):
            if self.cond_placeholder is None:
                log_w = tf.log(tf.maximum(self.eps, w_abs / (w_abs_mean * self.gamma)))
                if self.r > 0:
                    log_w = tf.minimum(self.r, self.beta * log_w)
                self.masks.append(tf.maximum(self.alpha / self.beta * log_w, log_w))
                return w * self.masks[self.cursor]

            self.masks.append(tf.Variable(tf.ones_like(w), trainable=False))

            def prune(i, do_prune):
                def sub():
                    if do_prune:
                        mask = self.masks[i]
                        self.masks[i] = tf.assign(mask, tf.where(
                            tf.logical_and(
                                tf.equal(mask, 1),
                                tf.less_equal(w_abs, 0.9 * tf.maximum(w_abs_mean + self.beta * w_abs_std, self.eps))
                            ),
                            tf.zeros_like(mask), mask
                        ))
                        mask = self.masks[i]
                        self.masks[i] = tf.assign(mask, tf.where(
                            tf.logical_and(
                                tf.equal(mask, 0),
                                tf.greater(w_abs, 1.1 * tf.maximum(w_abs_mean + self.beta * w_abs_std, self.eps))
                            ),
                            tf.ones_like(mask), mask
                        ))
                    return w * self.masks[i]
                return sub

            return tf.cond(self.cond_placeholder, prune(self.cursor, True), prune(self.cursor, False))


class NanHandler:
    def __init__(self, method, reuse_values=True):
        self._values = None
        self.method = method
        self.reuse_values = reuse_values

    def transform(self, x, numerical_idx, refresh_values=False):
        if self.method is None:
            pass
        elif self.method == "delete":
            x = x[~np.any(np.isnan(x[..., numerical_idx]), axis=1)]
        else:
            if self._values is None:
                self._values = [None] * len(numerical_idx)
            for i, (v, numerical) in enumerate(zip(self._values, numerical_idx)):
                if not numerical:
                    continue
                feat = x[..., i]
                mask = np.isnan(feat)
                if not np.any(mask):
                    continue
                if self.reuse_values and not refresh_values and v is not None:
                    new_value = v
                else:
                    new_value = getattr(np, self.method)(feat[~mask])
                    if self.reuse_values and (v is None or refresh_values):
                        self._values[i] = new_value
                feat[mask] = new_value
        return x

    def reset(self):
        self._values = None


class PreProcessor:
    def __init__(self, method, scale_method, eps_floor=1e-4, eps_ceiling=1e12):
        self.method, self.scale_method = method, scale_method
        self.eps_floor, self.eps_ceiling = eps_floor, eps_ceiling
        self.redundant_idx = None
        self.min = self.max = self.mean = self.std = None

    def _scale(self, x, numerical_idx):
        targets = x[..., numerical_idx]
        self.redundant_idx = [False] * len(numerical_idx)
        mean = std = None
        if self.mean is not None:
            mean = self.mean
        if self.std is not None:
            std = self.std
        if self.min is None:
            self.min = targets.min(axis=0)
        if self.max is None:
            self.max = targets.max(axis=0)
        if mean is None:
            mean = targets.mean(axis=0)
        abs_targets = np.abs(targets)
        max_features = abs_targets.max(axis=0)
        if self.scale_method is not None:
            max_features_res = max_features - mean
            mask = max_features_res > self.eps_ceiling
            n_large = np.sum(mask)
            if n_large > 0:
                idx_lst, val_lst = [], []
                mask_cursor = -1
                for i, numerical in enumerate(numerical_idx):
                    if not numerical:
                        continue
                    mask_cursor += 1
                    if not mask[mask_cursor]:
                        continue
                    idx_lst.append(i)
                    val_lst.append(max_features_res[mask_cursor])
                    local_target = targets[..., mask_cursor]
                    local_abs_target = abs_targets[..., mask_cursor]
                    sign_mask = np.ones(len(targets))
                    sign_mask[local_target < 0] *= -1
                    scaled_value = self._scale_abs_features(local_abs_target) * sign_mask
                    targets[..., mask_cursor] = scaled_value
                    if self.mean is None:
                        mean[mask_cursor] = np.mean(scaled_value)
                    max_features[mask_cursor] = np.max(scaled_value)
                warn_msg = "{} value which is too large: [{}]{}".format(
                    "These {} columns contain".format(n_large) if n_large > 1 else "One column contains",
                    ", ".join(
                        "{}: {:8.6f}".format(idx, val)
                        for idx, val in zip(idx_lst, val_lst)
                    ),
                    ", {} will be scaled by '{}' method".format(
                        "it" if n_large == 1 else "they", self.scale_method
                    )
                )
                print(warn_msg)
                x[..., numerical_idx] = targets
        if std is None:
            if np.any(max_features > self.eps_ceiling):
                targets = targets - mean
            std = np.maximum(self.eps_floor, targets.std(axis=0))
        if self.mean is None and self.std is None:
            self.mean, self.std = mean, std
        return x

    def _scale_abs_features(self, abs_features):
        if self.scale_method == "truncate":
            return np.minimum(abs_features, self.eps_ceiling)
        if self.scale_method == "divide":
            return abs_features / self.max
        if self.scale_method == "log":
            return np.log(abs_features + 1)
        return getattr(np, self.scale_method)(abs_features)

    def _normalize(self, x, numerical_idx):
        x[..., numerical_idx] -= self.mean
        x[..., numerical_idx] /= np.maximum(self.eps_floor, self.std)
        return x

    def _min_max(self, x, numerical_idx):
        x[..., numerical_idx] -= self.min
        x[..., numerical_idx] /= np.maximum(self.eps_floor, self.max - self.min)
        return x

    def transform(self, x, numerical_idx):
        x = self._scale(np.array(x, dtype=np.float32), numerical_idx)
        x = getattr(self, "_" + self.method)(x, numerical_idx)
        return x


__all__ = [
    "init_w", "init_b", "fully_connected_linear", "prepare_tensorboard_verbose",
    "Toolbox", "Metrics", "Losses", "Activations", "TrainMonitor",
    "DNDF", "Pruner", "NanHandler", "PreProcessor"
]
