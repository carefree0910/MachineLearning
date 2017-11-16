import os
import math
import datetime
import numpy as np
import tensorflow as tf
import scipy.stats as ss

from scipy import interp
from sklearn import metrics


def init_w(shape, name):
    return tf.get_variable(name, shape, tf.float32, tf.contrib.layers.xavier_initializer())


def init_b(shape, name):
    return tf.get_variable(name, shape, tf.float32, tf.zeros_initializer())


def fully_connected_linear(i, net, shape, bias=True):
    with tf.name_scope("Linear{}".format(i)):
        w_name = "W{}".format(i)
        w = init_w(shape, w_name)
        if bias:
            b = init_b(shape[1], "b{}".format(i))
            return tf.add(tf.matmul(net, w), b, name="Linear{}".format(i))
        return tf.matmul(net, w, name="Linear{}_without_bias".format(i))


def build_tree_projection(dtype, appendix, net, n_tree, n_leaf):
    with tf.name_scope("Tree_Projection"):
        flat_decisions = []
        for i in range(n_tree):
            local_net = net
            with tf.name_scope("Decisions"):
                decisions = tf.nn.sigmoid(fully_connected_linear(
                    "_tree_mapping{}{}_{}".format(i, appendix, dtype), local_net,
                    [local_net.get_shape().as_list()[1], n_leaf], True
                ))
                decisions_comp = 1 - decisions
                decisions_pack = tf.stack([decisions, decisions_comp])
                flat_decisions.append(tf.reshape(decisions_pack, [-1]))
    return flat_decisions


def build_routes(n_leaf, tree_depth, flat_decisions, n_batch_placeholder):
    with tf.name_scope("Routes"):
        batch_0_indices = tf.reshape(tf.range(0, n_batch_placeholder * n_leaf, n_leaf), [-1, 1])
        in_repeat, out_repeat = n_leaf // 2, 1
        batch_complement_indices = tf.reshape(
            [[0] * in_repeat, [n_batch_placeholder * n_leaf] * in_repeat],
            [-1, n_leaf]
        )
        routes = [
            tf.gather(flat_decision, batch_0_indices + batch_complement_indices)
            for flat_decision in flat_decisions
        ]
        for d in range(1, tree_depth + 1):
            indices = tf.range(2 ** d, 2 ** (d + 1)) - 1
            tile_indices = tf.reshape(
                tf.tile(tf.expand_dims(indices, 1), [1, 2 ** (tree_depth - d + 1)]),
                [1, -1]
            )
            batch_indices = batch_0_indices + tile_indices

            in_repeat //= 2
            out_repeat *= 2

            batch_complement_indices = tf.reshape(
                [[0] * in_repeat, [n_batch_placeholder * n_leaf] * in_repeat] * out_repeat,
                [-1, n_leaf]
            )
            for i, flat_decision in enumerate(flat_decisions):
                routes[i] *= tf.gather(flat_decision, batch_indices + batch_complement_indices)
    return routes


def build_tree(net, concat_name, n_batch_placeholder):
    name = "DNDF_Features_{}".format(concat_name)
    n_tree = DNDFConfig.n_tree
    tree_depth = DNDFConfig.tree_depth
    n_leaf = 2 ** (tree_depth + 1)
    appendix = ""
    dtype = "feature"
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        flat_decisions = build_tree_projection(dtype, appendix, net, n_tree, n_leaf)
        routes = build_routes(n_leaf, tree_depth, flat_decisions, n_batch_placeholder)
        return tf.concat(routes, 1, concat_name)


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


class DNDFConfig:
    n_tree = 8
    tree_depth = 2
    fc_shape = n_tree * 2 ** (tree_depth + 1)


class Metrics:
    sign_dict = {
        "f1_score": 1,
        "r2_score": 1,
        "auc": 1, "multi_auc": 1, "acc": 1,
        "mse": -1, "ber": -1,
        "log_loss": -1,
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
                assert y.shape[1] == 2
                return y[..., 1]
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
        if len(y.shape) == 1:
            y = Toolbox.get_one_hot(y, int(np.max(y) + 1))
        n_classes = pred.shape[1]
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
        return ss.pearsonr(y, pred)[0]

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
        return tf.losses.mean_squared_error(y, pred, weights)

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
    def get_one_hot(y, n_classes):
        """ Get one hot representation of y

        Parameters
        ----------
        y : array-like
            Target 1-d labels

        n_classes : int
            Return shape will be [len(y), n_classes]

        Returns np.ndarray
        -------
            One hot representation of y.
        """
        if y is None:
            return
        one_hot = np.zeros([len(y), n_classes])
        one_hot[range(len(one_hot)), np.asarray(y, np.int)] = 1
        return one_hot


class TrainMonitor:
    def __init__(self, sign, snapshot_ratio, level=3, history_ratio=3, tolerance_ratio=2,
                 extension=5, std_floor=0.001, std_ceiling=0.01):
        self.sign, self.flat_flag = sign, False
        self.snapshot_ratio, self.level = snapshot_ratio, max(1, int(level))
        self.n_history = int(snapshot_ratio * history_ratio)
        if level < 3:
            if level == 1:
                tolerance_ratio /= 2
        self.n_tolerance = int(snapshot_ratio * tolerance_ratio)
        self.extension = extension
        self.std_floor, self.std_ceiling = std_floor, std_ceiling
        self._run_id = -1
        self._rs = None
        self._scores = []
        self._running_sum = self._running_square_sum = self._running_best = self.running_epoch = None
        self._is_best = self._over_fit_performance = self._best_checkpoint_performance = None
        self._descend_counter = self._flat_counter = self._over_fitting_flag = None

    @property
    def rs(self):
        return self._rs

    @property
    def params(self):
        return {
            "level": self.level, "n_history": self.n_history, "n_tolerance": self.n_tolerance,
            "extension": self.extension, "std_floor": self.std_floor, "std_ceiling": self.std_ceiling
        }

    @property
    def descend_counter(self):
        return self._descend_counter

    @property
    def over_fitting_flag(self):
        return self._over_fitting_flag

    @property
    def n_epoch(self):
        return len(self._scores) // self.snapshot_ratio

    def reset(self):
        self._run_id = 0
        self._scores = []
        self._is_best = None
        self.reset_monitors()
        self._running_sum = self._running_square_sum = self._running_best = None

    def reset_monitors(self):
        self._reset_rs()
        self._over_fit_performance = math.inf
        self._best_checkpoint_performance = -math.inf
        self._descend_counter = self._flat_counter = self._over_fitting_flag = 0

    def _reset_rs(self):
        self._rs = {"terminate": False, "save_checkpoint": False, "save_best": False, "info": None}

    def _update_running_epoch(self):
        n_epoch, running_epoch = self.n_epoch, self.running_epoch
        terminate = n_epoch >= running_epoch if running_epoch is not None else True
        if self.running_epoch is None:
            self.running_epoch = n_epoch
        else:
            self.running_epoch += n_epoch
            self.running_epoch //= 2
        if not terminate:
            self._descend_counter = max(self._descend_counter - 1, 0)
        return terminate

    def start_new_run(self):
        self._run_id += 1
        self.reset_monitors()
        return self

    def check(self, new_score):
        scores = self._scores
        scores.append(new_score * self.sign)
        n_history = min(self.n_history, len(scores))
        if n_history == 1:
            return self._rs
        # Update running sum & square sum
        if n_history < self.n_history or len(scores) == self.n_history:
            if self._running_sum is None or self._running_square_sum is None:
                self._running_sum = scores[0] + scores[1]
                self._running_square_sum = scores[0] ** 2 + scores[1] ** 2
            else:
                self._running_sum += scores[-1]
                self._running_square_sum += scores[-1] ** 2
        else:
            previous = scores[-n_history - 1]
            self._running_sum += scores[-1] - previous
            self._running_square_sum += scores[-1] ** 2 - previous ** 2
        # Update running best
        if self._running_best is None:
            if scores[0] > scores[1]:
                improvement = 0
                self._running_best, self._is_best = scores[0], False
            else:
                improvement = scores[1] - scores[0]
                self._running_best, self._is_best = scores[1], True
        elif self._running_best > scores[-1]:
            improvement = 0
            self._is_best = False
        else:
            improvement = scores[-1] - self._running_best
            self._running_best = scores[-1]
            self._is_best = True
        # Check
        self._rs["save_checkpoint"] = False
        mean = self._running_sum / n_history
        std = math.sqrt(max(self._running_square_sum / n_history - mean ** 2, 1e-12))
        std = min(std, self.std_ceiling)
        if std < self.std_floor:
            if self.flat_flag:
                self._flat_counter += 1
        else:
            if self.level >= 3 or self._is_best:
                self._flat_counter = max(self._flat_counter - 1, 0)
            elif self.flat_flag and self.level < 3 and not self._is_best:
                self._flat_counter += 1
            res = scores[-1] - mean
            if res < -std and scores[-1] < self._over_fit_performance - std:
                if self._descend_counter == 0:
                    self._rs["save_best"] = True
                    self._over_fit_performance = scores[-1]
                    if self._over_fit_performance > self._running_best:
                        self._best_checkpoint_performance = self._over_fit_performance
                        self._rs["save_checkpoint"] = True
                        self._rs["info"] = (
                            "Current run ({}) seems to be over-fitting, "
                            "saving checkpoint in case we need to restore".format(len(scores) + self._run_id)
                        )
                self._descend_counter += min(self.n_tolerance / 3, -res / std)
                self._over_fitting_flag = 1
            elif res > std:
                if res > 3 * std and self._is_best and improvement > std:
                    self._rs["save_best"] = True
                new_counter = self._descend_counter - res / std
                if self._descend_counter > 0 >= new_counter:
                    self._over_fit_performance = math.inf
                    if scores[-1] > self._best_checkpoint_performance:
                        self._best_checkpoint_performance = scores[-1]
                        if scores[-1] > self._running_best - std:
                            self._rs["save_checkpoint"] = True
                            self._rs["info"] = (
                                "Current run ({}) seems to be working well, "
                                "saving checkpoint in case we need to restore".format(len(scores)+self._run_id)
                            )
                    self._over_fitting_flag = 0
                self._descend_counter = max(new_counter, 0)
        if self._flat_counter >= self.n_tolerance * self.n_history:
            self._rs["info"] = "Performance not improving"
            self._rs["terminate"] = self._update_running_epoch()
            return self._rs
        if self._descend_counter >= self.n_tolerance:
            self._rs["info"] = "Over-fitting"
            self._rs["terminate"] = self._update_running_epoch()
            return self._rs
        if self._is_best:
            self._rs["terminate"] = False
            if self._rs["save_best"]:
                self._best_checkpoint_performance = scores[-1]
                self._rs["save_checkpoint"] = True
                self._rs["save_best"] = False
                self._rs["info"] = (
                    "Current run ({}) leads to best result we've ever had, "
                    "saving checkpoint since ".format(len(scores) + self._run_id)
                )
                if self._over_fitting_flag:
                    self._rs["info"] += "we've suffered from over-fitting"
                else:
                    self._rs["info"] += "performance has improved significantly"
        if not self._rs["save_checkpoint"] and len(scores) % self.snapshot_ratio == 0:
            if scores[-1] > self._best_checkpoint_performance:
                self._best_checkpoint_performance = scores[-1]
                self._rs["terminate"] = False
                self._rs["save_checkpoint"] = True
                self._rs["info"] = (
                    "Current run ({}) leads to best checkpoint we've ever had, "
                    "saving checkpoint in case we need to restore".format(len(scores) + self._run_id)
                )
        return self._rs


__all__ = [
    "init_w", "init_b", "fully_connected_linear", "build_tree", "prepare_tensorboard_verbose",
    "DNDFConfig", "Toolbox", "Metrics", "Losses", "Activations", "TrainMonitor"
]
