import numpy as np
import tensorflow as tf

from scipy import interp
from sklearn import metrics


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


class Metrics:
    sign_dict = {
        "f1_score": 1,
        "r2_score": 1,
        "auc": 1, "multi_auc": 1, "acc": 1,
        "mse": -1, "ber": -1,
        "log_loss": -1
    }
    require_prob = {name: False for name in sign_dict}
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
    def step(x, name):
        return tf.subtract(tf.cast(x >= 0, tf.float32) * 2, 1, name)

    @staticmethod
    def one_hot(x, name):
        return tf.multiply(
            x,
            tf.cast(tf.equal(x, tf.expand_dims(tf.reduce_max(x, 1), 1)), tf.float32),
            name=name
        )


class Toolbox:
    @staticmethod
    def get_data(file, sep=" "):
        print("Fetching data...")
        return [line.strip().split(sep) for line in file]

    @staticmethod
    def get_one_hot(y, n_classes):
        if y is None:
            return
        one_hot = np.zeros([len(y), n_classes])
        one_hot[range(len(one_hot)), np.asarray(y, np.int)] = 1
        return one_hot

__all__ = ["Losses", "Metrics", "Activations", "Toolbox"]
