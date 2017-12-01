import numpy as np
from sklearn.preprocessing import OneHotEncoder


class MultinomialNB:
    def __init__(self, alpha=1.):
        self.alpha = alpha
        self.enc = OneHotEncoder(dtype=np.float32)
        self.class_log_prior = self.feature_log_prob = None

    def fit(self, x, y, do_one_hot=True):
        if do_one_hot:
            x = self.enc.fit_transform(x)
        else:
            x = np.array(x, np.float32)
        n = x.shape[0]
        y = np.asarray(y, np.int8)
        self.class_log_prior = np.log(np.bincount(y) / n)
        masks = [y == i for i in range(len(self.class_log_prior))]
        masked_xs = [x[mask] for mask in masks]
        feature_counts = np.array([np.asarray(masked_x.sum(0))[0] for masked_x in masked_xs])
        smoothed_fc = feature_counts + self.alpha
        self.feature_log_prob = np.log(smoothed_fc / smoothed_fc.sum(1, keepdims=True))

    def _predict(self, x):
        return x.dot(self.feature_log_prob.T) + self.class_log_prior

    def predict(self, x):
        return self._predict(self.enc.transform(x))

    def predict_class(self, x):
        return self._predict(self.enc.transform(x)).argmax(1)
