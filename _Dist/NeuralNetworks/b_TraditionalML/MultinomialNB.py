import numpy as np
from sklearn.preprocessing import OneHotEncoder


class MultinomialNB:
    """ Naive Bayes algorithm with discrete inputs

    Parameters
    ----------
    alpha : float, optional (default=1.)
        Smooth parameter used in Naive Bayes, default is 1 (which indicates a laplace smoothing)

    Attributes
    ----------
    enc : OneHotEncoder
        One-Hot encoder used to transform (discrete) inputs

    class_log_prior : np.ndarray of float
        Log class prior used to calculate (linear) prediction

    feature_log_prob : np.ndarray of float
        Feature log probability used to calculate (linear) prediction

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randint(0, 10, [1000, 10])  #  Generate feature vectors
    >>> y = np.random.randint(0, 5, 1000)         #  Generate labels
    >>> nb = MultinomialNB().fit(x, y)            #  fit the model
    >>> nb.predict(x)                             #  (linear) prediction
    >>> nb.predict_class(x)                       #  predict labels

    """
    def __init__(self, alpha=1.):
        self.alpha = alpha
        self.enc = self.class_log_prior = self.feature_log_prob = None

    def fit(self, x, y, do_one_hot=True):
        """ Fit the model with x & y

        Parameters
        ----------
        x : {list of float, np.ndarray of float}
            Feature vectors used for training

            Note: features are assumed to be discrete

        y : {list of float, np.ndarray of float}
            Labels used for training

        do_one_hot : bool, optional (default=True)
            Whether do one-hot encoding on x

        Returns
        -------
        self : MultinomialNB
            Returns self.

        """
        if do_one_hot:
            self.enc = OneHotEncoder(dtype=np.float32)
            x = self.enc.fit_transform(x)
        else:
            self.enc = None
            x = np.array(x, np.float32)
        n = x.shape[0]
        y = np.array(y, np.int8)
        self.class_log_prior = np.log(np.bincount(y) / n)
        masks = [y == i for i in range(len(self.class_log_prior))]
        masked_xs = [x[mask] for mask in masks]
        feature_counts = np.array([np.asarray(masked_x.sum(0))[0] for masked_x in masked_xs])
        smoothed_fc = feature_counts + self.alpha
        self.feature_log_prob = np.log(smoothed_fc / smoothed_fc.sum(1, keepdims=True))
        return self

    def _predict(self, x):
        """ Internal method for calculating (linear) predictions

        Parameters
        ----------
        x : {np.ndarray of float, scipy.sparse.csr.csr_matrix of float}
            One-Hot encoded feature vectors

        Returns
        -------
        predictions : np.ndarray of float
            Returns (linear) predictions.

        """
        return x.dot(self.feature_log_prob.T) + self.class_log_prior

    def predict(self, x):
        """ API for calculating (linear) predictions

        Parameters
        ----------
        x : {list of float, np.ndarray of float}
            Target feature vectors

        Returns
        -------
        predictions : np.ndarray of float
            Returns (linear) predictions.

        """
        if self.enc is not None:
            x = self.enc.transform(x)
        return self._predict(x)

    def predict_class(self, x):
        """ API for predicting labels

        Parameters
        ----------
        x : {list of float, np.ndarray of float}
            Target feature vectors

        Returns
        -------
        labels : np.ndarray of int
            Returns labels.

        """
        return np.argmax(self.predict(x), 1)
