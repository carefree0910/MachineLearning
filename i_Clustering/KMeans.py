import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np

from Util.Util import DataUtil
from Util.Bases import ClassifierBase
from Util.ProgressBar import ProgressBar


class KMeans(ClassifierBase):
    def __init__(self, **kwargs):
        super(KMeans, self).__init__(**kwargs)
        self._centers = self._counter = None

        self._params["n_clusters"] = kwargs.get("n_clusters", 2)
        self._params["epoch"] = kwargs.get("epoch", 1000)
        self._params["norm"] = kwargs.get("norm", "l2")

    def fit(self, x, n_clusters=None, epoch=None, norm=None, animation_params=None):
        if n_clusters is None:
            n_clusters = self._params["n_clusters"]
        if epoch is None:
            epoch = self._params["epoch"]
        if norm is not None:
            self._params["norm"] = norm
        *animation_properties, animation_params = self._get_animation_params(animation_params)
        x = np.atleast_2d(x)
        arange = np.arange(n_clusters)[..., None]
        x_high_dim, labels_cache, counter = x[:, None, ...], None, 0
        self._centers = x[np.random.permutation(len(x))[:n_clusters]]
        bar = ProgressBar(max_value=epoch, name="KMeans")
        ims = []
        for i in range(epoch):
            labels = self.predict(x_high_dim, high_dim=True)
            if labels_cache is None:
                labels_cache = labels
            else:
                if np.all(labels_cache == labels):
                    bar.update(epoch)
                    break
                else:
                    labels_cache = labels
            for j, indices in enumerate(labels == arange):
                self._centers[j] = np.average(x[indices], axis=0)
            counter += 1
            animation_params["extra"] = self._centers
            self._handle_animation(i, x, labels, ims, animation_params, *animation_properties)
            bar.update()
        self._counter = counter
        self._handle_mp4(ims, animation_properties)

    def predict(self, x, get_raw_results=False, high_dim=False):
        if not high_dim:
            x = x[:, None, ...]
        dis = np.abs(x - self._centers) if self._params["norm"] == "l1" else (x - self._centers) ** 2
        return np.argmin(np.sum(dis, axis=2), axis=1)

if __name__ == '__main__':
    _x, _y = DataUtil.gen_random(size=2000, scale=6)
    k_means = KMeans(n_clusters=8, animation_params={
        "show": False, "mp4": True, "period": 1, "draw_background": True
    })
    k_means.fit(_x)
    k_means.visualize2d(_x, _y, dense=400, extra=k_means["centers"])
    _x, _y = DataUtil.gen_two_clusters()
    k_means = KMeans()
    k_means.fit(_x)
    k_means.visualize2d(_x, _y, dense=400, extra=k_means["centers"])
    _x, _y = DataUtil.gen_two_clusters(n_dim=3)
    k_means = KMeans()
    k_means.fit(_x)
    k_means.visualize3d(_x, _y, dense=100, extra=k_means["centers"])
