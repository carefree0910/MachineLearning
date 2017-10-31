import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

import math
import numpy as np

from Util.Metas import TimingMeta


class Cluster(metaclass=TimingMeta):
    def __init__(self, x, y, sample_weight=None, base=2):
        self._x, self._y = x.T, y
        if sample_weight is None:
            self._counters = np.bincount(self._y)
        else:
            # noinspection PyTypeChecker
            self._counters = np.bincount(self._y, weights=sample_weight * len(sample_weight))
        self._sample_weight = sample_weight
        self._con_chaos_cache = self._ent_cache = self._gini_cache = None
        self._base = base

    def __str__(self):
        return "Cluster"

    __repr__ = __str__

    def ent(self, ent=None, eps=1e-12):
        if self._ent_cache is not None and ent is None:
            return self._ent_cache
        y_len = len(self._y)
        if ent is None:
            ent = self._counters
        ent_cache = max(eps, -sum(
            [c / y_len * math.log(c / y_len, self._base) if c != 0 else 0 for c in ent]))
        if ent is None:
            self._ent_cache = ent_cache
        return ent_cache

    def gini(self, p=None):
        if self._gini_cache is not None and p is None:
            return self._gini_cache
        if p is None:
            p = self._counters
        gini_cache = 1 - np.sum((p / len(self._y)) ** 2)
        if p is None:
            self._gini_cache = gini_cache
        return gini_cache

    def con_chaos(self, idx, criterion="ent", features=None):
        if criterion == "ent":
            method = lambda cluster: cluster.ent()
        elif criterion == "gini":
            method = lambda cluster: cluster.gini()
        else:
            raise NotImplementedError("Conditional info criterion '{}' not defined".format(criterion))
        data = self._x[idx]
        if features is None:
            features = np.unique(data)
        tmp_labels = [data == feature for feature in features]
        # noinspection PyTypeChecker
        self._con_chaos_cache = [np.sum(label) for label in tmp_labels]
        label_lst = [self._y[label] for label in tmp_labels]
        rs, chaos_lst, xt = 0, [], self._x.T
        append = chaos_lst.append
        for data_label, tar_label in zip(tmp_labels, label_lst):
            tmp_data = xt[data_label]
            if self._sample_weight is None:
                chaos = method(Cluster(tmp_data, tar_label, base=self._base))
            else:
                new_weights = self._sample_weight[data_label]
                chaos = method(Cluster(tmp_data, tar_label, new_weights / np.sum(new_weights), base=self._base))
            rs += len(tmp_data) / len(data) * chaos
            append(chaos)
        return rs, chaos_lst

    def info_gain(self, idx, criterion="ent", get_chaos_lst=False, features=None):
        if criterion in ("ent", "ratio"):
            con_chaos, chaos_lst = self.con_chaos(idx, criterion="ent", features=features)
            gain = self.ent() - con_chaos
            if criterion == "ratio":
                gain /= self.ent(self._con_chaos_cache)
        elif criterion == "gini":
            con_chaos, chaos_lst = self.con_chaos(idx, criterion="gini", features=features)
            gain = self.gini() - con_chaos
        else:
            raise NotImplementedError("Info_gain criterion '{}' not defined".format(criterion))
        return (gain, chaos_lst) if get_chaos_lst else gain

    def bin_con_chaos(self, idx, tar, criterion="gini", continuous=False):
        if criterion == "ent":
            method = lambda cluster: cluster.ent()
        elif criterion == "gini":
            method = lambda cluster: cluster.gini()
        else:
            raise NotImplementedError("Conditional info criterion '{}' not defined".format(criterion))
        data = self._x[idx]
        tar = data == tar if not continuous else data < tar
        tmp_labels = [tar, ~tar]
        # noinspection PyTypeChecker
        self._con_chaos_cache = [np.sum(label) for label in tmp_labels]
        label_lst = [self._y[label] for label in tmp_labels]
        rs, chaos_lst, xt = 0, [], self._x.T
        append = chaos_lst.append
        for data_label, tar_label in zip(tmp_labels, label_lst):
            tmp_data = xt[data_label]
            if self._sample_weight is None:
                chaos = method(Cluster(tmp_data, tar_label, base=self._base))
            else:
                new_weights = self._sample_weight[data_label]
                chaos = method(Cluster(tmp_data, tar_label, new_weights / np.sum(new_weights), base=self._base))
            rs += len(tmp_data) / len(data) * chaos
            append(chaos)
        return rs, chaos_lst

    def bin_info_gain(self, idx, tar, criterion="gini", get_chaos_lst=False, continuous=False):
        if criterion in ("ent", "ratio"):
            con_chaos, chaos_lst = self.bin_con_chaos(idx, tar, "ent", continuous)
            gain = self.ent() - con_chaos
            if criterion == "ratio":
                # noinspection PyTypeChecker
                gain = gain / self.ent(self._con_chaos_cache)
        elif criterion == "gini":
            con_chaos, chaos_lst = self.bin_con_chaos(idx, tar, "gini", continuous)
            gain = self.gini() - con_chaos
        else:
            raise NotImplementedError("Info_gain criterion '{}' not defined".format(criterion))
        return (gain, chaos_lst) if get_chaos_lst else gain
