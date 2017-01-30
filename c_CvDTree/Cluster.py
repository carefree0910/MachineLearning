import math
import numpy as np


class Cluster:
    def __init__(self, x, y, sample_weights=None, base=2):
        self._x = np.array(x).T
        self._y = np.array(y)
        if sample_weights is None:
            self._counters = np.bincount(self._y)
        else:
            # noinspection PyTypeChecker
            self._counters = np.bincount(self._y, weights=sample_weights*len(sample_weights))
        self._sample_weights = sample_weights
        self._cache = None
        self._base = base

    def ent(self, ent=None, eps=1e-12):
        _len = len(self._y)
        if ent is None:
            ent = self._counters
        return max(eps, -sum([_c / _len * math.log(_c / _len, self._base) if _c != 0 else 0 for _c in ent]))

    def gini(self, p=None):
        if p is None:
            p = self._counters
        return 1 - np.sum((p / len(self._y)) ** 2)

    def con_chaos(self, idx, criterion="ent", features=None):
        if criterion == "ent":
            _method = lambda cluster: cluster.ent()
        elif criterion == "gini":
            _method = lambda cluster: cluster.gini()
        else:
            raise NotImplementedError("Conditional info criterion '{}' not defined".format(criterion))
        data = self._x[idx]
        if features is None:
            features = set(data)
        tmp_labels = [data == feature for feature in features]
        # noinspection PyTypeChecker
        self._cache = [np.sum(_label) for _label in tmp_labels]
        label_lst = [self._y[label] for label in tmp_labels]
        rs, chaos_lst = 0, []
        for data_label, tar_label in zip(tmp_labels, label_lst):
            tmp_data = self._x.T[data_label]
            if self._sample_weights is None:
                _chaos = _method(Cluster(tmp_data, tar_label, base=self._base))
            else:
                _new_weights = self._sample_weights[data_label]
                _chaos = _method(Cluster(tmp_data, tar_label, _new_weights / np.sum(_new_weights), base=self._base))
            rs += len(tmp_data) / len(data) * _chaos
            chaos_lst.append(_chaos)
        return rs, chaos_lst

    def info_gain(self, idx, criterion="ent", get_chaos_lst=False, features=None):
        if criterion in ("ent", "ratio"):
            _con_chaos, _chaos_lst = self.con_chaos(idx, criterion="ent", features=features)
            _gain = self.ent() - _con_chaos
            if criterion == "ratio":
                _gain /= self.ent([np.sum(_cache) for _cache in self._cache])
        elif criterion == "gini":
            _con_chaos, _chaos_lst = self.con_chaos(idx, criterion="gini", features=features)
            _gain = self.gini() - _con_chaos
        else:
            raise NotImplementedError("Info_gain criterion '{}' not defined".format(criterion))
        return (_gain, _chaos_lst) if get_chaos_lst else _gain

    def bin_con_chaos(self, idx, tar, criterion="gini", continuous=False):
        if criterion == "ent":
            _method = lambda cluster: cluster.ent()
        elif criterion == "gini":
            _method = lambda cluster: cluster.gini()
        else:
            raise NotImplementedError("Conditional info criterion '{}' not defined".format(criterion))
        data = self._x[idx]
        tar = data == tar if not continuous else data < tar
        tmp_labels = [tar, ~tar]
        # noinspection PyTypeChecker
        self._cache = [np.sum(_label) for _label in tmp_labels]
        label_lst = [self._y[label] for label in tmp_labels]
        rs, chaos_lst = 0, []
        for data_label, tar_label in zip(tmp_labels, label_lst):
            tmp_data = self._x.T[data_label]
            if self._sample_weights is None:
                _chaos = _method(Cluster(tmp_data, tar_label, base=self._base))
            else:
                _new_weights = self._sample_weights[data_label]
                _chaos = _method(Cluster(tmp_data, tar_label, _new_weights / np.sum(_new_weights), base=self._base))
            rs += len(tmp_data) / len(data) * _chaos
            chaos_lst.append(_chaos)
        return rs, chaos_lst

    def bin_info_gain(self, idx, tar, criterion="gini", get_chaos_lst=False, continuous=False):
        if criterion in ("ent", "ratio"):
            _con_chaos, _chaos_lst = self.bin_con_chaos(idx, tar, "ent", continuous)
            _gain = self.ent() - _con_chaos
            if criterion == "ratio":
                # noinspection PyTypeChecker
                _gain = _gain / self.ent(self._cache)
        elif criterion == "gini":
            _con_chaos, _chaos_lst = self.bin_con_chaos(idx, tar, "gini", continuous)
            _gain = self.gini() - _con_chaos
        else:
            raise NotImplementedError("Info_gain criterion '{}' not defined".format(criterion))
        return (_gain, _chaos_lst) if get_chaos_lst else _gain
