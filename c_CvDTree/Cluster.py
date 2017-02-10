import math
import numpy as np


class Cluster:
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

    def ent(self, ent=None, eps=1e-12):
        if self._ent_cache is not None and ent is None:
            return self._ent_cache
        _len = len(self._y)
        if ent is None:
            ent = self._counters
        _ent_cache = max(eps, -sum(
            [_c / _len * math.log(_c / _len, self._base) if _c != 0 else 0 for _c in ent]))
        if ent is None:
            self._ent_cache = _ent_cache
        return _ent_cache

    def gini(self, p=None):
        if self._gini_cache is not None and p is None:
            return self._gini_cache
        if p is None:
            p = self._counters
        _gini_cache = 1 - np.sum((p / len(self._y)) ** 2)
        if p is None:
            self._gini_cache = _gini_cache
        return _gini_cache

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
        self._con_chaos_cache = [np.sum(_label) for _label in tmp_labels]
        label_lst = [self._y[label] for label in tmp_labels]
        rs, chaos_lst = 0, []
        for data_label, tar_label in zip(tmp_labels, label_lst):
            tmp_data = self._x.T[data_label]
            if self._sample_weight is None:
                _chaos = _method(Cluster(tmp_data, tar_label, base=self._base))
            else:
                _new_weights = self._sample_weight[data_label]
                _chaos = _method(Cluster(tmp_data, tar_label, _new_weights / np.sum(_new_weights), base=self._base))
            rs += len(tmp_data) / len(data) * _chaos
            chaos_lst.append(_chaos)
        return rs, chaos_lst

    def info_gain(self, idx, criterion="ent", get_chaos_lst=False, features=None):
        if criterion in ("ent", "ratio"):
            _con_chaos, _chaos_lst = self.con_chaos(idx, criterion="ent", features=features)
            _gain = self.ent() - _con_chaos
            if criterion == "ratio":
                _gain /= self.ent(self._con_chaos_cache)
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
        self._con_chaos_cache = [np.sum(_label) for _label in tmp_labels]
        label_lst = [self._y[label] for label in tmp_labels]
        rs, chaos_lst = 0, []
        for data_label, tar_label in zip(tmp_labels, label_lst):
            tmp_data = self._x.T[data_label]
            if self._sample_weight is None:
                _chaos = _method(Cluster(tmp_data, tar_label, base=self._base))
            else:
                _new_weights = self._sample_weight[data_label]
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
                _gain = _gain / self.ent(self._con_chaos_cache)
        elif criterion == "gini":
            _con_chaos, _chaos_lst = self.bin_con_chaos(idx, tar, "gini", continuous)
            _gain = self.gini() - _con_chaos
        else:
            raise NotImplementedError("Info_gain criterion '{}' not defined".format(criterion))
        return (_gain, _chaos_lst) if get_chaos_lst else _gain
