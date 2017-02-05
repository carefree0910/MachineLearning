import numpy as np


class DataUtil:
    @staticmethod
    def get_dataset(name, path):
        x = []
        with open(path, "r", encoding="utf8") as file:
            if name == "mushroom" or "balloon" in name:
                for sample in file:
                    x.append(sample.strip().split(","))
            elif name == "bank1.0":
                for sample in file:
                    sample = sample.replace('"', "")
                    x.append(list(map(lambda c: c.strip(), sample.split(";"))))
            else:
                raise NotImplementedError
        return x

    @staticmethod
    def gen_xor(size=100, scale=1):
        x = np.random.randn(size) * scale
        y = np.random.randn(size) * scale
        z = np.zeros((size, 2))
        z[x * y >= 0, :] = [0, 1]
        z[x * y < 0, :] = [1, 0]
        return np.c_[x, y].astype(np.float32), z

    @staticmethod
    def gen_spin(size=20, n=7, n_class=7):
        xs = np.zeros((size * n, 2), dtype=np.float32)
        ys = np.zeros(size * n, dtype=np.int8)
        for j in range(n):
            ix = range(size * j, size * (j + 1))
            r = np.linspace(0.0, 1, size+1)[1:]
            t = np.array(
                np.linspace(j * (n + 1), (j + 1) * (n + 1), size) +
                np.array(np.random.random(size=size)) * 0.2)
            xs[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            ys[ix] = j % n_class
        z = []
        for yy in ys:
            tmp = [0 if i != yy else 1 for i in range(n_class)]
            z.append(tmp)
        return xs, np.array(z)

    @staticmethod
    def quantize_data(x, y, wc=None, continuous_rate=0.1, separate=False):
        if isinstance(x, list):
            xt = map(list, zip(*x))
        else:
            xt = x.T
        features = [set(feat) for feat in xt]
        if wc is None:
            wc = np.array([len(feat) >= continuous_rate * len(y) for feat in features])
        feat_dics = [{_l: i for i, _l in enumerate(feats)} if not wc[i] else None
                     for i, feats in enumerate(features)]
        label_dic = {_l: i for i, _l in enumerate(set(y))}
        if not separate:
            if np.all(~wc):
                dtype = np.int
            else:
                dtype = np.double
            x = np.array([[feat_dics[i][_l] if not wc[i] else _l for i, _l in enumerate(sample)]
                          for sample in x], dtype=dtype)
        else:
            x = np.array([[feat_dics[i][_l] if not wc[i] else _l for i, _l in enumerate(sample)]
                          for sample in x], dtype=np.double)
            x = (x[:, ~wc].astype(np.int), x[:, wc])
        y = np.array([label_dic[yy] for yy in y], dtype=np.int8)
        label_dic = {i: _l for _l, i in label_dic.items()}
        return x, y, wc, features, feat_dics, label_dic
