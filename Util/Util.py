import pickle
import numpy as np
from math import pi, sqrt, ceil
import matplotlib.pyplot as plt

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

# np.random.seed(142857)


class Util:

    @staticmethod
    def get_and_pop(dic, key, default):
        try:
            val = dic[key]
            dic.pop(key)
        except KeyError:
            val = default
        return val


class DataUtil:
    @staticmethod
    def get_dataset(name, path, train_num=None, tar_idx=None, shuffle=True, quantize=False, **kwargs):
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
        if shuffle:
            np.random.shuffle(x)
        tar_idx = -1 if tar_idx is None else tar_idx
        y = np.array([xx.pop(tar_idx) for xx in x])
        x = np.array(x)
        if not quantize:
            if train_num is None:
                return x, y
            return (x[:train_num], y[:train_num]), (x[train_num:], y[train_num:])
        x, y, wc, features, feat_dics, label_dic = DataUtil.quantize_data(x, y, **kwargs)
        if train_num is None:
            return x, y, wc, features, feat_dics, label_dic
        return (
            (x[:train_num], y[:train_num]), (x[train_num:], y[train_num:]),
            wc, features, feat_dics, label_dic
        )

    @staticmethod
    def gen_xor(size=100, scale=1):
        x = np.random.randn(size) * scale
        y = np.random.randn(size) * scale
        z = np.zeros((size, 2))
        z[x * y >= 0, :] = [0, 1]
        z[x * y < 0, :] = [1, 0]
        return np.c_[x, y].astype(np.float32), z

    @staticmethod
    def gen_spin(size=50, n=7, n_class=7, scale=4):
        xs = np.zeros((size * n, 2), dtype=np.float32)
        ys = np.zeros(size * n, dtype=np.int8)
        for i in range(n):
            ix = range(size * i, size * (i + 1))
            r = np.linspace(0.0, 1, size+1)[1:]
            t = np.linspace(2 * i * pi / n, 2 * (i + scale) * pi / n, size) + np.random.random(size=size) * 0.1
            xs[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            ys[ix] = i % n_class
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
        label_dic = {_l: i for i, _l in enumerate(set(y))}
        y = np.array([label_dic[yy] for yy in y], dtype=np.int8)
        label_dic = {i: _l for _l, i in label_dic.items()}
        return x, y, wc, features, feat_dics, label_dic


class VisUtil:

    @staticmethod
    def get_line_info(weight, weight_min, weight_max, weight_average, max_thickness=5):
        mask = weight >= weight_average
        min_avg_gap = (weight_average - weight_min)
        max_avg_gap = (weight_max - weight_average)
        weight -= weight_average
        max_mask = mask / max_avg_gap
        min_mask = ~mask / min_avg_gap
        weight *= max_mask + min_mask
        colors = np.array(
            [[(130 - 125 * n, 130, 130 + 125 * n) for n in line] for line in weight]
        )
        thicknesses = np.array(
            [[int((max_thickness - 1) * abs(n)) + 1 for n in line] for line in weight]
        )
        max_thickness = int(max_thickness)
        if max_thickness <= 1:
            max_thickness = 2
        if np.sum(thicknesses == max_thickness) == thicknesses.shape[0] * thicknesses.shape[1]:
            thicknesses = np.ones(thicknesses.shape, dtype=np.uint8)
        return colors, thicknesses

    @staticmethod
    def get_graphs_from_logs():
        with open("Results/logs.dat", "rb") as file:
            logs = pickle.load(file)
        for (hus, ep, bt), log in logs.items():
            hus = list(map(lambda _c: str(_c), hus))
            title = "hus: {} ep: {} bt: {}".format(
                "- " + " -> ".join(hus) + " -", ep, bt
            )
            fb_log, acc_log = log["fb_log"], log["acc_log"]
            xs = np.arange(len(fb_log)) + 1
            plt.figure()
            plt.title(title)
            plt.plot(xs, fb_log)
            plt.plot(xs, acc_log, c="g")
            plt.savefig("Results/img/" + "{}_{}_{}".format(
                "-".join(hus), ep, bt
            ))
            plt.close()

    @staticmethod
    def show_img(img, title, normalize=True):
        if normalize:
            img_max, img_min = np.max(img), np.min(img)
            img = 255.0 * (img - img_min) / (img_max - img_min)
        plt.figure()
        plt.title(title)
        plt.imshow(img.astype('uint8'), cmap=plt.cm.gray)
        plt.gca().axis('off')
        plt.show()

    @staticmethod
    def show_batch_img(batch_img, title, normalize=True):
        _n, height, width = batch_img.shape
        a = int(ceil(sqrt(_n)))
        g = np.ones((a * height + a, a * width + a), batch_img.dtype)
        g *= np.min(batch_img)
        _i = 0
        for y in range(a):
            for x in range(a):
                if _i < _n:
                    g[y * height + y:(y + 1) * height + y, x * width + x:(x + 1) * width + x] = batch_img[_i, :, :]
                    _i += 1
        max_g = g.max()
        min_g = g.min()
        g = (g - min_g) / (max_g - min_g)
        VisUtil.show_img(g, title, normalize)

    @staticmethod
    def trans_img(img, shape=None):
        if shape is not None:
            img = img.reshape(shape)
        if img.shape[0] == 1:
            return img.reshape(img.shape[1:])
        return img.transpose(1, 2, 0)
