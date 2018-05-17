import os
import cv2
import math
import pickle
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from math import pi, sqrt, ceil
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

plt.switch_backend("Qt5Agg")
np.random.seed(142857)


class Util:
    @staticmethod
    def callable(obj):
        _str_obj = str(obj)
        if callable(obj):
            return True
        if "<" not in _str_obj and ">" not in _str_obj:
            return False
        if _str_obj.find("function") >= 0 or _str_obj.find("staticmethod") >= 0:
            return True

    @staticmethod
    def freeze_graph(sess, ckpt, output):
        print("Loading checkpoint...")
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)
        print("Writing graph...")
        if not os.path.isdir("_Cache"):
            os.makedirs("_Cache")
        _dir = os.path.join("_Cache", "Model")
        saver.save(sess, _dir)
        graph_io.write_graph(sess.graph, "_Cache", "Model.pb", False)
        print("Freezing graph...")
        freeze_graph.freeze_graph(
            os.path.join("_Cache", "Model.pb"),
            "", True, os.path.join("_Cache", "Model"),
            output, "save/restore_all", "save/Const:0", "Frozen.pb", True, ""
        )
        print("Done")

    @staticmethod
    def load_frozen_graph(graph_dir, fix_nodes=True, entry=None, output=None):
        with gfile.FastGFile(graph_dir, "rb") as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
            if fix_nodes:
                for node in graph_def.node:
                    if node.op == 'RefSwitch':
                        node.op = 'Switch'
                        for index in range(len(node.input)):
                            if 'moving_' in node.input[index]:
                                node.input[index] = node.input[index] + '/read'
                    elif node.op == 'AssignSub':
                        node.op = 'Sub'
                        if 'use_locking' in node.attr:
                            del node.attr['use_locking']
            tf.import_graph_def(graph_def, name="")
            if entry is not None:
                entry = tf.get_default_graph().get_tensor_by_name(entry)
            if output is not None:
                output = tf.get_default_graph().get_tensor_by_name(output)
            return entry, output


class DataUtil:
    naive_sets = {
        "mushroom", "balloon", "mnist", "cifar", "test"
    }

    @staticmethod
    def is_naive(name):
        for naive_dataset in DataUtil.naive_sets:
            if naive_dataset in name:
                return True
        return False

    @staticmethod
    def get_dataset(name, path, n_train=None, tar_idx=None, shuffle=True,
                    quantize=False, quantized=False, one_hot=False, **kwargs):
        x = []
        with open(path, "r", encoding="utf8") as file:
            if DataUtil.is_naive(name):
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
        if quantized:
            x = np.asarray(x, dtype=np.float32)
            y = y.astype(np.int8)
            if one_hot:
                y = (y[..., None] == np.arange(np.max(y) + 1))
        else:
            x = np.asarray(x)
        if quantized or not quantize:
            if n_train is None:
                return x, y
            return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])
        x, y, wc, features, feat_dicts, label_dict = DataUtil.quantize_data(x, y, **kwargs)
        if one_hot:
            y = (y[..., None] == np.arange(np.max(y)+1)).astype(np.int8)
        if n_train is None:
            return x, y, wc, features, feat_dicts, label_dict
        return (
            (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:]),
            wc, features, feat_dicts, label_dict
        )

    @staticmethod
    def get_one_hot(y, n_class):
        one_hot = np.zeros([len(y), n_class])
        one_hot[range(len(y)), y] = 1
        return one_hot

    @staticmethod
    def gen_xor(size=100, scale=1, one_hot=True):
        x = np.random.randn(size) * scale
        y = np.random.randn(size) * scale
        z = np.zeros((size, 2))
        z[x * y >= 0, :] = [0, 1]
        z[x * y < 0, :] = [1, 0]
        if one_hot:
            return np.c_[x, y].astype(np.float32), z
        return np.c_[x, y].astype(np.float32), np.argmax(z, axis=1)

    @staticmethod
    def gen_spiral(size=50, n=7, n_class=7, scale=4, one_hot=True):
        xs = np.zeros((size * n, 2), dtype=np.float32)
        ys = np.zeros(size * n, dtype=np.int8)
        for i in range(n):
            ix = range(size * i, size * (i + 1))
            r = np.linspace(0.0, 1, size+1)[1:]
            t = np.linspace(2 * i * pi / n, 2 * (i + scale) * pi / n, size) + np.random.random(size=size) * 0.1
            xs[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            ys[ix] = i % n_class
        if not one_hot:
            return xs, ys
        return xs, DataUtil.get_one_hot(ys, n_class)

    @staticmethod
    def gen_random(size=100, n_dim=2, n_class=2, scale=1, one_hot=True):
        xs = np.random.randn(size, n_dim).astype(np.float32) * scale
        ys = np.random.randint(n_class, size=size).astype(np.int8)
        if not one_hot:
            return xs, ys
        return xs, DataUtil.get_one_hot(ys, n_class)

    @staticmethod
    def gen_two_clusters(size=100, n_dim=2, center=0, dis=2, scale=1, one_hot=True):
        center1 = (np.random.random(n_dim) + center - 0.5) * scale + dis
        center2 = (np.random.random(n_dim) + center - 0.5) * scale - dis
        cluster1 = (np.random.randn(size, n_dim) + center1) * scale
        cluster2 = (np.random.randn(size, n_dim) + center2) * scale
        data = np.vstack((cluster1, cluster2)).astype(np.float32)
        labels = np.array([1] * size + [0] * size)
        indices = np.random.permutation(size * 2)
        data, labels = data[indices], labels[indices]
        if not one_hot:
            return data, labels
        return data, DataUtil.get_one_hot(labels, 2)

    @staticmethod
    def gen_simple_non_linear(size=120, one_hot=True):
        xs = np.random.randn(size, 2).astype(np.float32) * 1.5
        ys = np.zeros(size, dtype=np.int8)
        mask = xs[..., 1] >= xs[..., 0] ** 2
        xs[..., 1][mask] += 2
        ys[mask] = 1
        if not one_hot:
            return xs, ys
        return xs, DataUtil.get_one_hot(ys, 2)

    @staticmethod
    def gen_nine_grid(size=120, one_hot=True):
        x, y = np.random.randn(2, size).astype(np.float32)
        labels = np.zeros(size, np.int8)
        xl, xr = x <= -1, x >= 1
        yf, yc = y <= -1, y >= 1
        x_mid_mask = ~xl & ~xr
        y_mid_mask = ~yf & ~yc
        mask2 = x_mid_mask & y_mid_mask
        labels[mask2] = 2
        labels[(x_mid_mask | y_mid_mask) & ~mask2] = 1
        xs = np.vstack([x, y]).T
        if not one_hot:
            return xs, labels
        return xs, DataUtil.get_one_hot(labels, 3)

    @staticmethod
    def gen_x_set(size=1000, centers=(1, 1), slopes=(1, -1), gaps=(0.1, 0.1), one_hot=True):
        xc, yc = centers
        x, y = (2 * np.random.random([size, 2]) + np.asarray(centers) - 1).T.astype(np.float32)
        l1 = (-slopes[0] * (x - xc) + y - yc) > 0
        l2 = (-slopes[1] * (x - xc) + y - yc) > 0
        labels = np.zeros(size, dtype=np.int8)
        mask = (l1 & ~l2) | (~l1 & l2)
        labels[mask] = 1
        x[mask] += gaps[0] * np.sign(x[mask] - centers[0])
        y[~mask] += gaps[1] * np.sign(y[~mask] - centers[1])
        xs = np.vstack([x, y]).T
        if not one_hot:
            return xs, labels
        return xs, DataUtil.get_one_hot(labels, 2)

    @staticmethod
    def gen_noisy_linear(size=10000, n_dim=100, n_valid=5, noise_scale=0.5, test_ratio=0.15, one_hot=True):
        x_train = np.random.randn(size, n_dim)
        x_train_noise = x_train + np.random.randn(size, n_dim) * noise_scale
        x_test = np.random.randn(int(size*test_ratio), n_dim)
        idx = np.random.permutation(n_dim)[:n_valid]
        w = np.random.randn(n_valid, 1)
        y_train = (x_train[..., idx].dot(w) > 0).astype(np.int8).ravel()
        y_test = (x_test[..., idx].dot(w) > 0).astype(np.int8).ravel()
        if not one_hot:
            return (x_train_noise, y_train), (x_test, y_test)
        return (x_train_noise, DataUtil.get_one_hot(y_train, 2)), (x_test, DataUtil.get_one_hot(y_test, 2))

    @staticmethod
    def gen_noisy_poly(size=10000, p=3, n_dim=100, n_valid=5, noise_scale=0.5, test_ratio=0.15, one_hot=True):
        p = int(p)
        assert p > 1, "p should be greater than 1"
        x_train = np.random.randn(size, n_dim)
        x_train_list = [x_train] + [x_train ** i for i in range(2, p+1)]
        x_train_noise = x_train + np.random.randn(size, n_dim) * noise_scale
        x_test = np.random.randn(int(size * test_ratio), n_dim)
        x_test_list = [x_test] + [x_test ** i for i in range(2, p+1)]
        idx_list = [np.random.permutation(n_dim)[:n_valid] for _ in range(p)]
        w_list = [np.random.randn(n_valid, 1) for _ in range(p)]
        o_train = [x[..., idx].dot(w) for x, idx, w in zip(x_train_list, idx_list, w_list)]
        o_test = [x[..., idx].dot(w) for x, idx, w in zip(x_test_list, idx_list, w_list)]
        y_train = (np.sum(o_train, axis=0) > 0).astype(np.int8).ravel()
        y_test = (np.sum(o_test, axis=0) > 0).astype(np.int8).ravel()
        if not one_hot:
            return (x_train_noise, y_train), (x_test, y_test)
        return (x_train_noise, DataUtil.get_one_hot(y_train, 2)), (x_test, DataUtil.get_one_hot(y_test, 2))

    @staticmethod
    def gen_special_linear(size=10000, n_dim=10, n_redundant=3, n_categorical=3,
                           cv_ratio=0.15, test_ratio=0.15, one_hot=True):
        x_train = np.random.randn(size, n_dim)
        x_train_redundant = np.ones([size, n_redundant]) * np.random.randint(0, 3, n_redundant)
        x_train_categorical = np.random.randint(3, 8, [size, n_categorical])
        x_train_stacked = np.hstack([x_train, x_train_redundant, x_train_categorical])
        n_test = int(size * test_ratio)
        x_test = np.random.randn(n_test, n_dim)
        x_test_redundant = np.ones([n_test, n_redundant]) * np.random.randint(3, 6, n_redundant)
        x_test_categorical = np.random.randint(0, 5, [n_test, n_categorical])
        x_test_stacked = np.hstack([x_test, x_test_redundant, x_test_categorical])
        w = np.random.randn(n_dim, 1)
        y_train = (x_train.dot(w) > 0).astype(np.int8).ravel()
        y_test = (x_test.dot(w) > 0).astype(np.int8).ravel()
        n_cv = int(size * cv_ratio)
        x_train_stacked, x_cv_stacked = x_train_stacked[:-n_cv], x_train_stacked[-n_cv:]
        y_train, y_cv = y_train[:-n_cv], y_train[-n_cv:]
        if not one_hot:
            return (x_train_stacked, y_train), (x_cv_stacked, y_cv), (x_test_stacked, y_test)
        return (
            (x_train_stacked, DataUtil.get_one_hot(y_train, 2)),
            (x_cv_stacked, DataUtil.get_one_hot(y_cv, 2)),
            (x_test_stacked, DataUtil.get_one_hot(y_test, 2))
        )

    @staticmethod
    def quantize_data(x, y, wc=None, continuous_rate=0.1, separate=False):
        if isinstance(x, list):
            xt = map(list, zip(*x))
        else:
            xt = x.T
        features = [set(feat) for feat in xt]
        if wc is None:
            wc = np.array([len(feat) >= int(continuous_rate * len(y)) for feat in features])
        else:
            wc = np.asarray(wc)
        feat_dicts = [
            {_l: i for i, _l in enumerate(feats)} if not wc[i] else None
            for i, feats in enumerate(features)
        ]
        if not separate:
            if np.all(~wc):
                dtype = np.int
            else:
                dtype = np.float32
            x = np.array([[feat_dicts[i][_l] if not wc[i] else _l for i, _l in enumerate(sample)]
                          for sample in x], dtype=dtype)
        else:
            x = np.array([[feat_dicts[i][_l] if not wc[i] else _l for i, _l in enumerate(sample)]
                          for sample in x], dtype=np.float32)
            x = (x[:, ~wc].astype(np.int), x[:, wc])
        label_dict = {l: i for i, l in enumerate(set(y))}
        y = np.array([label_dict[yy] for yy in y], dtype=np.int8)
        label_dict = {i: l for l, i in label_dict.items()}
        return x, y, wc, features, feat_dicts, label_dict

    @staticmethod
    def transform_data(x, y, wc, feat_dicts, label_dict):
        if np.all(~wc):
            dtype = np.int
        else:
            dtype = np.float32
        label_dict = {l: i for i, l in label_dict.items()}
        x = np.array([[feat_dicts[i][_l] if not wc[i] else _l for i, _l in enumerate(sample)]
                      for sample in x], dtype=dtype)
        y = np.array([label_dict[yy] for yy in y], dtype=np.int8)
        return x, y


class VisUtil:
    @staticmethod
    def get_colors(line, all_positive):
        # c_base = 60
        # colors = []
        # for weight in line:
        #     colors.append([int(255 * (1 - weight)), int(255 - c_base * abs(1 - 2 * weight)), int(255 * weight)])
        # return colors
        # noinspection PyTypeChecker
        colors = np.full([len(line), 3], [0, 195, 255], dtype=np.uint8)
        if all_positive:
            return colors.tolist()
        colors[line < 0] = [255, 195, 0]
        return colors.tolist()

    @staticmethod
    def get_line_info(weight, max_thickness=4, threshold=0.2):
        w_min, w_max = np.min(weight), np.max(weight)
        if w_min >= 0:
            weight -= w_min
            all_pos = True
        else:
            all_pos = False
        weight /= max(w_max, -w_min)
        masks = np.abs(weight) >= threshold  # type: np.ndarray
        colors = [VisUtil.get_colors(line, all_pos) for line in weight]
        thicknesses = np.array(
            [[int((max_thickness - 1) * abs(n)) + 1 for n in line] for line in weight]
        )
        return colors, thicknesses, masks

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

    @staticmethod
    def make_mp4(ims, name="", fps=20, scale=1, extend=30):
        print("Making mp4...")
        ims += [ims[-1]] * extend
        with imageio.get_writer("{}.mp4".format(name), mode='I', fps=fps) as writer:
            for im in ims:
                if scale != 1:
                    new_shape = (int(im.shape[1] * scale), int(im.shape[0] * scale))
                    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
                    im = cv2.resize(im, new_shape, interpolation=interpolation)
                writer.append_data(im[..., ::-1])
        print("Done")


class Overview:
    def __init__(self, label_dict, shape=(1440, 576)):
        self.shape = shape
        self.label_dict = label_dict
        self.n_col = self.n_row = 0
        self.ans = self.pred = self.results = self.prob = None

    def _get_detail(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            w, h = self.shape
            pw, ph = w / self.n_col, h / self.n_row
            idx = int(x // pw + self.n_col * (y // ph))
            prob = self.prob[idx]
            if self.ans is None or self.ans[idx] == self.pred[idx]:
                title = "Detail (prob: {:6.4})".format(prob)
            else:
                title = "True label: {} (prob: {:6.4})".format(
                    self.label_dict[self.ans[idx]], prob)
            while 1:
                cv2.imshow(title, self.results[idx])
                if cv2.waitKey(20) & 0xFF == 27:
                    break
            cv2.destroyWindow(title)

    def _get_results(self, ans, y_pred, images):
        y_pred = np.exp(y_pred)
        y_pred /= np.sum(y_pred, axis=1, keepdims=True)
        pred_classes = np.argmax(y_pred, axis=1)
        if ans is not None:
            true_classes = np.argmax(ans, axis=1)
            true_prob = y_pred[range(len(y_pred)), true_classes]
        else:
            true_classes = None
            true_prob = y_pred[range(len(y_pred)), pred_classes]
        self.ans, self.pred, self.prob = true_classes, pred_classes, true_prob
        c_base = 60
        results = []
        for i, img in enumerate(images):
            pred = y_pred[i]
            indices = np.argsort(pred)[-3:][::-1]
            ps, labels = pred[indices], self.label_dict[indices]
            if true_classes is None:
                color = np.array([255, 255, 255], dtype=np.uint8)
            else:
                p = ps[0]
                if p <= 1 / 2:
                    _l, _r = 2 * c_base + (255 - 2 * c_base) * 2 * p, c_base + (255 - c_base) * 2 * p
                else:
                    _l, _r = 255, 510 * (1 - p)
                if true_classes[i] == pred_classes[i]:
                    color = np.array([0, _l, _r], dtype=np.uint8)
                else:
                    color = np.array([0, _r, _l], dtype=np.uint8)
            rs = np.zeros((256, 640, 3), dtype=np.uint8)
            img = cv2.resize(img, (256, 256))
            rs[:, :256] = img
            rs[:, 256:] = color
            bar_len = 180
            for j, (p, _label) in enumerate(zip(ps, labels)):
                cv2.putText(rs, _label, (288, 64 + 64 * j), cv2.LINE_AA, 0.6, (0, 0, 0), 1)
                cv2.rectangle(rs, (420, 49 + 64 * j), (420 + int(bar_len * p), 69 + 64 * j), (125, 0, 125), -1)
            results.append(rs)
        return results

    def run(self, ans, y_pred, images):
        print("-" * 30)
        print("Visualizing results...")
        results = self._get_results(ans, y_pred, images)
        n_row = math.ceil(math.sqrt(len(results)))  # type: int
        n_col = math.ceil(len(results) / n_row)
        pictures = []
        for i in range(n_row):
            if i == n_row - 1:
                pictures.append(np.hstack(
                    [*results[i * n_col:], np.zeros((256, 640 * (n_row * n_col - len(results)), 3)) + 255]).astype(
                    np.uint8))
            else:
                pictures.append(np.hstack(
                    results[i * n_col:(i + 1) * n_col]).astype(np.uint8))
        self.results = results
        self.n_row, self.n_col = n_row, n_col
        big_canvas = np.vstack(pictures).astype(np.uint8)
        overview = cv2.resize(big_canvas, self.shape)

        cv2.namedWindow("Overview")
        cv2.setMouseCallback("Overview", self._get_detail)
        cv2.imshow("Overview", overview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("-" * 30)
        print("Done")
