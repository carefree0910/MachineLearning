import os
import cv2
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

np.random.seed(142857)


class Util:
    @staticmethod
    def get_and_pop(dic, key, default):
        try:
            val = dic[key]
            dic.pop(key)
        except KeyError:
            val = default
        return val

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
    def get_dataset(name, path, train_num=None, tar_idx=None, shuffle=True,
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
            if one_hot:
                z = []
                y = y.astype(np.int8)
                for yy in y:
                    z.append([0 if i != yy else 1 for i in range(np.max(y) + 1)])
                y = np.asarray(z, dtype=np.int8)
            else:
                y = y.astype(np.int8)
        else:
            x = np.asarray(x)
        if quantized or not quantize:
            if train_num is None:
                return x, y
            return (x[:train_num], y[:train_num]), (x[train_num:], y[train_num:])
        x, y, wc, features, feat_dics, label_dic = DataUtil.quantize_data(x, y, **kwargs)
        if one_hot:
            z = []
            for yy in y:
                z.append([0 if i != yy else 1 for i in range(len(label_dic))])
            y = np.asarray(z)
        if train_num is None:
            return x, y, wc, features, feat_dics, label_dic
        return (
            (x[:train_num], y[:train_num]), (x[train_num:], y[train_num:]),
            wc, features, feat_dics, label_dic
        )

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
        return xs, np.array(ys[..., None] == np.arange(n_class), dtype=np.int8)

    @staticmethod
    def gen_random(size=100, n_dim=2, n_class=2, one_hot=True):
        xs = np.random.randn(size, n_dim).astype(np.float32)
        ys = np.random.randint(n_class, size=size).astype(np.int8)
        if not one_hot:
            return xs, ys
        return xs, np.array(ys[..., None] == np.arange(n_class), dtype=np.int8)

    @staticmethod
    def gen_two_clusters(size=100, n_dim=2, center=0, dis=2, scale=1, one_hot=True):
        center1 = (np.random.random(n_dim) + center - 0.5) * scale + dis
        center2 = (np.random.random(n_dim) + center - 0.5) * scale - dis
        cluster1 = (np.random.randn(size, n_dim) + center1) * scale
        cluster2 = (np.random.randn(size, n_dim) + center2) * scale
        data = np.vstack((cluster1, cluster2)).astype(np.float32)
        labels = np.array([1] * size + [0] * size)
        _indices = np.random.permutation(size * 2)
        data, labels = data[_indices], labels[_indices]
        if not one_hot:
            return data, labels
        labels = np.array([[0, 1] if label == 1 else [1, 0] for label in labels], dtype=np.int8)
        return data, labels

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
    def get_colors(line, all_pos):
        # c_base = 60
        # colors = []
        # for weight in line:
        #     colors.append([int(255 * (1 - weight)), int(255 - c_base * abs(1 - 2 * weight)), int(255 * weight)])
        # return colors
        # noinspection PyTypeChecker
        colors = np.full([len(line), 3], [0, 195, 255], dtype=np.uint8)
        if all_pos:
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
    def make_mp4(ims, name="", fps=1, scale=1):
        print("Making mp4...")
        with imageio.get_writer("{}.mp4".format(name), mode='I', fps=fps) as writer:
            for im in ims:
                if scale != 1:
                    new_shape = (int(im.shape[1] * scale), int(im.shape[0] * scale))
                    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
                    im = cv2.resize(im, new_shape, interpolation=interpolation)
                writer.append_data(im[..., ::-1])
        print("Done")
