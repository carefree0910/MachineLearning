import os
import cv2
import math
import time
import shutil
import imghdr
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from Util.ProgressBar import ProgressBar


class Config:
    n_class = 19
    extractors_path = "Models/Extractors/"
    predictors_path = "Models/Predictors/"
    image_paths = ["sources/{:04d}".format(i) for i in range(n_class)]

    @staticmethod
    def get_image_paths():
        new_paths = []
        for i, path in enumerate(Config.image_paths):
            files = [path + "/" + file for file in os.listdir(path)]
            new_paths += files
        return new_paths

    @staticmethod
    def get_labels():
        labels = []
        for i in range(Config.n_class):
            labels += [[0 if j != i else 1 for j in range(
                Config.n_class)] for _ in range(len(os.listdir("sources/{:04d}".format(i))))]
        np.save("sources/labels", labels)

    @staticmethod
    def shuffle(folder):
        _features = np.load(folder + "/features.npy")
        _labels = np.load(folder + "/labels.npy")
        _indices = np.random.permutation(len(_features))
        _features, _labels = _features[_indices], _labels[_indices]
        np.save(folder + "/features", _features)
        np.save(folder + "/labels", _labels)

    @staticmethod
    def split(folder):
        _features = np.load(folder + "/features.npy")
        _labels = np.load(folder + "/labels.npy")
        _indices = np.random.permutation(len(_features))
        _features, _labels = _features[_indices], _labels[_indices]
        train_len = int(0.9 * len(_features))
        x_train, x_test = _features[:train_len], _features[train_len:]
        y_train, y_test = _labels[:train_len], _labels[train_len:]
        np.save(folder + "/x_train", x_train)
        np.save(folder + "/x_test", x_test)
        np.save(folder + "/y_train", y_train)
        np.save(folder + "/y_test", y_test)


class Extractor:
    def __init__(self, extractor="v3", image_paths=None, labels=None, mat_dir=None):
        self._extractor = extractor if "v3" not in extractor else "v3"
        self._image_paths = Config.get_image_paths() if image_paths is None else image_paths
        self._labels, self._mat_dir = labels, mat_dir

    def _create_graph(self):
        tf.reset_default_graph()
        with gfile.FastGFile(Config.extractors_path+self._extractor+"/Model.pb", "rb") as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
            if "cnn" in self._extractor:
                # Fix nodes
                for node in graph_def.node:
                    if node.op == 'RefSwitch':
                        node.op = 'Switch'
                        for index in range(len(node.input)):
                            if 'moving_' in node.input[index]:
                                node.input[index] += '/read'
                    elif node.op == 'Assign':
                        node.op = 'Sub'
                        if 'use_locking' in node.attr:
                            del node.attr['use_locking']
            tf.import_graph_def(graph_def, name="")

    def _extract(self, verbose):
        features = []
        with tf.Session() as sess:
            if self._extractor == "v3":
                _output = "pool_3:0"
            elif self._extractor == "ResNet-v2":
                _output = "InceptionResnetV2/Logits/Flatten/Reshape:0"
            elif self._extractor == "cnn":
                _output = "final_result/Reshape:0"
            else:
                _output = "OutputFlow/Reshape:0"
            flattened_tensor = sess.graph.get_tensor_by_name(_output)
            if self._extractor == "v3":
                _entry = "DecodeJpeg/contents:0"
            elif self._extractor == "ResNet-v2":
                _entry = "Placeholder:0"
            else:
                _entry = "Entry/Placeholder:0"
            pop_lst = []
            if "cnn" in self._extractor or "ResNet" in self._extractor and self._mat_dir is not None:
                features = np.load(self._mat_dir)
            else:
                def process(img_path):
                    img_data = gfile.FastGFile(img_path, "rb").read()
                    feature = sess.run(flattened_tensor, {
                        _entry: img_data
                    })
                    features.append(np.squeeze(feature))
                for i, image_path in enumerate(self._image_paths):
                    if not os.path.isfile(image_path):
                        continue
                    if "v3" in self._extractor:
                        if verbose:
                            print("Processing {}...".format(image_path))
                        try:
                            process(image_path)
                        except Exception as err:
                            if verbose:
                                print(err)
                            name, extension = os.path.splitext(image_path)
                            base = os.path.basename(image_path)
                            if extension.lower() in (".jpg", ".jpeg"):
                                new_name = name[:image_path.rfind(base)] + "{:06d}{}".format(i, extension)
                                print("Renaming {} to {}...".format(image_path, new_name))
                                os.rename(image_path, new_name)
                                process(new_name)
                            else:
                                new_name_base = name[:image_path.rfind(base)] + "{:06d}".format(i)
                                new_name = new_name_base + ".jpg"
                                print("Transforming {} to {}...".format(image_path, new_name))
                                try:
                                    if imghdr.what(image_path) is None:
                                        raise ValueError("{} is not an image".format(image_path))
                                    os.rename(image_path, new_name_base + extension)
                                    cv2.imwrite(new_name, cv2.imread(new_name_base + extension))
                                    os.remove(new_name_base + extension)
                                    process(new_name)
                                except Exception as err:
                                    print(err)
                                    print("Moving {} to '_err' folder...".format(image_path))
                                    if not os.path.isdir("_err"):
                                        os.makedirs("_err")
                                    shutil.move(image_path, os.path.join("_err", os.path.basename(image_path)))
                                    pop_lst.append(i)
                    else:
                        if verbose:
                            print("Reading {}...".format(image_path))
                        image_data = cv2.imread(image_path)
                        if self._extractor == "ResNet-v2":
                            features.append(cv2.resize(image_data, (299, 299)))
                        else:
                            features.append(cv2.resize(image_data, (64, 64)))
            if "v3" not in self._extractor:
                features = np.array(features)
                print("Extracting features...")
                rs = []
                batch_size = math.floor(1e6 / np.prod(features.shape[1:]))
                epoch = int(math.ceil(len(features) / batch_size))
                bar = ProgressBar(max_value=epoch, name="Extract")
                for i in range(epoch):
                    if i == epoch - 1:
                        rs.append(sess.run(flattened_tensor, {
                            _entry: features[i*batch_size:]
                        }))
                    else:
                        rs.append(sess.run(flattened_tensor, {
                            _entry: features[i*batch_size:(i+1)*batch_size]
                        }))
                    bar.update()
                return np.vstack(rs).astype(np.float32)
            if pop_lst:
                labels = []
                pop_cursor, pop_idx = 0, pop_lst[0]
                for i, label in enumerate(self._labels):
                    if i == pop_idx:
                        pop_cursor += 1
                        if pop_cursor < len(pop_lst):
                            pop_idx = pop_lst[pop_cursor]
                        else:
                            pop_idx = -1
                        continue
                    labels.append(label)
                labels = np.array(labels, dtype=np.float32)
            elif self._labels is None:
                labels = None
            else:
                labels = np.array(self._labels, dtype=np.float32)
            return np.array(features, dtype=np.float32), labels

    def run(self, verbose=True):
        self._create_graph()
        return self._extract(verbose)


class Predictor:
    def __init__(self, predictor="v3"):
        self._predictor = predictor
        self._entry, self._output = "Entry/Placeholder:0", "OutputFlow/add_9:0"

    def _create_graph(self):
        tf.reset_default_graph()
        with gfile.FastGFile(Config.predictors_path+self._predictor+"/Model.pb", "rb") as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
            if "cnn" in self._predictor:
                # Fix nodes
                for node in graph_def.node:
                    if node.op == 'RefSwitch':
                        node.op = 'Switch'
                        for index in range(len(node.input)):
                            if 'moving_' in node.input[index]:
                                node.input[index] += '/read'
                    elif node.op == 'Assign':
                        node.op = 'Sub'
                        if 'use_locking' in node.attr:
                            del node.attr['use_locking']
            tf.import_graph_def(graph_def, name="")

    def predict(self, x):
        self._create_graph()
        x, rs = np.atleast_2d(x).astype(np.float32), []
        with tf.Session() as sess:
            flattened_tensor = sess.graph.get_tensor_by_name(self._output)
            print("Predicting...")
            batch_size = math.floor(1e6 / np.prod(x.shape[1:]))
            epoch = math.ceil(len(x) / batch_size)  # type: int
            bar = ProgressBar(max_value=epoch, name="Predict")
            for i in range(epoch):
                if i == epoch - 1:
                    rs.append(sess.run(flattened_tensor, {
                        self._entry: x[i*batch_size:]
                    }))
                else:
                    rs.append(sess.run(flattened_tensor, {
                        self._entry: x[i*batch_size:(i+1)*batch_size]
                    }))
                bar.update()
            return np.vstack(rs).astype(np.float32)


class Pipeline:
    shape = (1440, 576)

    def __init__(self):
        self._image_paths = []
        self._img_dir = self._rs_dir = None

        self._n_row = self._n_col = None
        self._results = []
        self._ans = self._pred = self._prob = None

    @staticmethod
    def get_image_paths(img_dir, pre_process=False):
        image_paths = [img_dir + "/" + file for file in os.listdir(img_dir)]
        if not pre_process:
            return [img for img in image_paths if os.path.isfile(img)]
        return [img for img in image_paths if os.path.isfile(img) and imghdr.what(img) is not None]

    def _get_results(self, ans, img_dir=None, rs_dir=None):
        if img_dir is None:
            img_dir = self._img_dir
        if rs_dir is None:
            rs_dir = self._rs_dir
        y_pred = np.exp(np.load(rs_dir + "/prediction.npy"))
        y_pred /= np.sum(y_pred, axis=1, keepdims=True)
        pred_classes = np.argmax(y_pred, axis=1)
        if ans is not None:
            true_classes = np.argmax(ans, axis=1)
            true_prob = y_pred[range(len(y_pred)), true_classes]
        else:
            true_classes = None
            true_prob = y_pred[range(len(y_pred)), pred_classes]
        self._ans, self._pred, self._prob = true_classes, pred_classes, true_prob
        images = []
        c_base = 60
        for i, img in enumerate(Pipeline.get_image_paths(img_dir, True)):
            _pred = y_pred[i]
            _indices = np.argsort(_pred)[-3:][::-1]
            label_dic = np.load(os.path.join("_Data", "_Cache", "LABEL_DIC.npy"))
            _ps, _labels = _pred[_indices], label_dic[_indices]
            _img = cv2.imread(img)
            if true_classes is None:
                color = np.array([255, 255, 255], dtype=np.uint8)
            else:
                _p = _ps[0]
                if _p <= 1 / 2:
                    _l, _r = 2 * c_base + (255 - 2 * c_base) * 2 * _p, c_base + (255 - c_base) * 2 * _p
                else:
                    _l, _r = 255, 510 * (1 - _p)
                if true_classes[i] == pred_classes[i]:
                    color = np.array([0, _l, _r], dtype=np.uint8)
                else:
                    color = np.array([0, _r, _l], dtype=np.uint8)
            canvas = np.zeros((256, 640, 3), dtype=np.uint8)
            _img = cv2.resize(_img, (256, 256))
            canvas[:, :256] = _img
            canvas[:, 256:] = color
            bar_len = 180
            for j, (_p, _label) in enumerate(zip(_ps, _labels)):
                cv2.putText(canvas, _label, (288, 64 + 64 * j), cv2.LINE_AA, 0.6, (0, 0, 0), 1)
                cv2.rectangle(canvas, (420, 49 + 64 * j), (420 + int(bar_len * _p), 69 + 64 * j), (125, 0, 125), -1)
            images.append(canvas)
        return images

    def _get_detail(self, event, x, y, *_):
        label_dic = np.load(os.path.join("_Data", "_Cache", "LABEL_DIC.npy"))
        if event == cv2.EVENT_LBUTTONDBLCLK:
            _w, _h = Pipeline.shape
            _pw, _ph = _w / self._n_col, _h / self._n_row
            _idx = int(x // _pw + self._n_col * (y // _ph))
            _prob = self._prob[_idx]
            if self._ans is None or self._ans[_idx] == self._pred[_idx]:
                title = "Detail (prob: {:6.4})".format(_prob)
            else:
                title = "True label: {} (prob: {:6.4})".format(
                    label_dic[self._ans[_idx]], _prob)
            while 1:
                cv2.imshow(title, self._results[_idx])
                if cv2.waitKey(20) & 0xFF == 27:
                    break
            cv2.destroyWindow(title)

    def run(self, images_dir="Test", image_shape=(64, 64), model="v3",
            delete_cache=True, extract_only=False, visualize_only=False, overview=True, verbose=True):
        _t = time.time()
        y_pred = None
        if not visualize_only:
            if model != "v3":
                print("=" * 30)
                print("Resizing images...")
                print("=" * 30)
            else:
                print("=" * 30)
                print("Reading images...")
                print("=" * 30)
            if extract_only:
                self._image_paths = Pipeline.get_image_paths(images_dir)
            else:
                self._image_paths = Pipeline.get_image_paths(images_dir, True)
            rs = []
            _new_path, _mat_dir = images_dir + "/_cache", None
            if not os.path.isdir(_new_path):
                os.makedirs(_new_path)
            if model != "v3":
                for i, img in enumerate(self._image_paths):
                    _img = cv2.imread(img)
                    if model != "ResNet-v2":
                        _img = cv2.resize(_img, image_shape)
                    else:
                        image_shape = (299, 299)
                        _img = cv2.resize(_img, image_shape)/127.5-1
                    if model == "v3(64)":
                        _slash_idx = img.rfind("/")
                        _new_img = _new_path + "/" + img[_slash_idx+1:]
                        cv2.imwrite(_new_img, _img)
                        print("{} transformed to {} with shape {}".format(img, _new_img, image_shape))
                        self._image_paths[i] = _new_img
                    else:
                        print("{} transformed to shape {}".format(img, image_shape))
                        rs.append(_img.astype(np.float32))
                _mat_dir = _new_path + "/_mat.npy"
                np.save(_mat_dir, rs)
            print("Done")
            print("=" * 30)
            print("Using {} to extract features...".format(model))
            print("=" * 30)
            features, _ = Extractor(model, self._image_paths, _mat_dir).run(verbose)
            if extract_only:
                np.save("features", features)
                Pipeline._delete_cache(images_dir)
                return
            print("-" * 30)
            print("Loading predictor...")
            y_pred = Predictor(model).predict(features)
            print("-" * 30)
        self._img_dir = images_dir
        self._rs_dir = images_dir + "/_Result"
        label_dic = np.load(os.path.join("_Data", "_Cache", "LABEL_DIC.npy"))
        if not visualize_only:
            if not os.path.isdir(images_dir + "/_Result"):
                os.makedirs(self._rs_dir)
            np.save(self._rs_dir + "/prediction", y_pred)
            labels = label_dic[np.argmax(y_pred, axis=1)]
            with open(self._rs_dir + "/labels.txt", "w") as file:
                file.write("\n".join(labels))
            print("Done; results saved to '{}' folder".format(self._rs_dir))
            if delete_cache:
                Pipeline._delete_cache(images_dir)
            print("-" * 30)
            print("Done")
        print("(Test) Time cost: {:8.6} s".format(time.time() - _t))
        if overview:
            print("-" * 30)
            print("Visualizing results...")
            if os.path.isfile(images_dir + "/_answer.npy"):
                _ans = np.load(images_dir + "/_answer.npy")
            else:
                _ans = None
            images = self._get_results(_ans)
            n_row = math.ceil(math.sqrt(len(images)))  # type: int
            n_col = math.ceil(len(images) / n_row)
            pictures = []
            for i in range(n_row):
                if i == n_row - 1:
                    pictures.append(np.hstack(
                        [*images[i*n_col:], np.zeros((256, 640*(n_row*n_col-len(images)), 3)) + 255]).astype(np.uint8))
                else:
                    pictures.append(np.hstack(
                        images[i*n_col:(i+1)*n_col]).astype(np.uint8))
            self._results = images
            self._n_row, self._n_col = n_row, n_col
            big_canvas = np.vstack(pictures).astype(np.uint8)
            overview = cv2.resize(big_canvas, Pipeline.shape)

            cv2.namedWindow("Overview")
            cv2.setMouseCallback("Overview", self._get_detail)
            cv2.imshow("Overview", overview)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print("-" * 30)
            print("Done")

    @staticmethod
    def _delete_cache(images_dir):
        print("-" * 30)
        print("Deleting '_cache' folder...")
        shutil.rmtree(images_dir + "/_cache")
