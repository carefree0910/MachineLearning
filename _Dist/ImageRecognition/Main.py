import os
import sys
import time
import shutil
import imghdr
import argparse
import numpy as np
import tensorflow as tf

from _Dist.ImageRecognition.ToolBox import Pipeline, Extractor
from NN.NN import NNDist


def fetch_img_data(name="_Data"):
    img_paths, labels = [], []
    data_folder_lst = os.listdir(name)
    for i in range(len(data_folder_lst) - 1, -1, -1):
        if not os.path.isdir(os.path.join(name, data_folder_lst[i])):
            data_folder_lst.pop(i)
        elif data_folder_lst[i] == "_Cache":
            data_folder_lst.pop(i)
    label_dic_dir = os.path.join(name, "_Cache", "LABEL_DIC")
    if not os.path.isfile(label_dic_dir):
        if not os.path.isdir(os.path.join(name, "_Cache")):
            os.makedirs(os.path.join(name, "_Cache"))
        np.save(label_dic_dir, np.array(data_folder_lst))
    for i, folder in enumerate(data_folder_lst):
        for img in os.listdir(os.path.join(name, folder)):
            img_dir = os.path.join(name, folder, img)
            if not os.path.isfile(img_dir):
                continue
            if imghdr.what(img_dir) is None:
                continue
            img_paths.append(img_dir)
            labels.append(i)
    max_label = max(labels)  # type: int
    labels = np.array(
        [[0 if i != yy else 1 for i in range(max_label + 1)] for yy in labels],
        dtype=np.float32
    )
    return img_paths, labels


def main(_):
    if FLAGS.gen_test:
        test_list = [
            file for file in os.listdir("Test") if os.path.isfile(os.path.join("Test", file)) and imghdr.what(
                os.path.join("Test", file)) is not None
        ]
        if test_list:
            print("Test set already exists")
        else:
            print("Generating Test set...")
            img_paths, labels = fetch_img_data()
            n_test = min(196, int(0.2 * len(labels)))
            _indices = np.random.choice(len(labels), n_test, replace=False)
            img_paths = np.array(img_paths)[_indices]
            labels = labels[_indices]
            for i, _path in enumerate(img_paths):
                base_name = os.path.basename(_path)
                shutil.move(_path, os.path.join("Test", "{:04d}{}".format(i, base_name[base_name.rfind("."):])))
            np.save("Test/_answer", labels)
            print("Done")
    predictor_dir = os.path.join("Models", "Predictors", FLAGS.model, "Model.pb")
    if not os.path.isfile(predictor_dir):
        _t = time.time()
        print("Predictor not found, training with images in '_Data' folder...")
        if not os.path.isfile("_Data/_Cache/features.npy") or not os.path.isfile("_Data/_Cache/labels.npy"):
            img_paths, labels = fetch_img_data()
            extractor = Extractor(FLAGS.model, img_paths, labels)
            features, labels = extractor.run()
            if not os.path.isdir("_Data/_Cache"):
                os.makedirs("_Data/_Cache")
            _indices = np.random.permutation(len(labels))
            features = features[_indices]
            labels = labels[_indices]
            np.save("_Data/_Cache/features", features)
            np.save("_Data/_Cache/labels", labels)
        else:
            features, labels = np.load("_Data/_Cache/features.npy"), np.load("_Data/_Cache/labels.npy")
        print("=" * 30)
        print("Training Neural Network...")
        print("=" * 30)
        nn = NNDist()
        nn.add("ReLU", (features.shape[1], 1024), std=0.001, init=0)
        nn.add("Normalize")
        nn.add("Dropout")
        nn.add("ReLU", (1024,), std=0.001, init=0)
        nn.add("Normalize")
        nn.add("Dropout")
        nn.add("ReLU", (512,), std=0.001, init=0)
        nn.add("Normalize")
        nn.add("Dropout")
        nn.add("CrossEntropy", (labels.shape[1],))
        nn.fit(features, labels, lr=0.0001, epoch=25, verbose=1)
        nn.save()
        if not os.path.isdir(os.path.join("Models", "Predictors", FLAGS.model)):
            os.makedirs(os.path.join("Models", "Predictors", FLAGS.model))
        print("Moving 'Frozen.pb' to {}...".format(predictor_dir))
        shutil.move(os.path.join("Models", "Cache", "Frozen.pb"), predictor_dir)
        print("Removing 'Cache' folder...")
        shutil.rmtree(os.path.join("Models", "Cache"))
        print("Done")
        print("-" * 30)
        print("(Train) Time cost: {:8.6} s".format(time.time() - _t))
    pipeline = Pipeline()
    pipeline.run(FLAGS.images_dir, FLAGS.image_shape, FLAGS.model,
                 FLAGS.delete_cache, FLAGS.extract_only, FLAGS.visualize_only, FLAGS.overview, FLAGS.verbose)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gen_test",
        type=bool,
        default=True,
        help="Whether generate test images"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="Test",
        help="Path to test set"
    )
    parser.add_argument(
        "--image_shape",
        type=tuple,
        default=(64, 64),
        help="Image shape"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="v3",
        help="Model used to extract & predict"
    )
    parser.add_argument(
        "--delete_cache",
        type=bool,
        default=True,
        help="Whether delete cache"
    )
    parser.add_argument(
        "--extract_only",
        type=bool,
        default=False,
        help="Whether extract only"
    )
    parser.add_argument(
        "--visualize_only",
        type=bool,
        default=False,
        help="Whether visualize only"
    )
    parser.add_argument(
        "--overview",
        type=bool,
        default=True,
        help="Whether overview"
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Whether verbose"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
