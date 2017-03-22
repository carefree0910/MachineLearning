import os
import sys
import shutil
import argparse
import numpy as np
import tensorflow as tf

from _Dist.ImageRecognition.ToolBox import Pipeline, Extractor
from NN.NN import NNDist


def fetch_data(name="_Data"):
    _img_paths, labels = [], []
    data_folder_lst = os.listdir(name)
    for i in range(len(data_folder_lst)-1, -1, -1):
        if data_folder_lst[i] == "_Cache":
            data_folder_lst.pop(i)
            break
    label_dic_dir = os.path.join(name, "_Cache", "LABEL_DIC")
    if not os.path.isfile(label_dic_dir):
        if not os.path.isdir(os.path.join(name, "_Cache")):
            os.makedirs(os.path.join(name, "_Cache"))
        np.save(label_dic_dir, np.array(data_folder_lst))
    for i, folder in enumerate(data_folder_lst):
        for img in os.listdir(os.path.join(name, folder)):
            _img_paths.append(os.path.join(name, folder, img))
            labels.append(i)
    labels = np.array(
        [[0 if i != yy else 1 for i in range(max(labels) + 1)] for yy in labels]
    )
    return _img_paths, labels


def main(_):
    if FLAGS.gen_test:
        if len(os.listdir("Test")) != 0:
            print("Test set already exists")
        else:
            print("Generating Test set...")
            _img_paths, labels = fetch_data()
            _indices = np.random.choice(len(labels), 200, replace=False)
            _img_paths = np.array(_img_paths)[_indices]
            labels = labels[_indices]
            for i, _path in enumerate(_img_paths):
                base_name = os.path.basename(_path)
                shutil.move(_path, os.path.join("Test", "{:04d}{}".format(i, base_name[base_name.rfind("."):])))
            np.save("Test/_answer", labels)
            print("Done")
    predictor_dir = os.path.join("Models", "Predictors", FLAGS.model, "Model.pb")
    if not os.path.isfile(predictor_dir):
        print("Predictor not found, training with images in '_Data' folder...")
        if not os.path.isfile("_Data/_Cache/features.npy") or not os.path.isfile("_Data/_Cache/labels.npy"):
            _img_paths, labels = fetch_data()
            extractor = Extractor(FLAGS.model, _img_paths)
            features = extractor.run()
            if not os.path.isdir("_Data/_Cache"):
                os.makedirs("_Data/_Cache")
            _indices = np.random.permutation(len(labels))
            features = features[_indices]
            labels = labels[_indices]
            np.save("_Data/_Cache/features", features)
            np.save("_Data/_Cache/labels", labels)
        else:
            features, labels = np.load("_Data/_Cache/features.npy"), np.load("_Data/_Cache/labels.npy")
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
