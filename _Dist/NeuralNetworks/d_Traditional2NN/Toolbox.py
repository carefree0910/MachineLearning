import os
import sys
root_path = os.path.abspath("../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import _tree, DecisionTreeClassifier

from _Dist.NeuralNetworks.c_BasicNN.NN import Basic


# Transformation base
class TransformationBase(Basic):
    def __init__(self, *args, **kwargs):
        super(TransformationBase, self).__init__(*args, **kwargs)
        self._transform_ws = self._transform_bs = None

    def _get_all_data(self, shuffle=True):
        train, train_weights = self._train_generator.get_all_data()
        if shuffle:
            np.random.shuffle(train)
        x, y = train[..., :-1], train[..., -1]
        if self._test_generator is not None:
            test, test_weights = self._test_generator.get_all_data()
            if shuffle:
                np.random.shuffle(test)
            x_test, y_test = test[..., :-1], test[..., -1]
        else:
            x_test = y_test = None
        return x, y, x_test, y_test

    def _transform(self):
        pass

    def _print_model_performance(self, clf, name, x, y, x_test, y_test):
        print("\n".join(["=" * 60, "{} performance".format(name), "-" * 60]))
        y_train_pred = clf.predict(x)
        y_test_pred = clf.predict(x_test)
        train_metric = self._metric(y, y_train_pred)
        test_metric = self._metric(y_test, y_test_pred)
        print("{}  -  Train : {:8.6}   CV : {:8.6}".format(
            self._metric_name, train_metric, test_metric
        ))
        print("-" * 60)

    def _build_model(self, net=None):
        self._transform()
        super(TransformationBase, self)._build_model(net)

    def _initialize(self):
        super(TransformationBase, self)._initialize()
        self.feed_weights()
        self.feed_biases()
        x, y, x_test, y_test = self._get_all_data()
        print("\n".join(["=" * 60, "Initial performance", "-" * 60]))
        self._evaluate(x, y, x_test, y_test)
        print("-" * 60)

    def feed_weights(self):
        for i, w in enumerate(self._transform_ws):
            if w is not None:
                self._sess.run(self._ws[i].assign(w))

    def feed_biases(self):
        for i, b in enumerate(self._transform_bs):
            if b is not None:
                self._sess.run(self._bs[i].assign(b))


# NaiveBayes -> NN
class NB2NN(TransformationBase):
    def __init__(self, *args, **kwargs):
        super(NB2NN, self).__init__(*args, **kwargs)
        self._name_appendix = "NaiveBayes"
        self.model_param_settings.setdefault("activations", None)

    def _transform(self):
        self.hidden_units = []
        x, y, x_test, y_test = self._get_all_data()
        nb = MultinomialNB()
        nb.fit(x, y)
        self._print_model_performance(nb, "Naive Bayes", x, y, x_test, y_test)
        self._transform_ws = [nb.feature_log_prob_.T]
        self._transform_bs = [nb.class_log_prior_]


# DTree -> NN
def export_structure(tree):
    tree = tree.tree_

    def recurse(node, depth):
        feature_dim = tree.feature[node]
        if feature_dim == _tree.TREE_UNDEFINED:
            yield depth, -1, tree.value[node]
        else:
            threshold = tree.threshold[node]
            yield depth, feature_dim, threshold
            yield from recurse(tree.children_left[node], depth + 1)
            yield depth, feature_dim, threshold
            yield from recurse(tree.children_right[node], depth + 1)

    return list(recurse(0, 0))


class DT2NN(TransformationBase):
    def __init__(self, *args, **kwargs):
        super(DT2NN, self).__init__(*args, **kwargs)
        self._name_appendix = "DTree"
        self.model_param_settings.setdefault("activations", ["sign", "one_hot"])

    def _transform(self):
        x, y, x_test, y_test = self._get_all_data()
        tree = DecisionTreeClassifier()
        tree.fit(x, y)
        self._print_model_performance(tree, "Decision Tree", x, y, x_test, y_test)

        tree_structure = export_structure(tree)
        n_leafs = sum([1 if pair[1] == -1 else 0 for pair in tree_structure])
        n_internals = n_leafs - 1

        print("Internals : {} ; Leafs : {}".format(n_internals, n_leafs))

        b = np.zeros(n_internals, dtype=np.float32)
        w1 = np.zeros([x.shape[1], n_internals], dtype=np.float32)
        w2 = np.zeros([n_internals, n_leafs], dtype=np.float32)
        w3 = np.zeros([n_leafs, self.n_class], dtype=np.float32)
        node_list = []
        node_sign_list = []
        node_id_cursor = leaf_id_cursor = 0
        max_route_length = 0
        self.hidden_units = [n_internals, n_leafs]

        for depth, feat_dim, rs in tree_structure:
            if feat_dim != -1:
                if depth == len(node_list):
                    node_sign_list.append(-1)
                    node_list.append([node_id_cursor, feat_dim, rs])
                    w1[feat_dim, node_id_cursor] = 1
                    b[node_id_cursor] = -rs
                    node_id_cursor += 1
                else:
                    node_list = node_list[:depth + 1]
                    node_sign_list = node_sign_list[:depth] + [1]
            else:
                valid_nodes = set()
                local_sign_list = node_sign_list[:]
                for i, ((node_id, node_dim, node_threshold), node_sign) in enumerate(
                    zip(node_list, node_sign_list)
                ):
                    valid_nodes.add((node_id, node_sign))
                    if i >= 1:
                        for j, ((local_id, local_dim, local_threshold), local_sign) in enumerate(zip(
                            node_list[:i], local_sign_list[:i]
                        )):
                            if node_sign == local_sign and node_dim == local_dim:
                                if (
                                    (node_sign == -1 and node_threshold < local_threshold) or
                                    (node_sign == 1 and node_threshold > local_threshold)
                                ):
                                    local_sign_list[j] = 0
                                    valid_nodes.remove((local_id, local_sign))
                                    break
                for node_id, node_sign in valid_nodes:
                    w2[node_id, leaf_id_cursor] = node_sign / len(valid_nodes)
                max_route_length = max(max_route_length, len(valid_nodes))
                w3[leaf_id_cursor] = rs / np.sum(rs)
                leaf_id_cursor += 1

        w2 *= max_route_length
        self._transform_ws = [w1, w2, w3]
        self._transform_bs = [b]


if __name__ == '__main__':
    from _Dist.NeuralNetworks.c_BasicNN.NN import Basic
    from _Dist.NeuralNetworks.e_AdvancedNN.NN import Advanced

    with open("../../../_Data/madelon.txt", "r") as file:
        data = [line.strip().split() for line in file]
        train, test = np.array(data[:2000], np.float32), np.array(data[2000:], np.float32)
        np.random.shuffle(train)
    x, y = train[..., :-1], train[..., -1]
    x_test, y_test = test[..., :-1], test[..., -1]
    x -= x.mean()
    x /= x.std()
    x_test -= x_test.mean()
    x_test /= x_test.std()

    Advanced(
        name="madelon",
        data_info={
            "numerical_idx": [True] * 500 + [False],
            "categorical_columns": []
        },
        model_param_settings={
            "lr": 1e-3,
            "keep_prob": 0.25,
            "use_batch_norm": False,
            "activations": ["relu", "relu"]
        }, model_structure_settings={
            "use_pruner": False,
            "use_wide_network": False,
            "hidden_units": [152, 153]
        }
    ).fit(x, y, x_test, y_test, snapshot_ratio=1)
