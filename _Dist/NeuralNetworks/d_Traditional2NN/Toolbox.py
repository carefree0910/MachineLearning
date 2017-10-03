import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import _tree, DecisionTreeClassifier

import sys
sys.path.append("../../../")
from _Dist.NeuralNetworks.c_BasicNN.NNCore import NNCore
from _Dist.NeuralNetworks.c_BasicNN.NNWrapper import NNWrapper


# Transformation Core
class TransformationCore(NNCore):
    def __init__(self, numerical_idx, categorical_columns, n_classes,
                 model_param_settings=None, network_structure_settings=None, verbose_settings=None):
        super(TransformationCore, self).__init__(
            numerical_idx, categorical_columns, n_classes,
            model_param_settings, network_structure_settings, verbose_settings
        )
        self._transform_ws = self._transform_bs = None

    def _transform(self, x, y, x_test, y_test):
        pass

    def init_all_settings(self):
        NNCore.init_all_settings(self)
        self.use_embedding_for_deep = self.use_one_hot_for_deep = False

    def build_deep(self, x, y, x_test, y_test):
        self._transform_ws, self._transform_bs = self._transform(x, y, x_test, y_test)
        return NNCore.build_deep(self, x, y, x_test, y_test)

    def build_model(self, x, y, x_test, y_test, print_settings):
        NNCore.build_model(self, x, y, x_test, y_test, print_settings)
        self.feed_weights(self._transform_ws)
        self.feed_biases(self._transform_bs)
        print("\n".join(["=" * 60, "Initial performance", "-" * 60]))
        print("Train ", end="")
        self.evaluate(x, y, verbose=False)
        if x_test is not None and y_test is not None:
            print("Test  ", end="")
            self.evaluate(x_test, y_test, verbose=False)
        print("-" * 60)


# NaiveBayes -> NN
# noinspection PyTypeChecker
class NB2NNCore(TransformationCore):
    def init_all_settings(self):
        super(NB2NNCore, self).init_all_settings()
        self.activation_names = ["Linear"]
        self.hidden_units = []

    def feed_biases(self, bs):
        self._sess.run(self._central_bias[0].assign(bs[0]))

    def _transform(self, x, y, x_test, y_test):
        y_argmax = y.argmax(axis=1)
        y_test_argmax = y_test.argmax(axis=1)
        nb = MultinomialNB()
        nb.fit(x, y_argmax)
        print("\n".join(["=" * 60, "Naive Bayes performance", "-" * 60]))
        print("Train : ", np.mean(y_argmax == nb.predict(x)))
        print("Test  : ", np.mean(y_test_argmax == nb.predict(x_test)))
        print("-" * 60)
        return [nb.feature_log_prob_.T], [nb.class_log_prior_]


class NB2NNWrapper(NNWrapper):
    def __init__(self, name, numerical_idx, features_lists, core=NB2NNCore, **kwargs):
        super(NB2NNWrapper, self).__init__(name, numerical_idx, features_lists, core, **kwargs)


# DTree -> NN
def export_structure(tree):
    tree = tree.tree_

    def recurse(node, depth):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            feat = tree.feature[node]
            threshold = tree.threshold[node]
            yield depth, feat, threshold
            yield from recurse(tree.children_left[node], depth + 1)
            yield depth, feat, threshold
            yield from recurse(tree.children_right[node], depth + 1)
        else:
            yield depth, -1, tree.value[node]

    return list(recurse(0, 1))


# noinspection PyTypeChecker
class DT2NNCore(TransformationCore):
    def _transform(self, x, y, x_test, y_test):
        y_argmax = y.argmax(axis=1)
        y_test_argmax = y_test.argmax(axis=1)
        tree = DecisionTreeClassifier()
        tree.fit(x, y_argmax)
        print("\n".join(["=" * 60, "Decision tree performance", "-" * 60]))
        print("Train : ", np.mean(y_argmax == tree.predict(x)))
        print("Test  : ", np.mean(y_test_argmax == tree.predict(x_test)))
        print("-" * 60)

        tree_structure = export_structure(tree)
        leafs = sum([1 if pair[1] == -1 else 0 for pair in tree_structure])
        internals = leafs - 1

        b = np.zeros(internals, dtype=np.float32)
        w1 = np.zeros([x.shape[1], internals], dtype=np.float32)
        w2 = np.zeros([internals, leafs], dtype=np.float32)
        w3 = np.zeros([leafs, y.shape[1]], dtype=np.float32)
        node_list = []
        node_sign_list = []
        node_id_cursor = leaf_id_cursor = 0
        self.hidden_units = [internals, leafs]

        max_route_length = 0
        for depth, feat_dim, rs in tree_structure:
            depth -= 1
            if feat_dim == -1:
                valid_nodes = set()
                local_sign_list = node_sign_list[:]
                for i, ((node_id, node_dim, node_threshold), node_sign) in enumerate(
                    zip(node_list, node_sign_list)
                ):
                    new_w = node_sign
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
                    if new_w != 0:
                        valid_nodes.add((node_id, new_w))
                for node_id, node_sign in valid_nodes:
                    w2[node_id, leaf_id_cursor] = node_sign / len(valid_nodes)
                max_route_length = max(max_route_length, len(valid_nodes))
                w3[leaf_id_cursor] = rs / np.sum(rs)
                leaf_id_cursor += 1
            else:
                if depth == len(node_list):
                    node_sign_list.append(-1)
                    node_list.append([node_id_cursor, feat_dim, rs])
                    w1[feat_dim, node_id_cursor] = 1
                    b[node_id_cursor] = -rs
                    node_id_cursor += 1
                else:
                    node_list = node_list[:depth + 1]
                    node_sign_list = node_sign_list[:depth] + [1]
        w2 *= max_route_length

        return [w1, w2, w3], [b]


class DT2NNWrapper(NNWrapper):
    def __init__(self, name, numerical_idx, features_lists, core=DT2NNCore, **kwargs):
        super(DT2NNWrapper, self).__init__(name, numerical_idx, features_lists, core, **kwargs)
