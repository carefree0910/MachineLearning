import os
import sys
root_path = os.path.abspath("../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

from _Dist.NeuralNetworks.c_BasicNN.NN import Basic


# Transformation base
class TransformationBase(Basic):
    def __init__(self, *args, **kwargs):
        super(TransformationBase, self).__init__(*args, **kwargs)
        self._transform_ws = self._transform_bs = None

    def _get_all_data(self, shuffle=True):
        train = self._train_generator.get_all_data()
        if shuffle:
            np.random.shuffle(train)
        x, y = train[..., :-1], train[..., -1]
        if self._cv_generator is not None:
            cv = self._cv_generator.get_all_data()
            if shuffle:
                np.random.shuffle(cv)
            x_cv, y_cv = cv[..., :-1], cv[..., -1]
        else:
            x_cv = y_cv = None
        return x, y, x_cv, y_cv

    def _transform(self):
        pass

    def _print_model_performance(self, clf, name, x, y, x_cv, y_cv):
        print("\n".join(["=" * 60, "{} performance".format(name), "-" * 60]))
        y_train_pred = clf.predict(x)
        y_cv_pred = clf.predict(x_cv)
        train_metric = self._metric(y, y_train_pred)
        test_metric = self._metric(y_cv, y_cv_pred)
        print("{}  -  Train : {:8.6}   CV : {:8.6}".format(
            self._metric_name, train_metric, test_metric
        ))
        print("-" * 60)

    def _build_model(self):
        self._transform()
        super(TransformationBase, self)._build_model()

    def _initialize(self):
        super(TransformationBase, self)._initialize()
        self.feed_weights(self._transform_ws)
        self.feed_biases(self._transform_bs)
        x, y, x_cv, y_cv = self._get_all_data()
        print("\n".join(["=" * 60, "Initial performance", "-" * 60]))
        y_train_pred = self.predict(x)
        y_cv_pred = self.predict(x_cv)
        if self.n_class > 1:
            y_train_pred, y_cv_pred = y_train_pred.argmax(1), y_cv_pred.argmax(1)
        train_metric = self._metric(y, y_train_pred)
        test_metric = self._metric(y_cv, y_cv_pred)
        print("{}  -  Train : {:8.6}   CV : {:8.6}".format(
            self._metric_name, train_metric, test_metric
        ))
        print("-" * 60)


# NaiveBayes -> NN
class NB2NN(TransformationBase):
    def __init__(self, *args, **kwargs):
        super(NB2NN, self).__init__(*args, **kwargs)
        self.activation = None
        self.hidden_units = []

    def _transform(self):
        x, y, x_cv, y_cv = self._get_all_data()
        nb = MultinomialNB()
        nb.fit(x, y)
        self._print_model_performance(nb, "Naive Bayes", x, y, x_cv, y_cv)
        self._transform_ws = [nb.feature_log_prob_.T]
        self._transform_bs = [nb.class_log_prior_]


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


class DT2NN(TransformationBase):
    def __init__(self, *args, **kwargs):
        super(DT2NN, self).__init__(*args, **kwargs)
        if isinstance(self.activations, str):
            self.activations = [self.activations] * 2

    def _transform(self):
        x, y, x_cv, y_cv = self._get_all_data()
        tree = DecisionTreeClassifier()
        tree.fit(x, y)
        self._print_model_performance(tree, "Decision Tree", x, y, x_cv, y_cv)

        tree_structure = export_structure(tree)
        leafs = sum([1 if pair[1] == -1 else 0 for pair in tree_structure])
        internals = leafs - 1

        b = np.zeros(internals, dtype=np.float32)
        w1 = np.zeros([x.shape[1], internals], dtype=np.float32)
        w2 = np.zeros([internals, leafs], dtype=np.float32)
        w3 = np.zeros([leafs, self.n_class], dtype=np.float32)
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

        self._transform_ws = [w1, w2, w3]
        self._transform_bs = [b]
