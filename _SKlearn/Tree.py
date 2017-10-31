import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

from Util.Bases import ClassifierBase
from Util.Metas import SKCompatibleMeta

from sklearn.tree import _tree, DecisionTreeClassifier


class SKTree(DecisionTreeClassifier, ClassifierBase, metaclass=SKCompatibleMeta):
    def export_structure(self):
        tree = self.tree_

        def recurse(node, depth):
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                feat = tree.feature[node]
                threshold = tree.threshold[node]
                yield depth, feat, threshold
                for pair in recurse(tree.children_left[node], depth + 1):
                    yield pair
                yield depth, feat, threshold
                for pair in recurse(tree.children_right[node], depth + 1):
                    yield pair
            else:
                yield depth, -1, tree.value[node]

        return [pair for pair in recurse(0, 1)]

    def print_structure(self):
        tree = self.tree_
        feature_names = ["x", "y"]

        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree.feature
        ]
        print("def tree({}):".format(", ".join(feature_names)))

        def recurse(node, depth):
            indent = "  " * depth
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree.threshold[node]
                print("{}if {} <= {}:".format(indent, name, threshold))
                recurse(tree.children_left[node], depth + 1)
                print("{}else:  # if {} > {}".format(indent, name, threshold))
                recurse(tree.children_right[node], depth + 1)
            else:
                print("{}return {}".format(indent, tree.value[node]))
        recurse(0, 1)
