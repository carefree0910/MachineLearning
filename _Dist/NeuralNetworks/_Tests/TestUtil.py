import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import matplotlib.pyplot as plt


def draw_acc(*models, ylim=(0.6, 1.05), draw_train=True):
    plt.figure()
    for model in models:
        name = str(model)
        metric = "acc" if "train_acc" in model.log else "binary_acc"
        el, tl = model.log["train_{}".format(metric)], model.log["test_{}".format(metric)]
        ee_base = np.arange(len(el))
        cse_base = np.linspace(0, len(el) - 1, len(tl))
        if draw_train:
            plt.plot(ee_base, el, label="Train acc ({})".format(name))
        plt.plot(cse_base, tl, label="Test acc ({})".format(name))
    plt.ylim(*ylim)
    plt.legend(prop={'size': 14})
    plt.show()
