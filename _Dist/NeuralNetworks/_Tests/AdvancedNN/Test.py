import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

from Util.Util import DataUtil
from _Dist.NeuralNetworks.c_BasicNN.NN import Basic
from _Dist.NeuralNetworks.e_AdvancedNN.NN import Advanced
from _Dist.NeuralNetworks._Tests.TestUtil import draw_acc

x_cv = y_cv = None
(x, y), (x_test, y_test) = DataUtil.gen_noisy_linear(one_hot=False)


def block_test(generator, **kwargs):
    (x_, y_), (x_test_, y_test_) = generator(**kwargs, one_hot=False)

    basic_ = Basic(**base_params).fit(x_, y_, x_test_, y_test_, snapshot_ratio=0)
    advanced_params["model_structure_settings"]["use_pruner"] = False
    wnd_dndf_ = Advanced(**advanced_params).fit(x_, y_, x_test_, y_test_, snapshot_ratio=0)
    advanced_params["model_structure_settings"]["use_pruner"] = True
    wnd_dndf_pruned_ = Advanced(**advanced_params).fit(x_, y_, x_test_, y_test_, snapshot_ratio=0)

    print("BasicNN              ", end="")
    basic_.evaluate(x_, y_, x_cv, y_cv, x_test_, y_test_)
    print("WnD & DNDF           ", end="")
    wnd_dndf_.evaluate(x_, y_, x_cv, y_cv, x_test_, y_test_)
    print("WnD & DNDF & Pruner  ", end="")
    wnd_dndf_pruned_.evaluate(x_, y_, x_cv, y_cv, x_test_, y_test_)

    return basic_, wnd_dndf_, wnd_dndf_pruned_


def block_test_pruner(generator, **kwargs):
    (x_, y_), (x_test_, y_test_) = generator(**kwargs, one_hot=False)

    advanced_params["model_structure_settings"]["use_pruner"] = True
    advanced_params["model_structure_settings"]["pruner_params"] = {"prune_method": "soft_prune"}
    soft_ = Advanced(**advanced_params).fit(x_, y_, x_test_, y_test_, snapshot_ratio=0)

    advanced_params["model_structure_settings"]["pruner_params"] = {
        "prune_method": "hard_prune",
        "alpha": 1e-8,
        "beta": 1e12
    }
    hard_ = Advanced(**advanced_params).fit(x_, y_, x_test_, y_test_, snapshot_ratio=0)

    advanced_params["model_structure_settings"]["pruner_params"] = {"prune_method": "surgery"}
    surgery_ = Advanced(**advanced_params).fit(x_, y_, x_test_, y_test_, snapshot_ratio=0)

    print("Soft     ", end="")
    soft_.evaluate(x_, y_, x_cv, y_cv, x_test_, y_test_)
    print("Hard     ", end="")
    hard_.evaluate(x_, y_, x_cv, y_cv, x_test_, y_test_)
    print("Surgery  ", end="")
    surgery_.evaluate(x_, y_, x_cv, y_cv, x_test_, y_test_)

    return soft_, hard_, surgery_


base_params = {
    "model_param_settings": {"n_epoch": 40, "metric": "acc"},
    "model_structure_settings": {"hidden_units": [500, 500]}
}
basic = Basic(**base_params).fit(x, y, x_test, y_test, snapshot_ratio=0)

numerical_idx = [True] * 100 + [False]
categorical_columns = []
advanced_params = {"data_info": {
    "numerical_idx": numerical_idx, "categorical_columns": categorical_columns
}}
advanced_params.update(base_params)
advanced_params["model_structure_settings"]["use_dndf"] = False
advanced_params["model_structure_settings"]["use_pruner"] = False

wnd = Advanced(**advanced_params).fit(x, y, x_test, y_test, snapshot_ratio=0)

advanced_params["model_structure_settings"]["use_dndf"] = True
wnd_dndf = Advanced(**advanced_params).fit(x, y, x_test, y_test, snapshot_ratio=0)

advanced_params["model_structure_settings"]["use_pruner"] = True
wnd_dndf_pruned = Advanced(**advanced_params).fit(x, y, x_test, y_test, snapshot_ratio=0)

print("BasicNN              ", end="")
basic.evaluate(x, y, x_cv, y_cv, x_test, y_test)
print("WnD                  ", end="")
wnd.evaluate(x, y, x_cv, y_cv, x_test, y_test)
print("WnD & DNDF           ", end="")
wnd_dndf.evaluate(x, y, x_cv, y_cv, x_test, y_test)
print("WnD & DNDF & Pruner  ", end="")
wnd_dndf_pruned.evaluate(x, y, x_cv, y_cv, x_test, y_test)
draw_acc(basic, wnd_dndf, wnd_dndf_pruned)

advanced_params["data_info"]["numerical_idx"] = [True] * 500 + [False]
advanced_params["model_structure_settings"]["pruner_params"] = {"prune_method": "hard_prune"}
basic, wnd_dndf, wnd_dndf_pruned = block_test(DataUtil.gen_noisy_linear, n_dim=500)
draw_acc(basic, wnd_dndf, wnd_dndf_pruned, ylim=(0.8, 0.95), draw_train=False)

basic, wnd_dndf, wnd_dndf_pruned = block_test(DataUtil.gen_noisy_linear, n_dim=500, noise_scale=1.)
draw_acc(basic, wnd_dndf, wnd_dndf_pruned, ylim=(0.7, 0.9), draw_train=False)

advanced_params["data_info"]["numerical_idx"] = [True] * 100 + [False]
basic, wnd_dndf, wnd_dndf_pruned = block_test(DataUtil.gen_noisy_linear, n_valid=100, noise_scale=0.)
draw_acc(basic, wnd_dndf, wnd_dndf_pruned, ylim=(0.95, 1.), draw_train=False)

advanced_params["model_structure_settings"]["pruner_params"] = {
    "prune_method": "soft_prune",
    "alpha": 0.01
}
basic, wnd_dndf, wnd_dndf_pruned = block_test(DataUtil.gen_noisy_linear, n_valid=100, noise_scale=0.)
draw_acc(basic, wnd_dndf, wnd_dndf_pruned, ylim=(0.95, 1.), draw_train=False)

for i, p in enumerate([3, 5, 8, 12]):
    basic, wnd_dndf, wnd_dndf_pruned = block_test(DataUtil.gen_noisy_poly, p=p)
    draw_acc(basic, wnd_dndf, wnd_dndf_pruned, ylim=(0.7-i*0.05, 0.9-i*0.05), draw_train=False)
