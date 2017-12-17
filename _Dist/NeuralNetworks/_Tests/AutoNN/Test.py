import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

from _Dist.NeuralNetworks.f_AutoNN.NN import AutoAdvanced


AutoAdvanced(
    "Adult",
    model_param_settings={"n_epoch": 3},
    data_info={"file_type": "csv", "data_folder": "../_Data"}
).fit(snapshot_ratio=0).fit(snapshot_ratio=0).save()
AutoAdvanced("Adult").load().fit(snapshot_ratio=0)
