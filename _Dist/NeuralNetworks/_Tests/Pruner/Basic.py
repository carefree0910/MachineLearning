import os
import sys
root_path = os.path.abspath("../../../../")
if root_path not in sys.path:
    sys.path.append(root_path)

from _Dist.NeuralNetworks.NNUtil import *
from _Dist.NeuralNetworks._Tests.Pruner.Base import Base


class Basic(Base):
    def __init__(self, *args, **kwargs):
        super(Basic, self).__init__(*args, **kwargs)
        self._name_appendix = "Basic"
        self.activations = self.hidden_units = None

    @property
    def name(self):
        return "NN" if self._name is None else self._name

    def init_model_param_settings(self):
        super(Basic, self).init_model_param_settings()
        self.activations = self.model_param_settings.get("activations", "relu")

    def init_model_structure_settings(self):
        super(Basic, self).init_model_structure_settings()
        self.hidden_units = self.model_structure_settings.get("hidden_units", [256, 256])

    def _build_layer(self, i, net):
        activation = self.activations[i]
        if activation is not None:
            net = getattr(Activations, activation)(net, "{}{}".format(activation, i))
        return net

    def _build_model(self, net=None):
        self._model_built = True
        if net is None:
            net = self._tfx
        current_dimension = net.shape[1].value
        if self.activations is None:
            self.activations = [None] * len(self.hidden_units)
        elif isinstance(self.activations, str):
            self.activations = [self.activations] * len(self.hidden_units)
        else:
            self.activations = self.activations
        for i, n_unit in enumerate(self.hidden_units):
            net = self._fully_connected_linear(net, [current_dimension, n_unit], i)
            net = self._build_layer(i, net)
            current_dimension = n_unit
        appendix = "_final_projection"
        fc_shape = self.hidden_units[-1] if self.hidden_units else current_dimension
        self._output = self._fully_connected_linear(net, [fc_shape, self.n_class], appendix)


if __name__ == '__main__':
    from Util.Util import DataUtil

    for generator in (DataUtil.gen_xor, DataUtil.gen_spiral, DataUtil.gen_nine_grid):
        x_train, y_train = generator(size=1000, one_hot=False)
        x_test, y_test = generator(size=100, one_hot=False)
        nn = Basic(model_param_settings={"n_epoch": 200}).scatter2d(x_train, y_train).fit(
            x_train, y_train, x_test, y_test, snapshot_ratio=0
        ).draw_losses().visualize2d(
            x_train, y_train, title="Train"
        ).visualize2d(
            x_test, y_test, padding=2, title="Test"
        )

    for size in (256, 1000, 10000):
        (x_train, y_train), (x_test, y_test) = DataUtil.gen_noisy_linear(
            size=size, n_dim=2, n_valid=2, test_ratio=100 / size, one_hot=False
        )
        nn = Basic(model_param_settings={"n_epoch": 200}).scatter2d(x_train, y_train).fit(
            x_train, y_train, x_test, y_test, snapshot_ratio=0
        ).draw_losses().visualize2d(x_train, y_train, title="Train").visualize2d(x_test, y_test, title="Test")
