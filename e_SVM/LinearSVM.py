import torch
import numpy as np
import tensorflow as tf
from torch.autograd import Variable

from NN.TF.Optimizers import OptFactory as TFOptFac
from NN.PyTorch.Auto.Optimizers import OptFactory as PyTorchOptFac

from Util.Timing import Timing
from Util.ProgressBar import ProgressBar
from Util.Bases import ClassifierBase, TFClassifierBase, TorchAutoClassifierBase


class LinearSVM(ClassifierBase):
    LinearSVMTiming = Timing()

    def __init__(self, **kwargs):
        super(LinearSVM, self).__init__(**kwargs)
        self._w = self._b = None

        self._params["c"] = kwargs.get("c", 1)
        self._params["lr"] = kwargs.get("lr", 0.001)
        self._params["epoch"] = kwargs.get("epoch", 10 ** 4)
        self._params["tol"] = kwargs.get("tol", 1e-3)

    @LinearSVMTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, sample_weight=None, c=None, lr=None, epoch=None, tol=None, animation_params=None):
        if sample_weight is None:
            sample_weight = self._params["sample_weight"]
        if c is None:
            c = self._params["c"]
        if lr is None:
            lr = self._params["lr"]
        if epoch is None:
            epoch = self._params["epoch"]
        if tol is None:
            tol = self._params["tol"]
        *animation_properties, animation_params = self._get_animation_params(animation_params)
        x, y = np.atleast_2d(x), np.asarray(y)
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.asarray(sample_weight) * len(y)

        self._w = np.zeros(x.shape[1])
        self._b = 0
        ims = []
        bar = ProgressBar(max_value=epoch, name="LinearSVM")
        for i in range(epoch):
            err = (1 - self.predict(x, get_raw_results=True) * y) * sample_weight
            indices = np.random.permutation(len(y))
            idx = indices[np.argmax(err[indices])]
            if err[idx] <= tol:
                bar.update(epoch)
                break
            delta = lr * c * y[idx] * sample_weight[idx]
            self._w *= 1 - lr
            self._w += delta * x[idx]
            self._b += delta
            self._handle_animation(i, x, y, ims, animation_params, *animation_properties)
            bar.update()
        self._handle_mp4(ims, animation_properties)

    @LinearSVMTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False, **kwargs):
        rs = np.sum(self._w * x, axis=1) + self._b
        if not get_raw_results:
            return np.sign(rs)
        return rs


class TFLinearSVM(TFClassifierBase):
    TFLinearSVMTiming = Timing()

    def __init__(self, **kwargs):
        super(TFLinearSVM, self).__init__(**kwargs)
        self._w = self._b = None

        self._params["c"] = kwargs.get("c", 1)
        self._params["lr"] = kwargs.get("lr", 0.001)
        self._params["batch_size"] = kwargs.get("batch_size", 128)
        self._params["epoch"] = kwargs.get("epoch", 10 ** 4)
        self._params["tol"] = kwargs.get("tol", 1e-3)
        self._params["optimizer"] = kwargs.get("optimizer", "Adam")

    @TFLinearSVMTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, c=None, lr=None, batch_size=None, epoch=None, tol=None,
            optimizer=None, animation_params=None):
        if c is None:
            c = self._params["c"]
        if lr is None:
            lr = self._params["lr"]
        if batch_size is None:
            batch_size = self._params["batch_size"]
        if epoch is None:
            epoch = self._params["epoch"]
        if tol is None:
            tol = self._params["tol"]
        if optimizer is None:
            optimizer = self._params["optimizer"]
        *animation_properties, animation_params = self._get_animation_params(animation_params)
        x, y = np.atleast_2d(x), np.asarray(y)
        y_2d = y[..., None]

        self._w = tf.Variable(np.zeros([x.shape[1], 1]), dtype=tf.float32, name="w")
        self._b = tf.Variable(0., dtype=tf.float32, name="b")
        self._tfx = tf.placeholder(tf.float32, [None, x.shape[1]])
        self._tfy = tf.placeholder(tf.float32, [None, 1])
        self._y_pred_raw = tf.matmul(self._tfx, self._w) + self._b
        self._y_pred = tf.sign(self._y_pred_raw)
        loss = tf.reduce_sum(
            tf.nn.relu(1 - self._tfy * self._y_pred_raw)
        ) + c * tf.nn.l2_loss(self._w)
        train_step = TFOptFac().get_optimizer_by_name(optimizer, lr).minimize(loss)
        self._sess.run(tf.global_variables_initializer())
        bar = ProgressBar(max_value=epoch, name="TFLinearSVM")
        ims = []
        train_repeat = self._get_train_repeat(x, batch_size)
        for i in range(epoch):
            l = self.batch_training(x, y_2d, batch_size, train_repeat, loss, train_step)
            if l < tol:
                bar.update(epoch)
                break
            self._handle_animation(i, x, y, ims, animation_params, *animation_properties)
            bar.update()
        self._handle_mp4(ims, animation_properties)

    @TFLinearSVMTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False, **kwargs):
        rs = self._y_pred_raw if get_raw_results else self._y_pred
        return self._sess.run(rs, {self._tfx: x}).ravel()


class TorchLinearSVM(TorchAutoClassifierBase):
    TorchLinearSVMTiming = Timing()

    def __init__(self, **kwargs):
        super(TorchLinearSVM, self).__init__(**kwargs)
        self._w = self._b = None

        self._params["c"] = kwargs.get("c", 1)
        self._params["lr"] = kwargs.get("lr", 0.001)
        self._params["batch_size"] = kwargs.get("batch_size", 128)
        self._params["epoch"] = kwargs.get("epoch", 10 ** 4)
        self._params["tol"] = kwargs.get("tol", 1e-3)
        self._params["optimizer"] = kwargs.get("optimizer", "Adam")

    @TorchLinearSVMTiming.timeit(level=1, prefix="[Core] ")
    def _loss(self, y, y_pred, c):
        return torch.sum(
            torch.clamp(1 - y * y_pred, min=0)
        ) + c * torch.sqrt(torch.sum(self._w * self._w))

    @TorchLinearSVMTiming.timeit(level=1, prefix="[API] ")
    def fit(self, x, y, c=None, lr=None, batch_size=None, epoch=None, tol=None,
            optimizer=None, animation_params=None):
        if c is None:
            c = self._params["c"]
        if lr is None:
            lr = self._params["lr"]
        if batch_size is None:
            batch_size = self._params["batch_size"]
        if epoch is None:
            epoch = self._params["epoch"]
        if tol is None:
            tol = self._params["tol"]
        if optimizer is None:
            optimizer = self._params["optimizer"]
        *animation_properties, animation_params = self._get_animation_params(animation_params)
        x, y = np.atleast_2d(x), np.asarray(y, dtype=np.float32)
        y_2d = y[..., None]

        self._w = Variable(torch.rand([x.shape[1], 1]), requires_grad=True)
        self._b = Variable(torch.Tensor([0.]), requires_grad=True)
        self._model_parameters = [self._w, self._b]
        self._optimizer = PyTorchOptFac().get_optimizer_by_name(
            optimizer, self._model_parameters, lr, epoch
        )

        x, y, y_2d = self._arr_to_variable(False, x, y, y_2d)
        loss_function = lambda _y, _y_pred: self._loss(_y, _y_pred, c)

        bar = ProgressBar(max_value=epoch, name="TorchLinearSVM")
        ims = []
        train_repeat = self._get_train_repeat(x, batch_size)
        for i in range(epoch):
            self._optimizer.update()
            l = self.batch_training(
                x, y_2d, batch_size, train_repeat, loss_function
            )
            if l < tol:
                bar.update(epoch)
                break
            self._handle_animation(i, x, y, ims, animation_params, *animation_properties)
            bar.update()
        self._handle_mp4(ims, animation_properties)

    @TorchLinearSVMTiming.timeit(level=1, prefix="[API] ")
    def _predict(self, x, get_raw_results=False, **kwargs):
        if not isinstance(x, Variable):
            x = Variable(torch.from_numpy(np.asarray(x).astype(np.float32)))
        rs = x.mm(self._w)
        rs = rs.add_(self._b.expand_as(rs)).squeeze(1)
        if get_raw_results:
            return rs
        return torch.sign(rs)
