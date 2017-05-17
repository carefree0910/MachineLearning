import numpy as np


# Generator Framework
class Generator:
    def __init__(self, im=None, om=None, **kwargs):
        self._im, self._om = im, om
        self._x_train = self._x_test = None
        self._y_train = self._y_test = None

    def gen(self, batch, test=False, **kwargs):
        if batch == 0:
            if test:
                return self._x_test, self._y_test
            return self._x_train, self._y_train
        batch = np.random.choice(len(self._x_train), batch)
        return self._x_train[batch], self._y_train[batch]
