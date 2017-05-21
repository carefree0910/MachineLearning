import numpy as np

from Util.Bases import ClassifierBase, TimingBase
from Util.Metas import SubClassTimingMeta, TimingMeta
from Util.Timing import Timing


class T1234(ClassifierBase):
    T1234Timing = Timing()

    @staticmethod
    @T1234Timing.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_results=False, **kwargs):
        y_pred = np.zeros(len(x))
        x_axis, y_axis = x.T
        x_lt_0, y_lt_0 = x_axis < 0, y_axis < 0
        x_gt_0, y_gt_0 = ~x_lt_0, ~y_lt_0
        y_pred[x_lt_0 & y_gt_0] = 1
        y_pred[x_lt_0 & y_lt_0] = 2
        y_pred[x_gt_0 & y_gt_0] = 3
        return y_pred


class Child(T1234, metaclass=SubClassTimingMeta):
    @staticmethod
    def test():
        for _ in range(10 ** 6):
            pass


class Test(TimingBase, metaclass=TimingMeta):
    @staticmethod
    def test1():
        for _ in range(10 ** 6):
            pass

    def test2(self):
        for _ in range(10 ** 6):
            _ = 1

if __name__ == '__main__':
    test = Test()
    n_call = 100
    for _ in range(n_call):
        test.test1()
        test.test2()
    test.show_timing_log()
