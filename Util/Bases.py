class TimingBase:
    def feed_timing(self, timing):
        pass

    def show_timing_log(self, level=2):
        pass


class ClassifierBase:
    def __str__(self):
        pass

    def __repr__(self):
        pass

    def __getitem__(self, item):
        pass

    @staticmethod
    def acc(y, y_pred, weights):
        pass

    def estimate(self, x, y):
        pass

    def visualize2d(self, x, y, dense=100):
        pass

    def feed_timing(self, timing):
        pass

    def show_timing_log(self, level=2):
        pass
