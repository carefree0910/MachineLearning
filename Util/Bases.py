class TimingBase:
    def __str__(self):
        pass

    def __repr__(self):
        pass

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
    def acc(y, y_pred, weights=None):
        pass

    def estimate(self, x, y):
        pass

    def visualize2d(self, x, y, margin=0.1, dense=200,
                    title=None, show_org=False, show_background=True, emphasize=None):
        pass

    def visualize3d(self, x, y, margin=0.1, dense=200,
                    title=None, show_org=False, show_background=True, emphasize=None):
        pass

    def feed_timing(self, timing):
        pass

    def show_timing_log(self, level=2):
        pass
