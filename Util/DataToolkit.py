import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


# noinspection PyTypeChecker
class DataToolkit:
    def __init__(self, data):
        self._data = np.asarray(data)
        self._sorted_data = np.sort(self._data)
        self._n = len(self._data)
        self._mean = self._variance = self._std = None
        self._moments = []
        self._q1 = self._q3 = None

    def get_moment(self, k):
        if len(self._moments) < k:
            self._moments += [None] * (k - len(self._moments))
        if self._moments[k-1] is None:
            self._moments[k-1] = np.sum((self._data - self.mean) ** k) / self._n
        return self._moments[k-1]

    def get_mp(self, p):
        _np = self._n * p
        int_np = int(_np)
        if not int(_np % 1):
            return self._sorted_data[int_np]
        return 0.5 * (self._sorted_data[int_np-1] + self._sorted_data[int_np])

    @property
    def min(self):
        return self._sorted_data[0]

    @property
    def max(self):
        return self._sorted_data[-1]

    @property
    def mean(self):
        if self._mean is None:
            self._mean = self._data.mean()
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            self._variance = np.sum((self._data - self.mean) ** 2) / (self._n - 1)
        return self._variance

    @property
    def std(self):
        if self._std is None:
            self._std = (np.sum((self._data - self.mean) ** 2) / (self._n - 1)) ** 0.5
        return self._std

    @property
    def g1(self):
        n, moment3 = self._n, self.get_moment(3)
        return n ** 2 * moment3 / ((n - 1) * (n - 2) * self.std ** 3)

    @property
    def g2(self):
        n, moment4 = self._n, self.get_moment(4)
        return n**2*(n+1)*moment4 / ((n-1)*(n-2)*(n-3)*self.std**4) - 3*(n-1)**2/((n-2)*(n-3))

    @property
    def med(self):
        n, hn = self._n, int(self._n*0.5)
        if n & 1:
            return self._sorted_data[hn-1]
        return 0.5 * (self._sorted_data[hn-1] + self._sorted_data[hn])

    @property
    def q1(self):
        if self._q1 is None:
            self._q1 = self.get_mp(0.25)
        return self._q1

    @property
    def q3(self):
        if self._q3 is None:
            self._q3 = self.get_mp(0.75)
        return self._q3

    @property
    def r(self):
        return self._sorted_data[-1] - self._sorted_data[0]

    @property
    def r1(self):
        return self.q3 - self.q1

    @property
    def trimean(self):
        return 0.25 * (self.q1 + self.q3) + 0.5 * self.med

    @property
    def loval(self):
        return self.q1 - 1.5 * self.r1

    @property
    def hival(self):
        return self.q3 + 1.5 * self.r1

    def draw_histogram(self, bin_size=10):
        bins = np.arange(self._sorted_data[0]-self.r1, self._sorted_data[-1]+self.r1, bin_size)
        plt.hist(self._data, bins=bins, alpha=0.5)
        plt.title("Histogram (bin_size: {})".format(bin_size))
        plt.show()

    def qq_plot(self):
        stats.probplot(self._data, dist="norm", plot=plt)
        plt.show()

    def box_plot(self):
        plt.figure()
        plt.boxplot(self._data, vert=False, showmeans=True)
        plt.show()

if __name__ == '__main__':
    toolkit = DataToolkit([
        53, 70.2, 84.3, 55.3, 78.5, 63.5, 71.4, 53.4, 82.5, 67.3, 69.5, 73, 55.7, 85.8, 95.4, 51.1, 74.4,
        54.1, 77.8, 52.4, 69.1, 53.5, 64.3, 82.7, 55.7, 70.5, 87.5, 50.7, 72.3, 59.5
    ])
    print("mean     : ", toolkit.mean)
    print("variance : ", toolkit.variance)
    print("g1       : ", toolkit.g1)
    print("g2       : ", toolkit.g2)
    print("med      : ", toolkit.med)
    print("r        : ", toolkit.r)
    print("q3       : ", toolkit.q3)
    print("q1       : ", toolkit.q1)
    print("r1       : ", toolkit.r1)
    print("trimean  : ", toolkit.trimean)
    print("hival    : ", toolkit.hival)
    print("loval    : ", toolkit.loval)
    print("min      : ", toolkit.min)
    print("max      : ", toolkit.max)
    toolkit.draw_histogram()
    toolkit.qq_plot()
    toolkit.box_plot()
