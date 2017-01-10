from Zhihu.NaiveSVM.Util import *
from Zhihu.NaiveSVM.SVM import *

from sklearn.svm import SVC

np.random.seed(142857)  # for reproducibility


def main():

    svm = SVM()

    timing = Timing(enabled=True)
    timing_level = 1
    svm.feed_timing(timing)

    x, y = DataUtil.gen_spin(10, n=4)
    svm.fit(x, y, kernel="gaussian")
    svm.evaluate()
    svm.visualize_2d()
    timing.show_timing_log(timing_level)

    clf = SVC()
    clf.fit(x, y)

    plot_scale=2
    plot_precision=0.01
    plot_num = int(1 / plot_precision)
    xf = np.linspace(np.min(x) * plot_scale, np.max(x) * plot_scale, plot_num)
    yf = np.linspace(np.min(x) * plot_scale, np.max(x) * plot_scale, plot_num)
    input_x, input_y = np.meshgrid(xf, yf)
    input_xs = np.c_[input_x.ravel(), input_y.ravel()]
    output_ys_2d = clf.predict(input_xs).reshape(len(xf), len(yf))
    plt.contourf(input_x, input_y, output_ys_2d, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    main()
