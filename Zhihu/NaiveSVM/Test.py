from Zhihu.SVM.Util import *
from Zhihu.SVM.Dev import *

np.random.seed(142857)  # for reproducibility


def main():

    svm = SVM()

    timing = Timing(enabled=True)
    timing_level = 1
    svm.feed_timing(timing)

    x, y = DataUtil.gen_spin(10)
    svm.fit(x, y)
    svm.evaluate()
    svm.visualize_2d()
    timing.show_timing_log(timing_level)

if __name__ == '__main__':
    main()
