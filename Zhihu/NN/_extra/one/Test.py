from Util.Util import DataUtil
from Zhihu.NN._extra.one.Networks import *

np.random.seed(142857)  # for reproducibility


def main():

    nn = NNDist()
    epoch = 1000

    timing = Timing(enabled=True)
    timing_level = 1
    nn.feed_timing(timing)

    x, y = DataUtil.gen_spiral(100)

    nn.add(ReLU((x.shape[1], 24)))
    nn.add(ReLU((24,)))
    nn.add(Softmax((y.shape[1],)))

    nn.fit(x, y, epoch=epoch)
    nn.visualize_2d(x, y)
    nn.evaluate(x, y)

    timing.show_timing_log(timing_level)

if __name__ == '__main__':
    main()
