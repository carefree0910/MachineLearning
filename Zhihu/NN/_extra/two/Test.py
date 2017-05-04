from Zhihu.NN._extra.two.Networks import *

from Util.Util import DataUtil

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

    nn.fit(x, y, epoch=epoch, verbose=2, metrics=["acc", "f1_score"], train_rate=0.8)
    nn.draw_logs()
    nn.visualize_2d(x, y)

    timing.show_timing_log(timing_level)

if __name__ == '__main__':
    main()
